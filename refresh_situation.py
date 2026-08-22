"""Pull the National Dengue Control Unit weekly update into a current-situation table.

Deliberately NOT merged into the modelling series. NDCU counts suspected cases through
NaDSys; the Epidemiology Unit's WER counts Medical-Officer-of-Health notifications.
On overlapping weeks they disagree in both directions (Gampaha +35%, Kalutara -16%),
so this is a different measurement, not a fresher copy of the same one. Splicing it
onto the end of the training series would put a definitional break exactly where the
model reads recent incidence.

What it is good for is telling a user what is happening *now*: NDCU publishes about a
week behind, against roughly nine weeks for the WER.

    python refresh_situation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import pandas as pd

from dengue import ndcu as N
from dengue.srilanka import DISTRICTS

ROOT = Path(__file__).parent
OUT = ROOT / "reports" / "srilanka_latest_situation.csv"
META = ROOT / "reports" / "situation_freshness.json"
WEEKS = 12


def main() -> int:
    print("fetching NDCU weekly updates ...", flush=True)
    reports = N.list_reports(limit=WEEKS + 4)
    frames = []
    for rep in reports[:WEEKS]:
        try:
            df = N.parse(N.fetch(rep))
        except Exception as e:
            print(f"  {rep['year']} wk {rep['week']}: {type(e).__name__} {str(e)[:50]}")
            continue
        if not df.empty:
            frames.append(df)
            print(f"  wk {rep['week']:>2} -> {df.week_start.iloc[0].date()}  "
                  f"{len(df)} districts, {int(df.cases.sum()):,} cases", flush=True)
    if not frames:
        print("nothing parsed - leaving the existing situation table alone")
        return 1

    allp = pd.concat(frames, ignore_index=True)
    check = N.validate(allp)
    print(f"\nself-check against each report's own stated total: "
          f"{check['passed']}/{check['checked']} -> {check['verdict']}")
    if check["verdict"] == "FAIL":
        print("REJECTED - not writing")
        return 1

    latest = allp.week_start.max()
    cur = allp[allp.week_start == latest].copy()

    # week-on-week direction comes from the report's own previous-week column, so it
    # is internally consistent even if an earlier issue failed to parse
    cur["change"] = cur.cases - cur.prev_week
    cur["change_pct"] = cur.change / cur.prev_week.clip(lower=1) * 100
    cur["population"] = cur.district.map(lambda d: DISTRICTS.get(d, (0, 0, 0, 0))[2])
    cur["inc100k"] = cur.cases / cur.population.clip(lower=1) * 1e5
    cur["lat"] = cur.district.map(lambda d: DISTRICTS.get(d, (0, 0, 0, 0))[0])
    cur["lon"] = cur.district.map(lambda d: DISTRICTS.get(d, (0, 0, 0, 0))[1])

    keep = ["district", "week_start", "week_end", "cases", "prev_week", "change",
            "change_pct", "cumulative", "population", "inc100k", "lat", "lon"]
    cur[keep].sort_values("cases", ascending=False).to_csv(OUT, index=False)

    # a short trailing history per district for sparklines
    hist = allp[["district", "week_start", "cases"]].sort_values(["district", "week_start"])
    hist.to_csv(ROOT / "reports" / "srilanka_situation_history.csv", index=False)

    now = datetime.now(timezone.utc)
    META.write_text(json.dumps({
        "source": "National Dengue Control Unit, Weekly Dengue Update",
        "refreshed_at": str(now), "latest_week_start": str(latest.date()),
        "latest_week_end": str(cur.week_end.iloc[0].date()),
        "age_days": (pd.Timestamp(now.date()) - latest).days,
        "weeks_parsed": int(allp.week_start.nunique()),
        "self_check": check["verdict"], "national_total": int(cur.cases.sum()),
    }, indent=2, default=str))

    print(f"\nlatest week {latest.date()}..{cur.week_end.iloc[0].date()}  "
          f"({(pd.Timestamp(now.date()) - latest).days} days old)")
    print(f"national total {int(cur.cases.sum()):,} cases across {len(cur)} districts")
    print(cur.nlargest(6, "cases")[["district", "cases", "prev_week", "change_pct"]]
          .round(1).to_string(index=False))
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
