"""Refresh Sri Lankan surveillance data and regenerate the outbreak forecasts.

Two sources, in order of trust:

  1. denguedatahub  - tidy, structured, reliable, but lags the publisher by months.
  2. WER PDFs       - the Epidemiology Unit's weekly reports, current to within a
                      few weeks, parsed directly.

The PDF parser is never trusted on faith. Every run re-parses weeks that
denguedatahub already covers and compares them; if the agreement drops below the
threshold the WER data is discarded and the refresh falls back to denguedatahub
alone rather than writing numbers it cannot vouch for.

Safe to run repeatedly - it is incremental and idempotent.

    python refresh_data.py            # refresh data + forecasts
    python refresh_data.py --check    # report freshness, change nothing
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import pandas as pd
import requests

from dengue import wer as W

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
SURVEILLANCE = RAW / "srilanka_weekly_district.csv"
HUB_URL = (
    "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/"
    "data-raw/srilanka_weekly_data.csv"
)
FRESHNESS = REPORTS / "data_freshness.json"

MIN_AGREEMENT = 0.95  # WER is rejected below this match rate against the hub
MAX_NEW_REPORTS = 20  # cap PDFs fetched per run

# --check answers with an exit status so CI can branch on it.
EXIT_CURRENT = 0  # local data is up to date
EXIT_STALE = 10   # newer surveillance data is available


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_hub() -> pd.DataFrame:
    """denguedatahub district-weekly surveillance."""
    r = requests.get(HUB_URL, timeout=180)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d["week_start"] = pd.to_datetime(d["start.date"], format="mixed", errors="coerce")
    return d.dropna(subset=["week_start"])


def harvest_wer(after: pd.Timestamp, hub: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse WER reports, validating against the weeks denguedatahub already has."""
    try:
        reports = W.list_reports(limit=MAX_NEW_REPORTS + 12)
    except Exception as e:
        log(f"  WER index unreachable ({type(e).__name__}); using denguedatahub only")
        return pd.DataFrame(), {"verdict": "INDEX UNREACHABLE"}

    frames, fetched = [], 0
    for rep in reports:
        if fetched >= MAX_NEW_REPORTS:
            break
        try:
            df = W.parse(W.fetch(rep))
        except Exception as e:
            log(f"    Vol {rep['vol']} No {rep['no']}: {type(e).__name__} {str(e)[:50]}")
            continue
        if df.empty:
            continue
        frames.append(df)
        fetched += 1
        log(
            f"    Vol {rep['vol']} No {rep['no']:>2} -> week "
            f"{df.week_start.iloc[0].date()} ({len(df)} districts)"
        )

    if not frames:
        return pd.DataFrame(), {"verdict": "NOTHING PARSED"}

    parsed = pd.concat(frames, ignore_index=True).drop_duplicates(["district", "week_start"])
    check = W.validate_against(parsed, hub)
    log(
        f"  validation vs denguedatahub: {check['overlapping_rows']} overlapping rows, "
        f"exact match {check.get('exact_match_rate', 0):.1%} -> {check['verdict']}"
    )

    if check.get("within_tolerance_rate", 0) < MIN_AGREEMENT:
        log("  WER parse REJECTED - falling back to denguedatahub alone")
        return pd.DataFrame(), check

    new = parsed[parsed.week_start > after]
    log(
        f"  {len(new)} district-weeks newer than denguedatahub "
        f"({new.week_start.min().date() if len(new) else '-'} -> "
        f"{new.week_start.max().date() if len(new) else '-'})"
    )
    return new, check


def merge(hub: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """denguedatahub, extended with WER weeks it does not yet cover."""
    if new.empty:
        return hub
    add = pd.DataFrame(
        {
            "year": new.week_start.dt.isocalendar().year.astype(int),
            "week": new.week_start.dt.isocalendar().week.astype(int),
            "start.date": new.week_start.dt.strftime("%m/%d/%Y"),
            "end.date": new.week_end.dt.strftime("%m/%d/%Y"),
            "district": new.district,
            "cases": new.cases.astype(int),
        }
    )
    keep = ["year", "week", "start.date", "end.date", "district", "cases"]
    out = pd.concat([hub[keep], add], ignore_index=True)
    out["_ws"] = pd.to_datetime(out["start.date"], format="mixed", errors="coerce")
    return (
        out.dropna(subset=["_ws"])
        .sort_values(["district", "_ws"])
        .drop_duplicates(["district", "_ws"], keep="first")
        .drop(columns="_ws")
    )


def extend_weather(until: pd.Timestamp) -> pd.Timestamp:
    """Top up the NASA POWER cache so it covers every surveillance week."""
    from dengue.srilanka import DISTRICTS, WEATHER_CACHE, _from_nasa_power

    have = pd.read_parquet(WEATHER_CACHE) if WEATHER_CACHE.exists() else pd.DataFrame()
    if not have.empty:
        have["time"] = pd.to_datetime(have["time"])
        current = have.time.max()
        if current >= until:
            log(f"  weather already covers {current.date()}")
            return current
    else:
        current = pd.Timestamp("2006-01-01")

    start = (current + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (until + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    log(f"  extending weather {start} -> {end} for {len(DISTRICTS)} districts")
    frames = []
    for name, (lat, lon, _, _) in DISTRICTS.items():
        try:
            frames.append(_from_nasa_power(name, lat, lon, start, end))
        except Exception as e:
            log(f"    {name}: {type(e).__name__} {str(e)[:40]}")
    if not frames:
        return current
    fresh = pd.concat(frames, ignore_index=True)
    fresh["time"] = pd.to_datetime(fresh["time"])
    out = (
        pd.concat([have, fresh], ignore_index=True)
        .drop_duplicates(["district", "time"], keep="last")
        .sort_values(["district", "time"])
    )
    out.to_parquet(WEATHER_CACHE, index=False)
    return out.time.max()


def rebuild_forecasts() -> dict:
    """Regenerate continuation + emergence forecasts from the refreshed panel.

    Both scripts expose main(panel), so they are imported and called rather than
    executed with runpy - which was only ever needed because they did all their
    work at module import. The panel is built ONCE here and handed to both; it
    used to be rebuilt per script, in one process, against a documented memory
    ceiling.
    """
    import finalize_emergence
    import finalize_srilanka
    from dengue.srilanka import build_panel

    log("\n[5/5] regenerating forecasts")
    panel = build_panel()
    log(f"  panel built once: {len(panel):,} district-weeks")

    log("  continuation -> models/srilanka_outbreak.joblib")
    finalize_srilanka.main(panel)
    log("  emergence    -> models/srilanka_emergence.joblib")
    finalize_emergence.main(panel)

    dual = pd.read_csv(REPORTS / "srilanka_dual_risk.csv")
    return {
        "forecast_week": str(pd.to_datetime(dual.week_start).max().date()),
        "districts": len(dual),
    }


def refresh_situation() -> None:
    """Refresh the NDCU situation tables.

    Wired in because the app renders the NDCU freshness stamp: leaving this script
    to be run by hand meant that stamp went stale while the freshness file beside
    it kept advancing. It is supplementary to the forecast, so a failure here is
    logged and the refresh continues - the WER/denguedatahub path is the one the
    forecasts actually depend on.
    """
    log("\n[4/5] NDCU situation update")
    try:
        import refresh_situation as situation

        if situation.main() == 0:
            log("  situation tables updated")
        else:
            log("  NDCU parse produced nothing usable - existing tables left alone")
    except Exception as e:
        log(f"  NDCU refresh failed ({type(e).__name__}: {str(e)[:80]}) - continuing")


def refresh_enso() -> str | None:
    """Top up the cached Nino 3.4 series from NOAA.

    Fails soft for the same reason `refresh_situation` does, and more pointedly: the
    forecast path must not acquire a dependency on a foreign agency's web server
    being up on a Tuesday. The cached series is committed, so a failure here leaves
    the model running on last week's ENSO values rather than on none.
    """
    log("\n[4b/5] ENSO index")
    try:
        from dengue.enso import fetch_nino34

        df = fetch_nino34(force=True)
        through = f"{int(df.year.iloc[-1])}-{int(df.month.iloc[-1]):02d}"
        log(f"  nino3.4 covers through {through}")
        return through
    except Exception as e:
        log(f"  ENSO refresh failed ({type(e).__name__}: {str(e)[:80]}) - continuing")
        return None


def write_freshness(**kw) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FRESHNESS.write_text(json.dumps(kw, indent=2, default=str))


def main(check_only: bool) -> int:
    now = datetime.now(UTC)
    log(f"DengueShield data refresh - {now:%Y-%m-%d %H:%M} UTC\n")

    log("[1/5] denguedatahub")
    hub = fetch_hub()
    hub_last = hub.week_start.max()
    log(f"  {len(hub):,} rows, latest week {hub_last.date()}")

    local_last = None
    if SURVEILLANCE.exists():
        cur = pd.read_csv(SURVEILLANCE)
        cur["_ws"] = pd.to_datetime(cur["start.date"], format="mixed", errors="coerce")
        local_last = cur["_ws"].max()
        log(f"  local file latest week {local_last.date()}")

    log("\n[2/5] Epidemiology Unit weekly reports")
    new, check = harvest_wer(hub_last, hub)

    latest = max([hub_last] + ([new.week_start.max()] if len(new) else []))
    age_days = (pd.Timestamp(now.date()) - latest).days
    log(f"\n  latest available surveillance week: {latest.date()} ({age_days} days old)")

    if check_only:
        # A check reports; it does not write. The freshness file is a record of a
        # refresh, and rewriting it here made --check a side-effecting operation.
        current = local_last is not None and latest <= local_last
        if current:
            log("\nAlready current - nothing to do.")
        else:
            log(
                f"\nNewer data available (local "
                f"{local_last.date() if local_last is not None else '-'}"
                f" -> {latest.date()}). Run without --check to apply."
            )
        # Exit status, not log prose: CI used to grep stdout for a sentence, so
        # rewording a log line silently inverted the branch.
        return EXIT_CURRENT if current else EXIT_STALE

    log("\n[3/5] merging and topping up weather")
    merged = merge(hub, new)
    SURVEILLANCE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(SURVEILLANCE, index=False)
    log(f"  wrote {len(merged):,} rows -> {SURVEILLANCE.relative_to(ROOT)}")
    weather_to = extend_weather(latest)
    log(f"  weather covers through {pd.Timestamp(weather_to).date()}")

    refresh_situation()
    enso_through = refresh_enso()
    info = rebuild_forecasts()
    write_freshness(
        enso_through=enso_through,
        refreshed_at=now,
        latest_week=latest,
        age_days=age_days,
        hub_latest=hub_last,
        wer_weeks_added=len(new),
        wer_validation=check.get("verdict"),
        wer_exact_match_rate=check.get("exact_match_rate"),
        weather_through=pd.Timestamp(weather_to),
        **info,
    )
    log(
        f"\nDone. Forecast week {info['forecast_week']}, "
        f"{info['districts']} districts. Freshness -> {FRESHNESS.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="report freshness without changing anything"
    )
    raise SystemExit(main(ap.parse_args().check))
