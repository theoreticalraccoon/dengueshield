"""Can emergence be given more to learn from? Paired, over folds.

Emergence is the weakest of the four models (PR-AUC 0.405 against a persistence
baseline of 0.250) and the one with the least data: the eligibility rule throws away
every district-week that is currently in outbreak, leaving ~18k training rows with
~4% positives. Feature work does not move it - neighbour features and re-tuning both
lose ground (docs/adr/0003). So these are changes to what it trains on and what it
is asked, not to its columns.

  A  production                  eligible rows only, binary 4-week label
  B  +ineligible rows            train on every district-week with an eligibility
                                 flag, evaluate on eligible rows only. Ten times the
                                 positives, but most of the added rows are trivially
                                 positive (a district already in outbreak is still in
                                 outbreak next week), so this may well teach
                                 "high incidence -> yes", which is the wrong lesson
                                 for a quiet district. That is the experiment.
  C  discrete-time hazard        one row per (district-week, week-ahead k), asking
                                 "does it cross in exactly k weeks?", then combine
                                 the four hazards into a 4-week probability. The
                                 binary label collapses a hazard into a lump; this
                                 asks the question the data is actually shaped like.
  D  B + C                       both.

Evaluation is identical for all four: PR-AUC on the eligible rows of the held-out
year, so the comparison is like for like. Selection is paired over folds; the locked
test years are never touched.

    .venv/Scripts/python.exe experiments/accuracy_v2/search_emergence.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from dengue.config import EMERGENCE_HORIZON, LGBM_PARAMS_SL, SEED
from dengue.emergence import (
    HISTORY_FEATURES,
    add_base_features,
    add_history_features,
    label_emergence,
)
from dengue.metrics import pos_weight
from dengue.srilanka import COMMON_FEATURES, build_panel
from dengue.validation import compare, fit_predict_seeds, rolling_origin

PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600
FIRST_TEST_YEAR = 2016


def hazard_frame(scored: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """One row per (district-week, k) for k in 1..H: does it cross in exactly week k?

    The risk set shrinks as k grows - a district that has already crossed by week 2
    is not at risk of crossing for the first time in week 3 - which is the whole
    point of a hazard formulation and what the lumped binary label cannot express.
    """
    g = scored.sort_values(["district", "week_start"]).groupby("district", sort=False)
    above = {k: g["_above"].shift(-k) for k in range(1, EMERGENCE_HORIZON + 1)}

    rows = []
    for k in range(1, EMERGENCE_HORIZON + 1):
        crossed_before = np.zeros(len(scored), dtype=bool)
        for j in range(1, k):
            crossed_before |= above[j].fillna(0).to_numpy() >= 1
        at_risk = ~crossed_before & above[k].notna().to_numpy()

        part = scored.loc[at_risk, [*feats, "anio", "eligible"]].copy()
        part["y"] = (above[k].to_numpy()[at_risk] >= 1).astype(float)
        part["k"] = k
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def fold_scores(panel: pd.DataFrame, feats: list[str], arm: str) -> list[float]:
    """PR-AUC per fold on the ELIGIBLE rows of the held-out year, whatever the arm."""
    scored = label_emergence(panel, keep_ineligible=True)
    labelled = scored.dropna(subset=["y"])
    hazard = hazard_frame(scored, feats) if arm in ("C", "D") else None

    out = []
    for year, _train, _test in rolling_origin(labelled, first_test_year=FIRST_TEST_YEAR):
        # The evaluation set never varies: eligible rows of the held-out year.
        te = labelled[(labelled.anio == year) & labelled.eligible]
        if len(te) == 0 or te.y.sum() == 0:
            continue

        if arm in ("C", "D"):
            tr = hazard[hazard.anio <= year - 1]
            if arm == "C":
                tr = tr[tr.eligible]
            params = {**PARAMS, "scale_pos_weight": pos_weight(tr.y.values)}
            hz_feats = [*feats, "k"]
            block = pd.concat(
                [te.assign(k=k)[hz_feats] for k in range(1, EMERGENCE_HORIZON + 1)],
                ignore_index=True,
            )
            flat = fit_predict_seeds(
                params, tr, block.assign(y=0.0), hz_feats, num_boost_round=ROUNDS
            )
            per_k = flat.reshape(EMERGENCE_HORIZON, len(te))
            p = 1.0 - np.prod(1.0 - np.clip(per_k, 0.0, 1.0), axis=0)
        else:
            tr = labelled[labelled.anio <= year - 1]
            use = [*feats, "eligible"] if arm == "B" else feats
            if arm == "A":
                tr = tr[tr.eligible]
            params = {**PARAMS, "scale_pos_weight": pos_weight(tr.y.values)}
            p = fit_predict_seeds(params, tr, te, use, num_boost_round=ROUNDS)

        out.append(average_precision_score(te.y.values, p))
    return out


def main() -> int:
    panel = add_history_features(add_base_features(build_panel()))
    feats = [c for c in COMMON_FEATURES if c in panel.columns]
    feats += [c for c in HISTORY_FEATURES if c in panel.columns]

    base = fold_scores(panel, feats, "A")
    print(f"folds={len(base)}  features={len(feats)}")
    print(f"  A production (eligible rows only)   {np.mean(base):.4f}\n")

    rows = []
    for arm, label in (
        ("B", "B +ineligible rows as context"),
        ("C", "C discrete-time hazard"),
        ("D", "D hazard + ineligible rows"),
    ):
        scores = fold_scores(panel, feats, arm)
        if len(scores) != len(base):
            print(f"  {label:34} skipped: {len(scores)} folds vs {len(base)}")
            continue
        c = compare(base, scores)
        print(f"  {label:34} {c.format()}", flush=True)
        rows.append(
            {
                "arm": arm,
                "label": label,
                "baseline_mean": c.baseline_mean,
                "candidate_mean": c.candidate_mean,
                "delta": c.mean_delta,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "n_folds": c.n_folds,
                "ships": c.helps,
            }
        )

    out = Path(__file__).parent / "search_emergence.json"
    out.write_text(
        json.dumps(
            {
                "protocol": f"rolling-origin {FIRST_TEST_YEAR}..validation, 3 seeds, paired bootstrap",
                "production_mean": float(np.mean(base)),
                "arms": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nsurvives the paired test: {[r['arm'] for r in rows if r['ships']] or 'none'}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
