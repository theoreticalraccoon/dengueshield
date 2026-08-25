"""Does the continuation model's capacity need cutting? Paired, over folds.

`LGBM_PARAMS_SL` sets `num_leaves=63` with the comment that 26 districts need
smaller trees than Brazil's thousands of municipalities. The reasoning is right and
the number was never checked: nothing in the repo ever searched this model's
hyperparameters. `finalize_srilanka.py` fits `ROUNDS = 500` and stops.

An unpaired sweep already suggested every lower-capacity configuration beats
production, but at fold sd ~0.09 over 8 folds that is inside the noise. This runs
the same sweep through `dengue.validation.compare`, which bootstraps the per-fold
DIFFERENCE - the folds are shared, so their difficulty cancels.

Exploratory. Nothing here is promoted automatically; the winner is copied into
config.py by hand after reading the output.

    .venv/Scripts/python.exe experiments/accuracy_v2/search_continuation.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import average_precision_score

from dengue.config import LGBM_PARAMS_SL, SL_HORIZON, SL_INC
from dengue.metrics import pos_weight
from dengue.srilanka import COMMON_FEATURES, add_features, build_panel
from dengue.validation import compare, fit_predict_seeds, rolling_origin

FIRST_TEST_YEAR = 2016

# The production configuration, and the candidates. Each is a delta from
# LGBM_PARAMS_SL so the diff from production is what the reader sees.
PRODUCTION = ("production (leaves=63, 500 rounds)", {}, 500)
CANDIDATES = [
    ("leaves=31", {"num_leaves": 31}, 500),
    ("leaves=15", {"num_leaves": 15}, 500),
    ("leaves=7", {"num_leaves": 7}, 500),
    ("leaves=7 min_child=80", {"num_leaves": 7, "min_child_samples": 80}, 500),
    ("leaves=15 min_child=80", {"num_leaves": 15, "min_child_samples": 80}, 500),
    ("leaves=31 reg_lambda=20", {"num_leaves": 31, "reg_lambda": 20.0}, 500),
    ("250 rounds", {}, 250),
    ("leaves=15, 250 rounds", {"num_leaves": 15}, 250),
    ("1000 rounds", {}, 1000),
]


def fold_scores(sl, feats, overrides: dict, rounds: int) -> tuple[list[float], list[float]]:
    """PR-AUC per fold, plus the persistence baseline on the identical folds."""
    scores, baseline = [], []
    for _year, train, test in rolling_origin(sl, first_test_year=FIRST_TEST_YEAR):
        if test.y.sum() == 0:
            continue
        params = {
            **LGBM_PARAMS_SL,
            **overrides,
            "scale_pos_weight": pos_weight(train.y.values),
        }
        p = fit_predict_seeds(params, train, test, feats, num_boost_round=rounds)
        scores.append(average_precision_score(test.y.values, p))
        baseline.append(average_precision_score(test.y.values, test.baseline_persistence.values))
    return scores, baseline


def main() -> int:
    sl = add_features(build_panel(), horizon=SL_HORIZON, outbreak_inc=SL_INC)
    feats = [c for c in COMMON_FEATURES if c in sl.columns]

    label, overrides, rounds = PRODUCTION
    base_scores, persistence = fold_scores(sl, feats, overrides, rounds)
    print(f"folds={len(base_scores)}  features={len(feats)}")
    print(f"  persistence baseline        {np.mean(persistence):.4f}")
    print(f"  {label:28} {np.mean(base_scores):.4f}\n")

    rows = []
    for name, over, rnd in CANDIDATES:
        scores, _ = fold_scores(sl, feats, over, rnd)
        c = compare(base_scores, scores)
        print(f"  {name:28} {c.format()}", flush=True)
        rows.append(
            {
                "candidate": name,
                "overrides": over,
                "rounds": rnd,
                "baseline_mean": c.baseline_mean,
                "candidate_mean": c.candidate_mean,
                "delta": c.mean_delta,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "n_folds": c.n_folds,
                "ships": c.helps,
            }
        )

    # Exploratory output stays in experiments/. reports/ is the production evidence
    # a release freezes, and `artifacts.freezable` exists because search output used
    # to leak into it.
    out = Path(__file__).parent / "search_continuation.json"
    out.write_text(
        json.dumps(
            {
                "protocol": f"rolling-origin {FIRST_TEST_YEAR}..validation, 3 seeds, paired bootstrap",
                "persistence_baseline": float(np.mean(persistence)),
                "production": {"label": label, "mean": float(np.mean(base_scores))},
                "candidates": rows,
            },
            indent=2,
            default=float,
        )
    )
    winners = [r for r in rows if r["ships"]]
    print(f"\nsurvives the paired test: {[r['candidate'] for r in winners] or 'none'}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
