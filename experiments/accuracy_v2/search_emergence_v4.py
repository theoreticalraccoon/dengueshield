"""Can emergence be asked a better-shaped question? Plus the two live feature leads.

`search_emergence_v3.py` tested four new feature blocks and shipped none of them.
Climate was the only one positive on both axes (+0.021 per fold, +0.010 per district)
and both intervals covered zero; importation was an outright regression. So this
stops adding columns and changes the FORM of the problem, which is where the largest
unexploited fact about this task lives:

    the binary label throws away almost everything the outcome knows.

23,053 eligible district-weeks collapse to 1,090 positives. A week where incidence
reached 9.8 and a week where it reached 0.2 are both `y=0`, and a week that hit 40 is
worth exactly as much as one that grazed 10.0. The regression arms below keep the
magnitude and rank districts by predicted future incidence instead, which gives every
row a target rather than one row in twenty-one.

This is NOT the discrete-time hazard of ADR 0003 arm C. That reformulation was still
binary - it asked "does it cross in exactly week k?" and recombined the hazards. The
question here is continuous.

  A   production            binary classifier, the shipped formulation
  R1  regression            LightGBM on log1p(max incidence over t+1..t+H), ranked
  R2  tweedie               same target, Tweedie objective, for the zero-inflation
  Q95 quantile              the 95th percentile of future incidence, ranked
  BL  blend                 rank-average of A and R1

  plus the two feature leads that are still open, on the binary arm:
  CA  climate anomalies     the three departure-from-normal columns only
  CC  climate core          anomalies + diurnal range + the long rainfall memory
  EN  enso                  lagged Nino 3.4, publication-lag guarded
  CE  climate + enso

Every arm is scored the same way: PR-AUC of its ranking against the same binary label
on the eligible rows of the held-out year. Ranking metrics do not care what scale a
score is on, which is what makes a regression and a classifier comparable at all. The
locked test years are never touched.

    .venv/Scripts/python.exe experiments/accuracy_v2/search_emergence_v4.py
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from dengue.config import EMERGENCE_HORIZON, LGBM_PARAMS_SL, SEED
from dengue.emergence import (
    CLIMATE_FEATURES,
    HISTORY_FEATURES,
    add_base_features,
    add_extra_features,
    add_history_features,
    label_emergence,
)
from dengue.enso import ENSO_FEATURES, add_enso_features
from dengue.metrics import pos_weight
from dengue.srilanka import COMMON_FEATURES, build_panel
from dengue.validation import (
    compare,
    fast_average_precision,
    fit_predict_seeds,
    pooled_compare,
    rolling_origin,
)

PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600
FIRST_TEST_YEAR = 2010
SEEDS = (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)

ANOMALIES = ["precip_anom_z", "tempmed_anom_z", "umid_anom_z"]
CLIMATE_CORE = [
    *ANOMALIES,
    "dtr",
    "dtr_roll4",
    "precip_lag_8",
    "precip_lag_12",
    "precip_roll12_sum",
]


def add_regression_target(df: pd.DataFrame, horizon: int = EMERGENCE_HORIZON) -> pd.DataFrame:
    """Peak incidence over the same window the binary label maxes over.

    Deliberately the identical window, so the two formulations differ ONLY in
    whether the outcome is thresholded. Anything else would confound the comparison
    with a change of question.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    future = pd.concat([g["p_inc100k"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y_max_inc"] = future.max(axis=1)
    df["y_log_inc"] = np.log1p(df.y_max_inc)
    return df


def fold_run(labelled: pd.DataFrame, feats: list[str], arm: str) -> tuple[list[float], pd.DataFrame]:
    """Per-fold PR-AUC and stacked out-of-fold scores, whatever the arm predicts."""
    scores: list[float] = []
    frames: list[pd.DataFrame] = []

    for _year, train, test in rolling_origin(labelled, first_test_year=FIRST_TEST_YEAR):
        if test.y.sum() == 0:
            continue

        if arm in ("binary", "blend"):
            params = {**PARAMS, "scale_pos_weight": pos_weight(train.y.values)}
            p = fit_predict_seeds(params, train, test, feats, num_boost_round=ROUNDS, seeds=SEEDS)
        if arm in ("regression", "blend"):
            # No class weighting: there are no classes. Every row carries a target.
            r = fit_predict_seeds(
                {**PARAMS, "objective": "regression"},
                train,
                test,
                feats,
                num_boost_round=ROUNDS,
                label_col="y_log_inc",
                seeds=SEEDS,
            )
        if arm == "tweedie":
            p = fit_predict_seeds(
                {**PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.3},
                train,
                test,
                feats,
                num_boost_round=ROUNDS,
                label_col="y_max_inc",
                seeds=SEEDS,
            )
        elif arm == "quantile":
            p = fit_predict_seeds(
                {**PARAMS, "objective": "quantile", "alpha": 0.95},
                train,
                test,
                feats,
                num_boost_round=ROUNDS,
                label_col="y_log_inc",
                seeds=SEEDS,
            )
        elif arm == "regression":
            p = r
        elif arm == "blend":
            # Rank-average, so the two scales never have to be reconciled.
            p = (rankdata(p) + rankdata(r)) / (2.0 * len(p))

        scores.append(fast_average_precision(test.y.to_numpy(dtype=float), np.asarray(p)))
        frames.append(pd.DataFrame({"district": test.district.values, "y": test.y.values, "p": p}))

    return scores, pd.concat(frames, ignore_index=True)


def main() -> int:
    panel = add_enso_features(
        add_regression_target(
            add_extra_features(add_history_features(add_base_features(build_panel())))
        )
    )
    base_feats = [c for c in COMMON_FEATURES if c in panel.columns]
    base_feats += [c for c in HISTORY_FEATURES if c in panel.columns]

    labelled = label_emergence(panel).dropna(subset=["y_log_inc"])

    arms = [
        ("R1 regression (log incidence)", "regression", base_feats),
        ("R2 tweedie", "tweedie", base_feats),
        ("Q95 quantile 0.95", "quantile", base_feats),
        ("BL blend of A and R1", "blend", base_feats),
        ("CA climate anomalies only", "binary", base_feats + ANOMALIES),
        ("CC climate core", "binary", base_feats + CLIMATE_CORE),
        ("EN enso", "binary", base_feats + ENSO_FEATURES),
        ("CE climate core + enso", "binary", base_feats + CLIMATE_CORE + ENSO_FEATURES),
        ("XX all climate", "binary", base_feats + CLIMATE_FEATURES),
    ]

    base_scores, base_oof = fold_run(labelled, base_feats, "binary")
    base_pooled = fast_average_precision(base_oof.y.to_numpy(dtype=float), base_oof.p.to_numpy())
    print(f"folds={len(base_scores)}  rows={len(labelled)}  positives={int(labelled.y.sum())}")
    print(
        f"  A production                   per-fold {np.mean(base_scores):.4f}  "
        f"pooled {base_pooled:.4f}  ROC {roc_auc_score(base_oof.y, base_oof.p):.4f}\n",
        flush=True,
    )

    rows = []
    for label, arm, feats in arms:
        scores, oof = fold_run(labelled, feats, arm)
        if len(scores) != len(base_scores):
            print(f"  {label:30} skipped: {len(scores)} folds vs {len(base_scores)}")
            continue

        by_fold = compare(base_scores, scores)
        by_district = pooled_compare(
            base_oof.y, base_oof.district, base_oof.p, oof.p, n_boot=1_000
        )
        ships = by_fold.helps and by_district.helps
        roc = float(roc_auc_score(oof.y, oof.p))

        print(f"  {label:30} folds     {by_fold.format()}")
        print(f"  {'':30} districts {by_district.format()}")
        print(f"  {'':30} ROC {roc:.4f}   -> {'SHIP' if ships else 'no'}\n", flush=True)

        rows.append(
            {
                "arm": label,
                "kind": arm,
                "n_features": len(feats),
                "per_fold_mean": by_fold.candidate_mean,
                "per_fold_delta": by_fold.mean_delta,
                "per_fold_ci": [by_fold.ci_low, by_fold.ci_high],
                "pooled_pr_auc": by_district.candidate_mean,
                "pooled_delta": by_district.mean_delta,
                "pooled_ci": [by_district.ci_low, by_district.ci_high],
                "pooled_roc_auc": roc,
                "ships": bool(ships),
            }
        )

    out = Path(__file__).parent / "search_emergence_v4.json"
    out.write_text(
        json.dumps(
            {
                "protocol": (
                    f"rolling-origin {FIRST_TEST_YEAR}..validation, {len(SEEDS)} seeds, "
                    "paired bootstrap over folds AND cluster bootstrap over districts; "
                    "ships only if both exclude zero in the same direction"
                ),
                "n_folds": len(base_scores),
                "baseline": {
                    "per_fold_mean": float(np.mean(base_scores)),
                    "pooled_pr_auc": base_pooled,
                    "pooled_roc_auc": float(roc_auc_score(base_oof.y, base_oof.p)),
                },
                "arms": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"survives BOTH tests: {[r['arm'] for r in rows if r['ships']] or 'none'}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
