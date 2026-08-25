"""The blend, measured on a scale that survives pooling.

`search_emergence_v4.py` produced the one result so far that cleared an interval: a
rank-average of the binary classifier and the incidence regression scored +0.0272
[+0.0033, +0.0561] over 14 folds. The same arm then read as a large REGRESSION on the
pooled district axis, and that second number is an artifact of how it was built
rather than a fact about the model.

The blend rank-normalised within each fold. Per-fold PR-AUC does not care - it is a
rank metric and within-fold order is preserved - but pooling those scores stacks
fourteen separate uniform distributions on top of each other. A quiet year and an
epidemic year both come out spanning 0 to 1, so the pooled ranking loses the
information that one year was worse than another, and the pooled PR-AUC falls for a
reason that has nothing to do with the model. Judging the arm on that number would
have thrown away a real effect; shipping it on the fold number alone would have
ignored a genuine second axis. Both required fixing the scale instead.

So the regression prediction is mapped to a probability with an isotonic fitted on
the TRAINING fold only, which is a monotone transform - it cannot change the per-fold
PR-AUC at all - but it puts both arms on an absolute scale that means the same thing
in every year, so the pooled comparison becomes valid.

  A     production            binary classifier
  R1p   regression, mapped    log-incidence regression -> in-fold isotonic
  BLp   blend                 mean of the two probabilities
  BLp+C blend + climate core  the best feature block, on both members

    .venv/Scripts/python.exe experiments/accuracy_v2/search_emergence_v5.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from dengue.config import EMERGENCE_HORIZON, LGBM_PARAMS_SL, SEED
from dengue.emergence import (
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

CLIMATE_CORE = [
    "precip_anom_z",
    "tempmed_anom_z",
    "umid_anom_z",
    "dtr",
    "dtr_roll4",
    "precip_lag_8",
    "precip_lag_12",
    "precip_roll12_sum",
]


def add_regression_target(df: pd.DataFrame, horizon: int = EMERGENCE_HORIZON) -> pd.DataFrame:
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    future = pd.concat([g["p_inc100k"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y_log_inc"] = np.log1p(future.max(axis=1))
    return df


def _classifier(train, test, feats):
    return fit_predict_seeds(
        {**PARAMS, "scale_pos_weight": pos_weight(train.y.values)},
        train,
        test,
        feats,
        num_boost_round=ROUNDS,
        seeds=SEEDS,
    )


def _regressor_as_probability(train, test, feats):
    """Regression score mapped onto a probability, using the training fold only.

    The isotonic is fitted on in-sample training predictions, which is not how one
    would calibrate a shipped model - but here it is a SCALE BRIDGE, not a
    calibration: it is monotone, so it leaves every per-fold rank metric untouched,
    and its only job is to make one fold's scores comparable with another's. The
    test fold is never involved.
    """
    params = {**PARAMS, "objective": "regression"}
    r_test = fit_predict_seeds(
        params, train, test, feats, num_boost_round=ROUNDS, label_col="y_log_inc", seeds=SEEDS
    )
    r_train = fit_predict_seeds(
        params, train, train, feats, num_boost_round=ROUNDS, label_col="y_log_inc", seeds=SEEDS
    )
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(r_train, train.y.to_numpy(dtype=float))
    return iso.predict(r_test)


def fold_run(labelled: pd.DataFrame, feats: list[str], arm: str) -> tuple[list[float], pd.DataFrame]:
    scores, frames = [], []
    for _year, train, test in rolling_origin(labelled, first_test_year=FIRST_TEST_YEAR):
        if test.y.sum() == 0:
            continue

        if arm == "binary":
            p = _classifier(train, test, feats)
        elif arm == "regression":
            p = _regressor_as_probability(train, test, feats)
        else:
            p = 0.5 * _classifier(train, test, feats) + 0.5 * _regressor_as_probability(
                train, test, feats
            )

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

    base_scores, base_oof = fold_run(labelled, base_feats, "binary")
    base_pooled = fast_average_precision(base_oof.y.to_numpy(dtype=float), base_oof.p.to_numpy())
    print(f"folds={len(base_scores)}  positives={int(labelled.y.sum())}")
    print(
        f"  A production                   per-fold {np.mean(base_scores):.4f}  "
        f"pooled {base_pooled:.4f}  ROC {roc_auc_score(base_oof.y, base_oof.p):.4f}\n",
        flush=True,
    )

    arms = [
        ("R1p regression as probability", "regression", base_feats),
        ("BLp blend", "blend", base_feats),
        ("BLp+C blend + climate core", "blend", base_feats + CLIMATE_CORE),
        ("BLp+CE blend + climate + enso", "blend", base_feats + CLIMATE_CORE + ENSO_FEATURES),
    ]

    rows = []
    for label, arm, feats in arms:
        scores, oof = fold_run(labelled, feats, arm)
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

    out = Path(__file__).parent / "search_emergence_v5.json"
    out.write_text(
        json.dumps(
            {
                "protocol": (
                    f"rolling-origin {FIRST_TEST_YEAR}..validation, {len(SEEDS)} seeds, "
                    "paired over folds AND clustered over districts; both must exclude zero. "
                    "Regression scores are mapped to probabilities by an isotonic fitted on "
                    "the training fold, so pooling across folds is meaningful."
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
