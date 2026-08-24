"""Train the Sri Lanka EMERGENCE model and produce the dual district forecast.

Gives the app two complementary numbers per district:

    continuation risk  - will an existing outbreak persist?   (finalize_srilanka.py)
    emergence risk     - is a NEW outbreak about to begin?     (this script)

Emergence is only asked of districts not currently in outbreak, which is exactly
where continuation is blind. Districts in neither population are scored NaN,
meaning "not asked" - see docs/adr/0001-blank-emergence-risk.md.

Depends on models/srilanka_outbreak.joblib, so finalize_srilanka.py must run first.
That ordering is now checked rather than assumed.

    python finalize_emergence.py
"""

from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue import predictor
from dengue.calibration import apply_calibrator, fit_isotonic_oof
from dengue.config import (
    EMERGENCE_HORIZON,
    EMERGENCE_SENSITIVITY_TARGET,
    LGBM_PARAMS_SL,
    MODELS,
    QUIET_WEEKS,
    REPORTS,
    SEED,
    SL_INC,
    TRAIN_END,
    VAL_END,
)
from dengue.emergence import (
    HISTORY_FEATURES,
    add_base_features,
    add_history_features,
    label_emergence,
)
from dengue.metrics import pos_weight, threshold_for_f1, threshold_for_sensitivity
from dengue.srilanka import COMMON_FEATURES, DISTRICTS, build_panel
from dengue.validation import ece, train_seed_ensemble

# Emergence needs far shallower trees than continuation: ~18k eligible training
# rows at ~4% positives. Selected by rolling-origin CV over seven folds ending
# 2023 (cv PR-AUC 0.363 -> 0.395); the locked test years were not consulted.
PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600

# First fold of the out-of-fold calibration sweep, matching finalize_srilanka.py.
CALIBRATION_FIRST_YEAR = 2016


def train(panel: pd.DataFrame) -> tuple[lgb.Booster, dict, list[str], object]:
    """Fit on train, tune the threshold on validation, score the locked test years."""
    feats = [c for c in COMMON_FEATURES if c in panel.columns]
    feats += [c for c in HISTORY_FEATURES if c in panel.columns]
    em = label_emergence(panel)

    tr = em[em.anio <= TRAIN_END]
    va = em[(em.anio > TRAIN_END) & (em.anio <= VAL_END)]
    te = em[em.anio > VAL_END]
    print(f"eligible district-weeks: train={len(tr)} val={len(va)} test={len(te)}")
    print(
        f"emergence rate: train={tr.y.mean():.4f} test={te.y.mean():.4f} "
        f"({int(te.y.sum())} positive test events)"
    )

    # Seed-averaged: see dengue.validation.SeedEnsemble. A single booster here is
    # a draw from a distribution 0.02 PR-AUC wide.
    dev_model = train_seed_ensemble(
        {**PARAMS, "scale_pos_weight": pos_weight(tr.y.values)},
        tr,
        feats,
        num_boost_round=ROUNDS,
    )

    # `scale_pos_weight` is ~23 here, so the raw score is far above the event rate.
    # The sensitivity target below is read off this scale, and the app prints it as
    # a percentage, so it has to be a probability first. Fitted out-of-fold across
    # the development years - never on test.
    dev = pd.concat([tr, va])
    calibrator, cal_report = fit_isotonic_oof(
        {k: v for k, v in PARAMS.items() if k != "scale_pos_weight"},
        dev,
        feats,
        num_boost_round=ROUNDS,
        first_test_year=CALIBRATION_FIRST_YEAR,
        pos_weight_fn=pos_weight,
    )
    print(f"  {cal_report.format()}")

    raw_test = dev_model.predict(te[feats])
    p_val = apply_calibrator(calibrator, dev_model.predict(va[feats]))
    p_test = apply_calibrator(calibrator, raw_test)

    # Both thresholds come from validation. F1 balances the two errors; the
    # sensitivity target says how many emerging outbreaks we insist on catching.
    thr_f1 = threshold_for_f1(va.y.values, p_val)
    thr = threshold_for_sensitivity(va.y.values, p_val, EMERGENCE_SENSITIVITY_TARGET)
    yhat = (p_test >= thr).astype(int)
    baseline = float(average_precision_score(te.y.values, te.p_inc100k.values))

    # Every operating point, so the recall/false-alarm trade is a decision the
    # reader can see rather than a single number they have to trust.
    districts_per_week = max(te.week_start.nunique(), 1)
    operating_points = []
    for target in (0.50, 0.60, 0.70, 0.80, 0.90):
        t = threshold_for_sensitivity(va.y.values, p_val, target)
        yh = (p_test >= t).astype(int)
        operating_points.append(
            {
                "sensitivity_target": target,
                "threshold": float(t),
                "recall": float(recall_score(te.y.values, yh, zero_division=0)),
                "precision": float(precision_score(te.y.values, yh, zero_division=0)),
                "accuracy": float((yh == te.y.values).mean()),
                "flagged_per_week": round(float(yh.sum()) / districts_per_week, 2),
            }
        )

    # Discrimination on the raw score, calibration on the calibrated one - see the
    # same split in finalize_srilanka.py. Recall and precision below stay on the
    # calibrated scale, because that is the scale the operating point lives on.
    metrics = {
        "pr_auc": float(average_precision_score(te.y.values, raw_test)),
        "roc_auc": float(roc_auc_score(te.y.values, raw_test)),
        # Must not EXCEED pr_auc. If it does, the calibrator saw test labels.
        "pr_auc_calibrated": float(average_precision_score(te.y.values, p_test)),
        "ece": float(ece(te.y.values, p_test)),
        "ece_uncalibrated": float(ece(te.y.values, raw_test)),
        "brier": float(brier_score_loss(te.y.values, np.clip(p_test, 0, 1))),
        "brier_uncalibrated": float(brier_score_loss(te.y.values, np.clip(raw_test, 0, 1))),
        "calibration_dev": cal_report.as_dict(),
        "recall": float(recall_score(te.y.values, yhat, zero_division=0)),
        "precision": float(precision_score(te.y.values, yhat, zero_division=0)),
        "prevalence": float(te.y.mean()),
        "n_positive": int(te.y.sum()),
        "threshold": float(thr),
        "threshold_f1": float(thr_f1),
        "sensitivity_target": EMERGENCE_SENSITIVITY_TARGET,
        "operating_points": operating_points,
        "horizon_weeks": EMERGENCE_HORIZON,
        "outbreak_inc": SL_INC,
        "quiet_weeks": QUIET_WEEKS,
        "baseline_persistence_pr_auc": baseline,
        "trivial_never_flag_accuracy": float(1 - te.y.mean()),
    }
    print(
        f"held-out test: PR-AUC={metrics['pr_auc']:.4f} "
        f"(persistence baseline {baseline:.4f}) recall={metrics['recall']:.4f}"
    )

    # Production model: refit on train+val at the validation-chosen threshold, with
    # the same calibrator - which is why it was fitted out-of-fold across these
    # years rather than on the validation split alone.
    model = train_seed_ensemble(
        {**PARAMS, "scale_pos_weight": pos_weight(dev.y.values)},
        dev,
        feats,
        num_boost_round=ROUNDS,
    )
    return model, metrics, feats, calibrator


def dual_forecast(
    panel: pd.DataFrame, model: lgb.Booster, feats: list[str], calibrator=None
) -> pd.DataFrame:
    """One row per district: continuation risk, emergence risk, and eligibility."""
    continuation = predictor.load(predictor.SL_CONTINUATION)
    if continuation is None:
        raise SystemExit(
            "models/srilanka_outbreak.joblib is missing - run finalize_srilanka.py first."
        )

    scored = label_emergence(panel, keep_ineligible=True)
    latest = panel.week_start.max()
    recent = scored[scored.week_start >= latest - pd.Timedelta(weeks=1)].copy()

    recent["emergence_risk"] = apply_calibrator(calibrator, model.predict(recent[feats]))
    recent.loc[~recent.eligible, "emergence_risk"] = np.nan  # question not asked
    recent["currently_in_outbreak"] = recent.p_inc100k >= SL_INC
    recent["continuation_risk"] = continuation.predict_frame(recent)

    cols = [
        "district",
        "week_start",
        "casos",
        "p_inc100k",
        "currently_in_outbreak",
        "continuation_risk",
        "emergence_risk",
        "casos_roll4_mean",
        "precip_roll4_sum",
        "tempmed",
        "umidmed",
        "population_total",
    ]
    out = recent.sort_values("week_start").groupby("district", as_index=False).last()[cols]
    out["lat"] = out.district.map(lambda d: DISTRICTS[d][0])
    out["lon"] = out.district.map(lambda d: DISTRICTS[d][1])
    return out.sort_values("continuation_risk", ascending=False)


def main(panel: pd.DataFrame | None = None) -> int:
    """Takes the RAW district-week panel, same as finalize_srilanka.main.

    refresh_data.py builds it once and hands the same frame to both, so the
    contract has to be identical; this script adds the features it needs.
    """
    raw = build_panel() if panel is None else panel
    panel = add_history_features(add_base_features(raw))

    model, metrics, feats, calibrator = train(panel)

    MODELS.mkdir(parents=True, exist_ok=True)
    predictor.save_bundle(
        MODELS / "srilanka_emergence.joblib",
        model=model,
        features=feats,
        threshold=metrics["threshold"],
        metrics=metrics,
        horizon=EMERGENCE_HORIZON,
        outbreak_inc=SL_INC,
        quiet_weeks=QUIET_WEEKS,
        calibrator=calibrator,
    )

    out = dual_forecast(panel, model, feats, calibrator)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "srilanka_dual_risk.csv", index=False)
    (REPORTS / "srilanka_emergence.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nforecast week {out.week_start.max().date()}  (horizon 1-{EMERGENCE_HORIZON} weeks)")
    print(
        out[["district", "casos", "p_inc100k", "currently_in_outbreak", "emergence_risk"]]
        .round(3)
        .head(14)
        .to_string(index=False)
    )
    print("\nsaved models/srilanka_emergence.joblib, reports/srilanka_dual_risk.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
