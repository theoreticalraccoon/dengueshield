"""Train and persist the production Sri Lanka outbreak model + current forecasts.

The transfer experiment showed the Sri Lanka-only model is the best performer on
Sri Lankan data (PR-AUC 0.708 vs 0.449 zero-shot from Brazil), so that is what
gets deployed. Trained on everything up to the validation cut, then used to score
the most recent weeks for the dashboard.

This answers CONTINUATION - will an existing outbreak persist? Emergence is a
different question asked of a different population; see finalize_emergence.py,
which consumes the bundle this writes.

    python finalize_srilanka.py
"""

from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dengue import predictor
from dengue.calibration import apply_calibrator, fit_isotonic_oof
from dengue.config import (
    LGBM_PARAMS_SL,
    MODELS,
    PROC,
    REPORTS,
    SL_HORIZON,
    SL_INC,
    TRAIN_END,
    VAL_END,
)
from dengue.metrics import pos_weight, threshold_for_f1
from dengue.srilanka import COMMON_FEATURES, DISTRICTS, add_features, build_panel
from dengue.validation import ece, train_seed_ensemble

# 500 previously, and never checked. Paired over 8 rolling-origin folds, 250 rounds
# beats 500 (+0.0130 [+0.0086, +0.0170]) and 1000 loses to it - the model was being
# boosted past its own optimum. See src/dengue/config.py for the capacity sweep.
ROUNDS = 250

# First fold of the out-of-fold calibration sweep. Earlier years exist but the panel
# thins out backwards, and a fold that trains on three districts calibrates nothing.
CALIBRATION_FIRST_YEAR = 2016


def train(panel: pd.DataFrame):
    """Fit on train, calibrate and threshold on development, score the locked test."""
    sl = add_features(panel, horizon=SL_HORIZON, outbreak_inc=SL_INC)
    feats = [c for c in COMMON_FEATURES if c in sl.columns]

    tr = sl[sl.anio <= TRAIN_END]
    va = sl[(sl.anio > TRAIN_END) & (sl.anio <= VAL_END)]
    te = sl[sl.anio > VAL_END]
    print(f"train={len(tr)} val={len(va)} test={len(te)}  features={len(feats)}")

    params = {**LGBM_PARAMS_SL, "scale_pos_weight": pos_weight(tr.y.values)}
    # Seed-averaged, because the spread between seeds on these test years is 0.02
    # PR-AUC - larger than the capacity change above bought. Development measured an
    # averaged model, so production has to deploy one.
    dev_model = train_seed_ensemble(params, tr, feats, num_boost_round=ROUNDS)

    # `scale_pos_weight` gets the ranking right and the scale wrong, so the raw
    # score is not a probability. Fit the correction on out-of-fold predictions
    # across the development years - never on test.
    dev = pd.concat([tr, va])
    calibrator, cal_report = fit_isotonic_oof(
        {k: v for k, v in params.items() if k != "scale_pos_weight"},
        dev,
        feats,
        num_boost_round=ROUNDS,
        first_test_year=CALIBRATION_FIRST_YEAR,
        pos_weight_fn=pos_weight,
    )
    print(f"  {cal_report.format()}")

    # The threshold now lives on the calibrated scale, because that is the scale
    # every consumer sees.
    p_va = apply_calibrator(calibrator, dev_model.predict(va[feats]))
    thr = threshold_for_f1(va.y.values, p_va)  # validation only; test stays locked

    raw_te = dev_model.predict(te[feats])
    p_te = apply_calibrator(calibrator, raw_te)

    # Discrimination is reported on the RAW score and calibration on the calibrated
    # one, because that is what each measures. Isotonic never reverses a pair, so it
    # cannot improve ranking; what it does do is merge scores it cannot tell apart
    # into ties, and PR-AUC charges for ties. Reporting the calibrated PR-AUC as the
    # headline would book that coarsening as a loss of model quality, which it is
    # not - the decisions are identical, since thresholding a monotone transform is
    # the same partition either way.
    metrics = {
        "pr_auc": float(average_precision_score(te.y.values, raw_te)),
        "roc_auc": float(roc_auc_score(te.y.values, raw_te)),
        # Must not EXCEED pr_auc. If it does, the calibrator saw test labels.
        "pr_auc_calibrated": float(average_precision_score(te.y.values, p_te)),
        "ece": float(ece(te.y.values, p_te)),
        "ece_uncalibrated": float(ece(te.y.values, raw_te)),
        "brier": float(brier_score_loss(te.y.values, np.clip(p_te, 0, 1))),
        "brier_uncalibrated": float(brier_score_loss(te.y.values, np.clip(raw_te, 0, 1))),
        "calibration_dev": cal_report.as_dict(),
        # From the same test rows as the score above it, so the two are always the
        # same experiment. Quoting a model and a baseline from different runs is the
        # bug docs/adr/0002 exists for.
        "baseline_persistence_pr_auc": float(
            average_precision_score(te.y.values, te.baseline_persistence.values)
        ),
        "threshold": float(thr),
        "horizon_weeks": SL_HORIZON,
        "outbreak_inc": SL_INC,
        "n_test": len(te),
    }
    print(
        f"held-out test: PR-AUC={metrics['pr_auc']:.4f} (calibrated {metrics['pr_auc_calibrated']:.4f}) "
        f"ROC-AUC={metrics['roc_auc']:.4f}  threshold={thr:.4f}"
    )
    print(
        f"  calibration on test: ECE {metrics['ece_uncalibrated']:.4f} -> {metrics['ece']:.4f}  "
        f"Brier {metrics['brier_uncalibrated']:.4f} -> {metrics['brier']:.4f}"
    )

    # Production model: refit on train + val, same hyperparameters, same calibrator
    # and threshold. The calibrator maps out-of-fold scores from these same years,
    # which is why it was not fitted on the validation split alone.
    model = train_seed_ensemble(
        {**params, "scale_pos_weight": pos_weight(dev.y.values)},
        dev,
        feats,
        num_boost_round=ROUNDS,
    )
    return model, thr, feats, metrics, calibrator


def current_forecast(
    panel: pd.DataFrame, model, thr: float, feats: list[str], calibrator=None
) -> pd.DataFrame:
    """Score the forecast frontier - the weeks whose future label is not yet known.

    horizon=0 keeps those rows, which the labelled training frame drops.
    """
    latest = panel.week_start.max()
    full = panel.sort_values(["district", "week_start"]).copy()
    frontier = add_features(full.assign(_keep=1), horizon=0, outbreak_inc=SL_INC)
    recent = frontier[frontier.week_start >= latest - pd.Timedelta(weeks=1)].copy()
    # Calibrated, so the bands below and the number the dashboard prints are on the
    # same scale as the threshold they are drawn around.
    recent["risk"] = apply_calibrator(calibrator, model.predict(recent[feats]))

    # Bands straddle the operating threshold and stay inside [0, 1] whatever thr is.
    edges = sorted({0.0, round(thr * 0.5, 6), round(thr, 6), round(thr + (1.0 - thr) / 2, 6), 1.0})
    labels = ["Low", "Moderate", "High", "Very High"][: len(edges) - 1]
    recent["risk_band"] = pd.cut(recent.risk, edges, labels=labels, include_lowest=True)

    cols = [
        "district",
        "week_start",
        "casos",
        "p_inc100k",
        "casos_roll4_mean",
        "precip_roll4_sum",
        "tempmed",
        "umidmed",
        "population_total",
        "risk",
        "risk_band",
    ]
    out = (
        recent.sort_values("week_start")
        .groupby("district", as_index=False)
        .last()[cols]
        .sort_values("risk", ascending=False)
    )
    out["lat"] = out.district.map(lambda d: DISTRICTS[d][0])
    out["lon"] = out.district.map(lambda d: DISTRICTS[d][1])
    return out


def main(panel: pd.DataFrame | None = None) -> int:
    """Panel is passed in by refresh_data.py so it is built once, not twice."""
    if panel is None:
        panel = build_panel()

    model, thr, feats, metrics, calibrator = train(panel)

    MODELS.mkdir(parents=True, exist_ok=True)
    predictor.save_bundle(
        MODELS / "srilanka_outbreak.joblib",
        model=model,
        features=feats,
        threshold=thr,
        metrics=metrics,
        horizon=SL_HORIZON,
        outbreak_inc=SL_INC,
        calibrator=calibrator,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    # The continuation metrics used to live only inside the joblib, so the About
    # screen had no artifact to quote and stated its calibration status in prose.
    (REPORTS / "srilanka_continuation.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )

    out = current_forecast(panel, model, thr, feats, calibrator)
    out.to_csv(REPORTS / "srilanka_current_risk.csv", index=False)

    panel[
        ["district", "week_start", "casos", "p_inc100k", "precip_total_semana", "tempmed", "umidmed"]
    ].to_parquet(PROC / "srilanka_history.parquet", index=False)

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(REPORTS / "srilanka_feature_importance.csv", header=["gain"])

    print(f"\nforecast week: {out.week_start.max().date()}  (horizon {SL_HORIZON} weeks)")
    print(
        out[["district", "casos", "p_inc100k", "risk", "risk_band"]].head(12).to_string(index=False)
    )
    print("\nsaved models/srilanka_outbreak.joblib, reports/srilanka_current_risk.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
