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

import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from dengue import predictor
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

ROUNDS = 500


def train(panel: pd.DataFrame):
    """Fit on train, tune the threshold on validation, score the locked test years."""
    sl = add_features(panel, horizon=SL_HORIZON, outbreak_inc=SL_INC)
    feats = [c for c in COMMON_FEATURES if c in sl.columns]

    tr = sl[sl.anio <= TRAIN_END]
    va = sl[(sl.anio > TRAIN_END) & (sl.anio <= VAL_END)]
    te = sl[sl.anio > VAL_END]
    print(f"train={len(tr)} val={len(va)} test={len(te)}  features={len(feats)}")

    params = {**LGBM_PARAMS_SL, "scale_pos_weight": pos_weight(tr.y.values)}
    dev_model = lgb.train(params, lgb.Dataset(tr[feats], label=tr.y.values), num_boost_round=ROUNDS)

    p_va = dev_model.predict(va[feats])
    thr = threshold_for_f1(va.y.values, p_va)  # validation only; test stays locked
    p_te = dev_model.predict(te[feats])

    metrics = {
        "pr_auc": float(average_precision_score(te.y.values, p_te)),
        "roc_auc": float(roc_auc_score(te.y.values, p_te)),
        "threshold": float(thr),
        "horizon_weeks": SL_HORIZON,
        "outbreak_inc": SL_INC,
        "n_test": len(te),
    }
    print(
        f"held-out test: PR-AUC={metrics['pr_auc']:.4f} "
        f"ROC-AUC={metrics['roc_auc']:.4f}  threshold={thr:.4f}"
    )

    # Production model: refit on train + val, same hyperparameters and threshold.
    dev = pd.concat([tr, va])
    model = lgb.train(
        {**params, "scale_pos_weight": pos_weight(dev.y.values)},
        lgb.Dataset(dev[feats], label=dev.y.values),
        num_boost_round=ROUNDS,
    )
    return model, thr, feats, metrics


def current_forecast(panel: pd.DataFrame, model, thr: float, feats: list[str]) -> pd.DataFrame:
    """Score the forecast frontier - the weeks whose future label is not yet known.

    horizon=0 keeps those rows, which the labelled training frame drops.
    """
    latest = panel.week_start.max()
    full = panel.sort_values(["district", "week_start"]).copy()
    frontier = add_features(full.assign(_keep=1), horizon=0, outbreak_inc=SL_INC)
    recent = frontier[frontier.week_start >= latest - pd.Timedelta(weeks=1)].copy()
    recent["risk"] = model.predict(recent[feats])

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

    model, thr, feats, metrics = train(panel)

    MODELS.mkdir(parents=True, exist_ok=True)
    predictor.save_bundle(
        MODELS / "srilanka_outbreak.joblib",
        model=model,
        features=feats,
        threshold=thr,
        metrics=metrics,
        horizon=SL_HORIZON,
        outbreak_inc=SL_INC,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    out = current_forecast(panel, model, thr, feats)
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
