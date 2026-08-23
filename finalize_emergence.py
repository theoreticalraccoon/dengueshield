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
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue import predictor
from dengue.config import (
    EMERGENCE_HORIZON,
    LGBM_PARAMS_SL,
    MODELS,
    QUIET_WEEKS,
    REPORTS,
    SEED,
    SL_INC,
    TRAIN_END,
    VAL_END,
)
from dengue.emergence import add_base_features, label_emergence
from dengue.metrics import pos_weight, threshold_for_f1
from dengue.srilanka import COMMON_FEATURES, DISTRICTS, build_panel

# Emergence needs shallower trees than continuation: far fewer eligible rows.
PARAMS = {**LGBM_PARAMS_SL, "num_leaves": 31, "reg_lambda": 10.0, "seed": SEED}
ROUNDS = 400


def train(panel: pd.DataFrame) -> tuple[lgb.Booster, dict, list[str]]:
    """Fit on train, tune the threshold on validation, score the locked test years."""
    feats = [c for c in COMMON_FEATURES if c in panel.columns]
    em = label_emergence(panel)

    tr = em[em.anio <= TRAIN_END]
    va = em[(em.anio > TRAIN_END) & (em.anio <= VAL_END)]
    te = em[em.anio > VAL_END]
    print(f"eligible district-weeks: train={len(tr)} val={len(va)} test={len(te)}")
    print(
        f"emergence rate: train={tr.y.mean():.4f} test={te.y.mean():.4f} "
        f"({int(te.y.sum())} positive test events)"
    )

    dev_model = lgb.train(
        {**PARAMS, "scale_pos_weight": pos_weight(tr.y.values)},
        lgb.Dataset(tr[feats], label=tr.y.values),
        num_boost_round=ROUNDS,
    )
    p_val = dev_model.predict(va[feats])
    p_test = dev_model.predict(te[feats])

    thr = threshold_for_f1(va.y.values, p_val)  # validation only; test stays locked
    yhat = (p_test >= thr).astype(int)
    baseline = float(average_precision_score(te.y.values, te.p_inc100k.values))

    metrics = {
        "pr_auc": float(average_precision_score(te.y.values, p_test)),
        "roc_auc": float(roc_auc_score(te.y.values, p_test)),
        "recall": float(recall_score(te.y.values, yhat, zero_division=0)),
        "precision": float(precision_score(te.y.values, yhat, zero_division=0)),
        "prevalence": float(te.y.mean()),
        "n_positive": int(te.y.sum()),
        "threshold": float(thr),
        "horizon_weeks": EMERGENCE_HORIZON,
        "outbreak_inc": SL_INC,
        "quiet_weeks": QUIET_WEEKS,
        "baseline_persistence_pr_auc": baseline,
    }
    print(
        f"held-out test: PR-AUC={metrics['pr_auc']:.4f} "
        f"(persistence baseline {baseline:.4f}) recall={metrics['recall']:.4f}"
    )

    # Production model: refit on train+val at the validation-chosen threshold.
    dev = pd.concat([tr, va])
    model = lgb.train(
        {**PARAMS, "scale_pos_weight": pos_weight(dev.y.values)},
        lgb.Dataset(dev[feats], label=dev.y.values),
        num_boost_round=ROUNDS,
    )
    return model, metrics, feats


def dual_forecast(panel: pd.DataFrame, model: lgb.Booster, feats: list[str]) -> pd.DataFrame:
    """One row per district: continuation risk, emergence risk, and eligibility."""
    continuation = predictor.load(predictor.SL_CONTINUATION)
    if continuation is None:
        raise SystemExit(
            "models/srilanka_outbreak.joblib is missing - run finalize_srilanka.py first."
        )

    scored = label_emergence(panel, keep_ineligible=True)
    latest = panel.week_start.max()
    recent = scored[scored.week_start >= latest - pd.Timedelta(weeks=1)].copy()

    recent["emergence_risk"] = model.predict(recent[feats])
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
    """Panel is passed in by refresh_data.py so it is built once, not twice."""
    if panel is None:
        panel = add_base_features(build_panel())

    model, metrics, feats = train(panel)

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
    )

    out = dual_forecast(panel, model, feats)
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
