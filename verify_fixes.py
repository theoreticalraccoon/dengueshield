"""Quantify what the date-exact lag rebuild changed, against the frozen v2 numbers."""

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue.config import BR_HORIZON, BR_INC, SEED
from dengue.experiment import setup
from dengue.metrics import threshold_for_f1
from dengue.model2_outbreak import load_panel

HORIZON, INC = BR_HORIZON, BR_INC
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
    "seed": SEED,
}


def run(rebuild):
    p = load_panel(rebuild_lags=rebuild)
    r = setup(panel=p, horizon=HORIZON, outbreak_inc=INC)
    f = r.feats
    tr, va, te = r.tr, r.va, r.te
    spw = r.spw
    m = lgb.train(
        {**PARAMS, "scale_pos_weight": spw},
        lgb.Dataset(tr[f], label=tr.y.values),
        num_boost_round=700,
    )
    pv, pt = m.predict(va[f]), m.predict(te[f])
    thr = threshold_for_f1(va.y.values, pv, grid=150)
    yh = (pt >= thr).astype(int)
    return {
        "rows": len(r.sup),
        "n_test": len(te),
        "test_prevalence": float(te.y.mean()),
        "pr_auc": average_precision_score(te.y.values, pt),
        "roc_auc": roc_auc_score(te.y.values, pt),
        "accuracy": accuracy_score(te.y.values, yh),
        "recall": recall_score(te.y.values, yh, zero_division=0),
        "precision": precision_score(te.y.values, yh, zero_division=0),
        "brier": brier_score_loss(te.y.values, pt),
    }


print("Rebuilding the 14-day outbreak model both ways on the locked 2024-25 test set.\n")
before = run(False)
print(
    f"  BEFORE (published positional lags): PR-AUC={before['pr_auc']:.4f} "
    f"acc={before['accuracy']:.4f} recall={before['recall']:.4f}",
    flush=True,
)
after = run(True)
print(
    f"  AFTER  (date-exact rebuilt lags)  : PR-AUC={after['pr_auc']:.4f} "
    f"acc={after['accuracy']:.4f} recall={after['recall']:.4f}",
    flush=True,
)

delta = {k: after[k] - before[k] for k in before if isinstance(before[k], float)}
print("\n  delta (after - before):")
for k, v in delta.items():
    print(f"    {k:16} {v:+.4f}")
with open("reports/lag_fix_impact.json", "w") as _f:
    json.dump(
        {"before_positional_lags": before, "after_date_exact_lags": after, "delta": delta},
        _f,
        indent=2,
        default=float,
    )
print("\nsaved reports/lag_fix_impact.json")
