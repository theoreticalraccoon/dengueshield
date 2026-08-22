"""Train and persist the production Sri Lanka outbreak model + current forecasts.

The transfer experiment showed the Sri Lanka-only model is the best performer on
Sri Lankan data (PR-AUC 0.708 vs 0.449 zero-shot from Brazil), so that is what
gets deployed. Trained on everything up to the validation cut, then used to score
the most recent weeks for the dashboard.
"""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from dengue.srilanka import build_panel, add_features, COMMON_FEATURES, DISTRICTS

HORIZON = 2
OUTBREAK_INC = 9.9          # Sri Lanka-calibrated epidemic threshold (per 100k/week)
TRAIN_END, VAL_END = 2021, 2023

panel = build_panel()
sl = add_features(panel, horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
feats = [c for c in COMMON_FEATURES if c in sl.columns]

tr = sl[sl.anio <= TRAIN_END]
va = sl[(sl.anio > TRAIN_END) & (sl.anio <= VAL_END)]
te = sl[sl.anio > VAL_END]
print(f"train={len(tr)} val={len(va)} test={len(te)}  features={len(feats)}")

params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63,
          "min_child_samples": 40, "subsample": 0.85, "subsample_freq": 1,
          "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1, "seed": 42,
          "scale_pos_weight": (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)}

# threshold selected on validation only
m_dev = lgb.train(params, lgb.Dataset(tr[feats], label=tr.y.values), num_boost_round=500)
p_va = m_dev.predict(va[feats])
qs = np.unique(np.quantile(p_va, np.linspace(0.5, 0.999, 200)))
thr = float(max((f1_score(va.y.values, (p_va >= t).astype(int), zero_division=0), t) for t in qs)[1])
p_te = m_dev.predict(te[feats])
print(f"held-out test: PR-AUC={average_precision_score(te.y.values, p_te):.4f} "
      f"ROC-AUC={roc_auc_score(te.y.values, p_te):.4f}  threshold={thr:.4f}")

# production model: refit on train + val, same hyperparameters and threshold
dev = pd.concat([tr, va])
params["scale_pos_weight"] = (dev.y.values == 0).sum() / max((dev.y.values == 1).sum(), 1)
model = lgb.train(params, lgb.Dataset(dev[feats], label=dev.y.values), num_boost_round=500)

joblib.dump({"model": model, "threshold": thr, "features": feats,
             "horizon": HORIZON, "outbreak_inc": OUTBREAK_INC},
            "models/srilanka_outbreak.joblib")

# ---- current district forecasts for the dashboard ----
latest = panel.week_start.max()
scored = add_features(panel, horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
# also score rows whose future label is unknown (the actual forecast frontier)
full = panel.sort_values(["district", "week_start"]).copy()
tmp = add_features(full.assign(_keep=1), horizon=0, outbreak_inc=OUTBREAK_INC)
recent = tmp[tmp.week_start >= latest - pd.Timedelta(weeks=1)].copy()
recent["risk"] = model.predict(recent[feats])
# bands straddle the operating threshold and stay inside [0, 1] whatever thr is
edges = sorted({0.0, round(thr * 0.5, 6), round(thr, 6),
                round(thr + (1.0 - thr) / 2, 6), 1.0})
labels = ["Low", "Moderate", "High", "Very High"][:len(edges) - 1]
recent["risk_band"] = pd.cut(recent.risk, edges, labels=labels, include_lowest=True)
cols = ["district", "week_start", "casos", "p_inc100k", "casos_roll4_mean",
        "precip_roll4_sum", "tempmed", "umidmed", "population_total", "risk", "risk_band"]
out = (recent.sort_values("week_start").groupby("district", as_index=False).last()[cols]
       .sort_values("risk", ascending=False))
out["lat"] = out.district.map(lambda d: DISTRICTS[d][0])
out["lon"] = out.district.map(lambda d: DISTRICTS[d][1])
out.to_csv("reports/srilanka_current_risk.csv", index=False)

# district history for the dashboard trend charts
panel[["district", "week_start", "casos", "p_inc100k", "precip_total_semana",
       "tempmed", "umidmed"]].to_parquet("data/processed/srilanka_history.parquet", index=False)

# feature importance for the "why" panel
imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
imp.to_csv("reports/srilanka_feature_importance.csv", header=["gain"])

print(f"\nforecast week: {out.week_start.max().date()}  (horizon {HORIZON} weeks)")
print(out[["district", "casos", "p_inc100k", "risk", "risk_band"]].head(12).to_string(index=False))
print("\nsaved models/srilanka_outbreak.joblib, reports/srilanka_current_risk.csv")
