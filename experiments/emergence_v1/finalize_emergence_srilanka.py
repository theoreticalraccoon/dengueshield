"""Train the Sri Lanka EMERGENCE model and produce dual district forecasts.

Gives the application two complementary numbers per district:

    continuation risk  - will an existing outbreak persist?   (frozen v2 model)
    emergence risk     - is a NEW outbreak about to begin?    (this model)

Emergence is only asked of districts not currently in outbreak, which is exactly
where the continuation model is blind.
"""
import sys, json, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score)

from dengue.srilanka import build_panel, COMMON_FEATURES, DISTRICTS

HERE = Path(__file__).parent
HORIZON, INC, QUIET = 4, 9.9, 2      # Sri Lanka-calibrated epidemic threshold
SEED = 42
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 10.0, "verbose": -1, "seed": SEED}


def add_base_features(df):
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    for L in (1, 2, 4, 8):
        df[f"casos_lag_{L}"] = g["casos"].shift(L)
    df["casos_roll4_mean"] = g["casos"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["casos_roll8_mean"] = g["casos"].transform(lambda s: s.shift(1).rolling(8).mean())
    df["casos_growth_1"] = df.casos / df.casos_lag_1.clip(lower=1)
    df["casos_growth_4"] = df.casos / df.casos_lag_4.clip(lower=1)
    df["inc_roll4"] = df.casos_roll4_mean / df.population_total.clip(lower=1) * 1e5
    for c in ("precip_total_semana", "tempmed", "umidmed"):
        df[f"{c}_lag_1"] = g[c].shift(1)
        df[f"{c}_lag_4"] = g[c].shift(4)
    df["precip_roll4_sum"] = g["precip_total_semana"].transform(lambda s: s.shift(1).rolling(4).sum())
    df["precip_roll8_sum"] = g["precip_total_semana"].transform(lambda s: s.shift(1).rolling(8).sum())
    df["tempmed_roll4_mean"] = g["tempmed"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["tempmed_roll8_mean"] = g["tempmed"].transform(lambda s: s.shift(1).rolling(8).mean())
    wk = df.week_start.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * wk / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * wk / 52.0)
    return df


def label_emergence(df, horizon=HORIZON, inc=INC, quiet=QUIET, keep_ineligible=False):
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    df["_above"] = (df.p_inc100k >= inc).astype(float)
    fut = pd.concat([g["_above"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y"] = (fut.max(axis=1) >= 1).astype(float)
    df.loc[fut.isna().any(axis=1), "y"] = np.nan
    dtf = g["week_start"].shift(-horizon)
    df.loc[(dtf - df.week_start).dt.days != 7 * horizon, "y"] = np.nan
    elig = df._above == 0
    for k in range(1, quiet + 1):
        elig &= (g["_above"].shift(k) == 0)
    df["eligible"] = elig
    if keep_ineligible:
        return df
    return df[elig].dropna(subset=["y"]).reset_index(drop=True)


panel = add_base_features(build_panel())
feats = [c for c in COMMON_FEATURES if c in panel.columns]
em = label_emergence(panel)
tr = em[em.anio <= 2021]
va = em[(em.anio > 2021) & (em.anio <= 2023)]
te = em[em.anio > 2023]
print(f"eligible district-weeks: train={len(tr)} val={len(va)} test={len(te)}")
print(f"emergence rate: train={tr.y.mean():.4f} test={te.y.mean():.4f} "
      f"({int(te.y.sum())} positive test events)")

spw = (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)
dev_model = lgb.train({**PARAMS, "scale_pos_weight": spw},
                      lgb.Dataset(tr[feats], label=tr.y.values), num_boost_round=400)
pv, pt = dev_model.predict(va[feats]), dev_model.predict(te[feats])
qs = np.unique(np.quantile(pv, np.linspace(0.5, 0.999, 200)))
thr = float(max((f1_score(va.y.values, (pv >= t).astype(int), zero_division=0), t) for t in qs)[1])
yh = (pt >= thr).astype(int)
metrics = {
    "pr_auc": float(average_precision_score(te.y.values, pt)),
    "roc_auc": float(roc_auc_score(te.y.values, pt)),
    "recall": float(recall_score(te.y.values, yh, zero_division=0)),
    "precision": float(precision_score(te.y.values, yh, zero_division=0)),
    "prevalence": float(te.y.mean()), "n_positive": int(te.y.sum()),
    "threshold": thr, "horizon_weeks": HORIZON, "outbreak_inc": INC,
}
# baseline for context
base = average_precision_score(te.y.values, te.p_inc100k.values)
metrics["baseline_persistence_pr_auc"] = float(base)
print(f"held-out test: PR-AUC={metrics['pr_auc']:.4f} (persistence baseline {base:.4f}) "
      f"recall={metrics['recall']:.4f}")

# production model: refit on train+val
dev = pd.concat([tr, va])
PARAMS["scale_pos_weight"] = (dev.y.values == 0).sum() / max((dev.y.values == 1).sum(), 1)
model = lgb.train(PARAMS, lgb.Dataset(dev[feats], label=dev.y.values), num_boost_round=400)
joblib.dump({"model": model, "threshold": thr, "features": feats, "horizon": HORIZON,
             "outbreak_inc": INC, "quiet_weeks": QUIET, "metrics": metrics},
            ROOT / "models" / "srilanka_emergence.joblib")

# ---- dual forecast table for the app ----
scored = label_emergence(panel, keep_ineligible=True)
latest = panel.week_start.max()
recent = scored[scored.week_start >= latest - pd.Timedelta(weeks=1)].copy()
recent["emergence_risk"] = model.predict(recent[feats])
recent.loc[~recent.eligible, "emergence_risk"] = np.nan     # question not asked
recent["currently_in_outbreak"] = recent.p_inc100k >= INC

cont = joblib.load(ROOT / "models" / "srilanka_outbreak.joblib")
recent["continuation_risk"] = cont["model"].predict(recent[cont["features"]])

cols = ["district", "week_start", "casos", "p_inc100k", "currently_in_outbreak",
        "continuation_risk", "emergence_risk", "casos_roll4_mean", "precip_roll4_sum",
        "tempmed", "umidmed", "population_total"]
out = (recent.sort_values("week_start").groupby("district", as_index=False).last()[cols])
out["lat"] = out.district.map(lambda d: DISTRICTS[d][0])
out["lon"] = out.district.map(lambda d: DISTRICTS[d][1])
out = out.sort_values("continuation_risk", ascending=False)
out.to_csv(ROOT / "reports" / "srilanka_dual_risk.csv", index=False)
with open(HERE / "srilanka_emergence_metrics.json", "w") as _f:
    json.dump(metrics, _f, indent=2)

print(f"\nforecast week {out.week_start.max().date()}  (emergence horizon 1-{HORIZON} weeks)")
print(out[["district", "casos", "p_inc100k", "currently_in_outbreak",
           "continuation_risk", "emergence_risk"]].round(3).head(14).to_string(index=False))
print("\nsaved models/srilanka_emergence.joblib, reports/srilanka_dual_risk.csv")
