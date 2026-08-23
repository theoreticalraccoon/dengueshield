"""SHAP explainability for Model 2 - why does the 14-day forecast work?

Computed on the DEVELOPMENT period (validation 2022-23), not the locked
2024-25 test set. Explanation must not become a back-door route to tuning
against held-out data.
"""

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from dengue.model2_outbreak import build_supervised, feature_columns, load_panel, temporal_split

HORIZON = 2  # the 14-day early-warning model - the flagship result
print(f"SHAP for horizon={HORIZON} weeks", flush=True)
panel = load_panel()
sup = build_supervised(panel, horizon=HORIZON, outbreak_inc=100.0)
feats = feature_columns(sup)
tr, va, te = temporal_split(sup)
print(f"train={len(tr)} val={len(va)} (test held out, not used here)", flush=True)

spw = (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)
m = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=127,
    min_child_samples=40,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    scale_pos_weight=spw,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
).fit(tr[feats], tr.y.values)

# sample the validation set for tractable exact TreeSHAP
rng = np.random.default_rng(42)
idx = rng.choice(len(va), size=min(25000, len(va)), replace=False)
Xs = va.iloc[idx][feats]
print(f"computing TreeSHAP on {len(Xs)} validation rows ...", flush=True)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
if sv.ndim == 3:
    sv = sv[:, :, 1]

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
mean_abs.to_csv("reports/model2_shap_importance.csv", header=["mean_abs_shap"])

# group features into interpretable driver categories
GROUPS = {
    "Recent dengue incidence": [
        "casos",
        "p_inc100k",
        "casos_lag_1",
        "casos_lag_2",
        "casos_lag_4",
        "casos_lag_8",
        "casos_roll4_mean",
        "casos_roll8_mean",
        "casos_growth_1",
        "casos_growth_4",
        "inc_roll4",
        "Rt",
        "Rt_lag_1",
        "Rt_lag_2",
        "Rt_lag_4",
    ],
    "Rainfall": [
        "precip_total_semana",
        "precip_media_semana",
        "precip_max_semana",
        "dias_lluvia_semana",
        "precip_total_semana_lag_1",
        "precip_total_semana_lag_4",
        "precip_roll4_sum",
        "precip_roll8_sum",
    ],
    "Temperature": [
        "tempmin",
        "tempmed",
        "tempmax",
        "tempmed_lag_1",
        "tempmed_lag_4",
        "tempmed_roll4_mean",
        "tempmed_roll8_mean",
        "LST_Day_mean",
        "LST_Night_mean",
    ],
    "Humidity": ["umidmin", "umidmed", "umidmax", "umidmed_lag_1", "umidmed_lag_4"],
    "Vegetation (NDVI/EVI)": ["NDVI_mean", "NDVI_std", "EVI_mean", "EVI_std"],
    "Land cover": [
        "pct_tree_cover",
        "pct_builtup",
        "pct_cropland",
        "pct_water",
        "pct_grassland",
        "elev_mean",
    ],
    "Sanitation": ["water_supply_pct", "sewage_pct", "garbage_collection_pct"],
    "Population": ["population_total", "population_density", "median_age", "sex_ratio", "area_km2"],
    "ENSO": ["nino34", "soi"],
    "Seasonality": ["week_sin", "week_cos"],
}
rows = []
for g, cols in GROUPS.items():
    cs = [c for c in cols if c in mean_abs.index]
    rows.append({"driver": g, "mean_abs_shap": float(mean_abs[cs].sum()), "n_features": len(cs)})
gd = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
gd["pct_of_total"] = 100 * gd.mean_abs_shap / gd.mean_abs_shap.sum()
gd.to_csv("reports/model2_shap_drivers.csv", index=False)

print("\n=== WHAT DRIVES THE 14-DAY FORECAST (grouped SHAP) ===")
mx = gd.mean_abs_shap.max()
for _, r in gd.iterrows():
    bar = "#" * round(28 * r.mean_abs_shap / mx)
    print(f"  {r.driver:24} {bar:28} {r.pct_of_total:5.1f}%")

print("\n=== TOP 20 INDIVIDUAL FEATURES ===")
print(mean_abs.head(20).to_string())

shap.summary_plot(sv, Xs, max_display=20, show=False)
plt.tight_layout()
plt.savefig("reports/model2_shap_summary.png", dpi=140)
plt.close()
gd.set_index("driver")["pct_of_total"].sort_values().plot(kind="barh", figsize=(8, 5))
plt.xlabel("% of total |SHAP|")
plt.title(f"Dengue outbreak drivers ({HORIZON * 7}-day forecast)")
plt.tight_layout()
plt.savefig("reports/model2_shap_drivers.png", dpi=140)
plt.close()
with open("reports/model2_shap.json", "w") as _f:
    json.dump(
        {
            "horizon_weeks": HORIZON,
            "drivers": gd.to_dict("records"),
            "top_features": mean_abs.head(30).to_dict(),
        },
        _f,
        indent=2,
        default=float,
    )
print("\nsaved reports/model2_shap_*.csv/.png/.json")
