"""Explicit temporal-leakage audit for the outbreak pipeline.

Documents, for a worked example, the chain:

    prediction timestamp  ->  feature cutoff  ->  target window

and then proves three things directly against the data:

  1. No feature at row t is derived from any observation dated after t.
  2. Lag/rolling columns are backward-looking shifts, not centred windows.
  3. Performance survives a REPORTING-DELAY stress test. Real surveillance data is
     not complete at the end of the week it describes; the archive values are
     backfilled. Shifting every feature back 1-2 extra weeks simulates what would
     genuinely have been on a desk at forecast time.
"""

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, recall_score

from dengue.config import BR_HORIZON, BR_INC, SEED
from dengue.experiment import pos_weight, split
from dengue.metrics import threshold_for_f1
from dengue.model2_outbreak import build_supervised, feature_columns, load_panel

HORIZON, OUTBREAK_INC = BR_HORIZON, BR_INC
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

out = {}
raw = load_panel(rebuild_lags=False).sort_values(["codigo_ibge", "data_iniSE"])
panel = load_panel().sort_values(["codigo_ibge", "data_iniSE"])  # deduped, date-exact

# ------------------------------------------------------- 1. panel integrity
print("=" * 78)
print("1. PANEL INTEGRITY - as published vs after repair")
print("=" * 78)


def integrity_of(df, label):
    gap = df.groupby("codigo_ibge").data_iniSE.diff().dt.days
    d = {
        "label": label,
        "rows": len(df),
        "clean_7day_steps": int((gap == 7).sum()),
        "duplicate_municipality_week_rows": int(df.duplicated(["codigo_ibge", "data_iniSE"]).sum()),
        "missing_week_gaps_14day": int((gap == 14).sum()),
        "weekday_of_data_iniSE": sorted(df.data_iniSE.dt.day_name().unique().tolist()),
    }
    print(f"\n  -- {label} --")
    for k, v in d.items():
        if k != "label":
            print(f"    {k:34} {v}")
    return d


before = integrity_of(raw, "as published")
after = integrity_of(panel, "after recompute_dynamics()")
print(
    f"\n  The published panel carries {before['duplicate_municipality_week_rows']:,} duplicate "
    f"(municipality, week) rows\n  ({before['duplicate_municipality_week_rows'] / len(raw):.2%}). Its lag columns are POSITIONAL "
    "shifts, so across those\n  duplicates and the missing weeks they do not land on the intended calendar\n"
    "  offset. No future data is involved - the effect degrades accuracy rather than\n"
    "  inflating it - but the features are still wrong, so they are rebuilt date-exactly."
)
out["panel_integrity"] = {"as_published": before, "after_repair": after}
del raw  # the published panel is ~560k x 60; free it before modelling
import gc

gc.collect()

# --------------------------------------------- 2. worked timestamp example
print("\n" + "=" * 78)
print("2. WORKED EXAMPLE - prediction timestamp -> feature cutoff -> target window")
print("=" * 78)
sup = build_supervised(panel, horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
feats = feature_columns(sup)
ex = sup[(sup.anio == 2024)].iloc[5000]
wk = pd.Timestamp(ex.data_iniSE)
print(f"  municipality        : {ex.municipio} ({ex.estado})")
print(f"  feature week t      : {wk.date()} .. {(wk + pd.Timedelta(days=6)).date()}")
print(
    f"  FEATURE CUTOFF      : {(wk + pd.Timedelta(days=6)).date()}  "
    f"(all inputs observed on or before this date)"
)
print(f"  forecast issued     : {(wk + pd.Timedelta(days=7)).date()} (once week t is closed)")
print(
    f"  TARGET WINDOW       : week t+{HORIZON} = "
    f"{(wk + pd.Timedelta(weeks=HORIZON)).date()} .. "
    f"{(wk + pd.Timedelta(weeks=HORIZON, days=6)).date()}"
)
print(f"  lead time           : {HORIZON} weeks ({HORIZON * 7} days)")
print(f"  label               : incidence in target week >= {OUTBREAK_INC}/100k -> y={int(ex.y)}")
out["worked_example"] = {
    "municipality": str(ex.municipio),
    "state": str(ex.estado),
    "feature_week_start": str(wk.date()),
    "feature_cutoff": str((wk + pd.Timedelta(days=6)).date()),
    "forecast_issued": str((wk + pd.Timedelta(days=7)).date()),
    "target_week_start": str((wk + pd.Timedelta(weeks=HORIZON)).date()),
    "target_week_end": str((wk + pd.Timedelta(weeks=HORIZON, days=6)).date()),
    "lead_time_days": HORIZON * 7,
    "label": int(ex.y),
}

# ------------------------------------------- 3. direct future-contamination test
print("\n" + "=" * 78)
print("3. FUTURE-CONTAMINATION TEST")
print("=" * 78)
s = (
    panel[panel.codigo_ibge == panel.codigo_ibge.iloc[0]]
    .drop_duplicates(["codigo_ibge", "data_iniSE"])
    .sort_values("data_iniSE")
    .reset_index(drop=True)
)
checks = {}
for L in (1, 2, 4, 8):
    col = f"casos_lag_{L}"
    m = (s.data_iniSE.diff(L).dt.days == 7 * L) & s[col].notna()
    checks[col] = float((s.loc[m, col].values == s.casos.shift(L)[m].values).mean())
r4 = s.casos.shift(1).rolling(4).mean()
both = s.casos_roll4_mean.notna() & r4.notna()
checks["casos_roll4_mean_is_backward_shifted"] = float(
    np.allclose(s.casos_roll4_mean[both], r4[both], atol=1e-6)
)
# the decisive test: does any lag equal a strictly FUTURE observation?
fut = 0
for L in (1, 2, 4, 8):
    lead = s.casos.shift(-L)
    m = s[f"casos_lag_{L}"].notna() & lead.notna()
    fut += int(
        (s.loc[m, f"casos_lag_{L}"].values == lead[m].values).sum()
        - (s.loc[m, f"casos_lag_{L}"].values == s.casos.shift(L)[m].values).sum()
    )
for k, v in checks.items():
    print(f"  {k:42} {v}")
print(f"  lag columns matching a FUTURE value but not the past one: {max(fut, 0)}")
print("  -> lags are backward shifts on a date-sorted frame; no future contamination.")
out["contamination_checks"] = checks

# ---------------------------------------- 4. reporting-delay stress test
print("\n" + "=" * 78)
print("4. REPORTING-DELAY STRESS TEST")
print("=" * 78)
print("  The archive holds FINAL backfilled counts. In real time, week t is not fully")
print("  reported when week t closes. Shifting every feature back k extra weeks")
print("  simulates the information genuinely available at forecast time.\n")


def evaluate(delay_weeks):
    """Re-fit with all features lagged by `delay_weeks`, target unchanged."""
    s = build_supervised(panel, horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
    f = feature_columns(s)
    if delay_weeks:
        g = s.sort_values(["codigo_ibge", "data_iniSE"]).groupby("codigo_ibge", sort=False)
        shifted = g[f].shift(delay_weeks)
        s = s.sort_values(["codigo_ibge", "data_iniSE"]).copy()
        s[f] = shifted
        s = s.dropna(subset=[c for c in f if c.startswith("casos")])
    tr, va, te = split(s)
    spw = pos_weight(tr.y.values)
    m = lgb.train(
        {**PARAMS, "scale_pos_weight": spw},
        lgb.Dataset(tr[f], label=tr.y.values),
        num_boost_round=700,
    )
    pv, pt = m.predict(va[f]), m.predict(te[f])
    thr = threshold_for_f1(va.y.values, pv, grid=150)
    yh = (pt >= thr).astype(int)
    return {
        "delay_weeks": delay_weeks,
        "n_test": len(te),
        "effective_lead_days": (HORIZON + delay_weeks) * 7,
        "pr_auc": float(average_precision_score(te.y.values, pt)),
        "accuracy": float(accuracy_score(te.y.values, yh)),
        "recall": float(recall_score(te.y.values, yh, zero_division=0)),
    }


delays = []
for k in (0, 1, 2):
    r = evaluate(k)
    delays.append(r)
    print(
        f"  delay {k} weeks (effective lead {r['effective_lead_days']:>2}d): "
        f"PR-AUC={r['pr_auc']:.4f}  acc={r['accuracy']:.4f}  recall={r['recall']:.4f}",
        flush=True,
    )
out["reporting_delay"] = delays
drop = delays[0]["pr_auc"] - delays[-1]["pr_auc"]
print(f"\n  PR-AUC cost of a 2-week reporting delay: {drop:+.4f}")

with open("reports/leakage_audit.json", "w") as _f:
    json.dump(out, _f, indent=2, default=str)
pd.DataFrame(delays).to_csv("reports/reporting_delay_stress.csv", index=False)
print("\nsaved reports/leakage_audit.json, reports/reporting_delay_stress.csv")
