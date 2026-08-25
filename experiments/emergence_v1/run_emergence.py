"""EXPERIMENT: emerging-outbreak detection (exploratory extension to frozen v2).

The v2 outbreak model answers "will this place be in outbreak in 14 days?" and does
it very well - but its recall is ~0.92 where incidence is already high and ~0 where
it is low. It tracks trajectory. Early warning needs the opposite: catching the
transition INTO an outbreak.

TARGET
    Eligible rows are only those NOT currently in outbreak (incidence < threshold at
    time t), optionally also quiet for the preceding `quiet_weeks`.
        positive  : incidence crosses the threshold at some point in t+1 .. t+H
        negative  : stays below the threshold throughout t+1 .. t+H
    Rows already above the threshold at t are DROPPED, not labelled negative - the
    question is not asked of them.

BASELINES (all must be beaten to claim the model adds anything)
    persistence   - current incidence as a score
    growth rate   - week-on-week and 4-week case growth
    moving average- 4-week rolling incidence

FEATURE SETS
    incidence only / environment only / combined - the ablation that matters most,
    because environment was near-worthless for outbreak CONTINUATION and may behave
    completely differently for EMERGENCE.

Nothing here touches frozen/v2_final/. Results live in this directory.
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue.model2_outbreak import feature_columns, load_panel

HERE = Path(__file__).parent
SEED = 42
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 60,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "verbose": -1,
    "seed": SEED,
}

HISTORICAL = [
    "casos",
    "p_inc100k",
    "Rt",
    "casos_lag_1",
    "casos_lag_2",
    "casos_lag_4",
    "casos_lag_8",
    "Rt_lag_1",
    "Rt_lag_2",
    "Rt_lag_4",
    "casos_roll4_mean",
    "casos_roll8_mean",
    "casos_growth_1",
    "casos_growth_4",
    "inc_roll4",
]


def build_emergence(df, horizon=4, outbreak_inc=100.0, quiet_weeks=2):
    """Label the TRANSITION into outbreak, among places not currently in outbreak."""
    df = (
        df.drop_duplicates(["codigo_ibge", "data_iniSE"])
        .sort_values(["codigo_ibge", "data_iniSE"])
        .copy()
    )
    g = df.groupby("codigo_ibge", sort=False)

    # seasonality + growth dynamics (same construction as the main pipeline)
    wk = df.data_iniSE.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * wk / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * wk / 52.0)
    df["casos_growth_1"] = df.casos / df.casos_lag_1.clip(lower=1)
    df["casos_growth_4"] = df.casos / df.casos_lag_4.clip(lower=1)
    df["inc_roll4"] = df.casos_roll4_mean / df["pop"].clip(lower=1) * 1e5

    above = (df.p_inc100k >= outbreak_inc).astype(float)
    df["_above"] = above

    # does the threshold get crossed anywhere in t+1 .. t+H ?
    fut = pd.concat([g["_above"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y"] = (fut.max(axis=1) >= 1).astype(float)
    df.loc[fut.isna().any(axis=1), "y"] = np.nan

    # contiguity: t+H must really be H weeks later
    dtf = g["data_iniSE"].shift(-horizon)
    df.loc[(dtf - df.data_iniSE).dt.days != 7 * horizon, "y"] = np.nan

    # ELIGIBILITY: not currently in outbreak, and quiet for the preceding weeks
    elig = df._above == 0
    for k in range(1, quiet_weeks + 1):
        elig &= g["_above"].shift(k) == 0
    df = df[elig].dropna(subset=["y"]).copy()

    # baselines available at time t
    df["base_persistence"] = df.p_inc100k
    df["base_growth"] = df.casos_growth_1.fillna(1.0) * df.casos_growth_4.fillna(1.0)
    df["base_movavg"] = df.inc_roll4.fillna(0.0)
    return df.drop(columns=["_above"])


def score(y, p, thr, name, extra=None):
    yh = (p >= thr).astype(int)
    d = {
        "condition": name,
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, yh),
        "recall": recall_score(y, yh, zero_division=0),
        "precision": precision_score(y, yh, zero_division=0),
        "f1": f1_score(y, yh, zero_division=0),
        "n_test": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "lift_over_prevalence": float(average_precision_score(y, p) / max(y.mean(), 1e-9)),
    }
    if extra:
        d.update(extra)
    return d


def tune_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.9995, 200)))
    return float(max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in qs)[1])


def fit_eval(tr, va, te, feats, name):
    spw = (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)
    m = lgb.train(
        {**PARAMS, "scale_pos_weight": spw},
        lgb.Dataset(tr[feats], label=tr.y.values),
        num_boost_round=600,
    )
    pv, pt = m.predict(va[feats]), m.predict(te[feats])
    return m, score(te.y.values, pt, tune_thr(va.y.values, pv), name, {"n_features": len(feats)})


print("loading panel ...", flush=True)
panel = load_panel()
all_feats_ref = None
results = {}

for HORIZON, INC in [(4, 100.0), (2, 100.0), (4, 50.0)]:
    tag = f"h{HORIZON}_inc{int(INC)}"
    print(f"\n{'=' * 78}\nEMERGENCE  horizon=1..{HORIZON} weeks   threshold={INC}/100k")
    print("=" * 78, flush=True)
    em = build_emergence(panel, horizon=HORIZON, outbreak_inc=INC, quiet_weeks=2)
    feats = [c for c in feature_columns(em) if c in em.columns]
    if all_feats_ref is None:
        all_feats_ref = feats
    HIST = [c for c in HISTORICAL if c in feats]
    ENV = [c for c in feats if c not in HIST]

    tr = em[em.anio <= 2021]
    va = em[(em.anio > 2021) & (em.anio <= 2023)]
    te = em[em.anio > 2023]
    print(f"  eligible rows: train={len(tr)} val={len(va)} test={len(te)}")
    print(
        f"  emergence rate: train={tr.y.mean():.4f} test={te.y.mean():.4f} "
        f"({int(te.y.sum())} positive test events)"
    )
    if len(te) == 0 or te.y.sum() < 30:
        print("  too few positive events - skipping")
        continue

    rows = []
    # ---- trivial baselines ----
    for bname, col in [
        ("baseline_persistence", "base_persistence"),
        ("baseline_growth_rate", "base_growth"),
        ("baseline_moving_average", "base_movavg"),
    ]:
        pv, pt = va[col].values, te[col].values
        rows.append(score(te.y.values, pt, tune_thr(va.y.values, pv), bname, {"n_features": 1}))
        print(
            f"  {bname:26} PR-AUC={rows[-1]['pr_auc']:.4f}  "
            f"recall={rows[-1]['recall']:.4f}  lift={rows[-1]['lift_over_prevalence']:.2f}x",
            flush=True,
        )

    # ---- feature-set ablation ----
    for mname, cols in [
        ("model_incidence_only", HIST),
        ("model_environment_only", ENV),
        ("model_combined", feats),
    ]:
        _, r = fit_eval(tr, va, te, cols, mname)
        rows.append(r)
        print(
            f"  {mname:26} k={len(cols):>2}  PR-AUC={r['pr_auc']:.4f}  "
            f"recall={r['recall']:.4f}  lift={r['lift_over_prevalence']:.2f}x",
            flush=True,
        )

    df = pd.DataFrame(rows)
    results[tag] = rows
    inc_o = df.loc[df.condition == "model_incidence_only", "pr_auc"].iloc[0]
    env_o = df.loc[df.condition == "model_environment_only", "pr_auc"].iloc[0]
    comb = df.loc[df.condition == "model_combined", "pr_auc"].iloc[0]
    best_base = df[df.condition.str.startswith("baseline")].pr_auc.max()
    print(f"\n  environment adds {comb - inc_o:+.4f} PR-AUC beyond incidence alone")
    print(f"  environment ALONE vs incidence alone: {env_o - inc_o:+.4f}")
    print(f"  best model beats best trivial baseline by {comb - best_base:+.4f}")
    results[tag + "_summary"] = {
        "incidence_only": float(inc_o),
        "environment_only": float(env_o),
        "combined": float(comb),
        "best_baseline": float(best_base),
        "env_gain_over_incidence": float(comb - inc_o),
        "model_gain_over_baseline": float(comb - best_base),
        "test_prevalence": float(te.y.mean()),
        "n_positive": int(te.y.sum()),
    }

HERE.mkdir(parents=True, exist_ok=True)
with open(HERE / "emergence_results.json", "w") as _f:
    json.dump(results, _f, indent=2, default=float)
flat = pd.concat(
    [pd.DataFrame(v).assign(config=k) for k, v in results.items() if isinstance(v, list)],
    ignore_index=True,
)
flat.to_csv(HERE / "emergence_results.csv", index=False)
print(f"\nsaved {HERE / 'emergence_results.csv'}")
