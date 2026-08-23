"""Model 2: spatial generalization + information-source ablation.

Two questions the temporal holdout cannot answer:

  1. SPATIAL HOLDOUT - can the model predict places it has never seen? A model can
     score well temporally by memorising "municipality X always gets dengue". Holding
     out whole municipalities, and then whole states, separates memorised place
     identity from genuinely learned environmental relationships.

  2. INFORMATION ABLATION - how much of the skill is just epidemiological momentum?
     Historical-only vs environment-only vs combined, all against persistence.

Neither experiment tunes anything. Thresholds come from a validation split that is
disjoint from the reported test rows in every condition.
"""

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue.config import BR_HORIZON, BR_INC, SEED
from dengue.experiment import pos_weight, split
from dengue.model2_outbreak import build_supervised, feature_columns, load_panel

HORIZON = BR_HORIZON
OUTBREAK_INC = BR_INC

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

# ---- information sources -------------------------------------------------------
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


def fit_predict(tr, va, te, feats):
    spw = pos_weight(tr.y.values)
    m = lgb.train(
        {**PARAMS, "scale_pos_weight": spw},
        lgb.Dataset(tr[feats], label=tr.y.values),
        num_boost_round=700,
    )
    return m, m.predict(va[feats]), m.predict(te[feats])


def tune_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 200)))
    return float(max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in qs)[1])


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
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "n_test": len(y),
        "test_prevalence": float(y.mean()),
        "trivial_accuracy": float(max(y.mean(), 1 - y.mean())),
    }
    if extra:
        d.update(extra)
    return d


print(f"loading panel (horizon={HORIZON}w, outbreak>={OUTBREAK_INC}/100k) ...", flush=True)
sup = build_supervised(load_panel(), horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
FEATS = feature_columns(sup)
HIST = [c for c in HISTORICAL if c in FEATS]
ENV = [c for c in FEATS if c not in HIST]
print(f"  rows={len(sup)}  features={len(FEATS)}  historical={len(HIST)}  environmental={len(ENV)}")
print(f"  municipalities={sup.codigo_ibge.nunique()}  states={sup.estado.nunique()}")

results = {}

# =============================================================== 1. SPATIAL HOLDOUT
print(f"\n{'=' * 78}\n1. SPATIAL HOLDOUT - can it predict unseen places?\n{'=' * 78}", flush=True)
rng = np.random.default_rng(SEED)
munis = sup.codigo_ibge.unique()
rng.shuffle(munis)
n_hold = round(0.20 * len(munis))
held_m = set(munis[:n_hold])
seen = sup[~sup.codigo_ibge.isin(held_m)]
unseen = sup[sup.codigo_ibge.isin(held_m)]
print(f"  seen municipalities={len(munis) - n_hold}  held-out={n_hold}")

rows = []

# (a) reference: temporal holdout, all municipalities seen
tr, va, te_seen = split(seen)
_, pv, pt = fit_predict(tr, va, te_seen, FEATS)
thr = tune_thr(va.y.values, pv)
rows.append(score(te_seen.y.values, pt, thr, "A_temporal_seen_municipalities"))
print(
    f"  A temporal, SEEN munis      PR-AUC={rows[-1]['pr_auc']:.4f} acc={rows[-1]['accuracy']:.4f}"
)

# (b) spatiotemporal: unseen municipalities AND unseen years - the strictest test
te_unseen = unseen[unseen.anio > 2023]
va_unseen = unseen[(unseen.anio > 2021) & (unseen.anio <= 2023)]
m, pv_u, pt_u = fit_predict(tr, va_unseen, te_unseen, FEATS)
thr_u = tune_thr(va_unseen.y.values, pv_u)
rows.append(score(te_unseen.y.values, pt_u, thr_u, "B_spatiotemporal_unseen_municipalities"))
print(
    f"  B spatio-temporal, UNSEEN   PR-AUC={rows[-1]['pr_auc']:.4f} acc={rows[-1]['accuracy']:.4f}"
)

# (c) pure spatial: unseen municipalities, same period the model trained on
te_sp = unseen[unseen.anio <= 2021]
_, _, pt_sp = fit_predict(tr, va, te_sp, FEATS)
rows.append(score(te_sp.y.values, pt_sp, thr, "C_spatial_only_unseen_municipalities"))
print(
    f"  C spatial only, UNSEEN      PR-AUC={rows[-1]['pr_auc']:.4f} acc={rows[-1]['accuracy']:.4f}"
)

# (d) persistence on the unseen spatiotemporal rows
rows.append(
    score(
        te_unseen.y.values,
        te_unseen.baseline_persistence.values,
        0.5,
        "D_persistence_unseen_municipalities",
    )
)
print(
    f"  D persistence on UNSEEN     PR-AUC={rows[-1]['pr_auc']:.4f} acc={rows[-1]['accuracy']:.4f}"
)

# (e) leave-whole-regions-out: hold out entire states, the hardest spatial test
print("\n  leave-one-state-group-out (entire states unseen):", flush=True)
states = sorted(sup.estado.dropna().unique())
rng.shuffle(states)
folds = np.array_split(np.array(states, dtype=object), 5)
state_rows = []
for i, hs in enumerate(folds):
    hs = set(hs)
    s_tr = sup[(~sup.estado.isin(hs)) & (sup.anio <= 2021)]
    s_va = sup[(~sup.estado.isin(hs)) & (sup.anio > 2021) & (sup.anio <= 2023)]
    s_te = sup[(sup.estado.isin(hs)) & (sup.anio > 2023)]
    if len(s_te) == 0 or s_te.y.sum() < 10:
        continue
    _, spv, spt = fit_predict(s_tr, s_va, s_te, FEATS)
    r = score(
        s_te.y.values,
        spt,
        tune_thr(s_va.y.values, spv),
        f"state_fold_{i}",
        {"held_states": sorted(hs)},
    )
    state_rows.append(r)
    print(
        f"    fold {i}: {len(hs)} states, n={r['n_test']:>6}  PR-AUC={r['pr_auc']:.4f} "
        f"acc={r['accuracy']:.4f} recall={r['recall']:.4f}",
        flush=True,
    )
if state_rows:
    sdf = pd.DataFrame(state_rows)
    print(
        f"    MEAN over state folds: PR-AUC={sdf.pr_auc.mean():.4f} acc={sdf.accuracy.mean():.4f}"
    )

results["spatial"] = rows
results["state_folds"] = state_rows

# ============================================================ 2. INFORMATION ABLATION
print(
    f"\n{'=' * 78}\n2. INFORMATION ABLATION - where does the skill come from?\n{'=' * 78}",
    flush=True,
)
tr, va, te = split(sup)
abl = []
for name, cols in [
    ("A_historical_only", HIST),
    ("B_environmental_only", ENV),
    ("C_historical_plus_environment", FEATS),
]:
    _, pv, pt = fit_predict(tr, va, te, cols)
    r = score(te.y.values, pt, tune_thr(va.y.values, pv), name, {"n_features": len(cols)})
    abl.append(r)
    print(
        f"  {name:32} k={len(cols):>2}  PR-AUC={r['pr_auc']:.4f}  acc={r['accuracy']:.4f}  "
        f"recall={r['recall']:.4f}",
        flush=True,
    )
abl.append(
    score(
        te.y.values,
        te.baseline_persistence.values,
        0.5,
        "D_persistence_baseline",
        {"n_features": 1},
    )
)
print(
    f"  {'D_persistence_baseline':32} k= 1  PR-AUC={abl[-1]['pr_auc']:.4f}  "
    f"acc={abl[-1]['accuracy']:.4f}  recall={abl[-1]['recall']:.4f}"
)
results["information_ablation"] = abl

comb = next(r for r in abl if r["condition"] == "C_historical_plus_environment")
hist_o = next(r for r in abl if r["condition"] == "A_historical_only")
env_o = next(r for r in abl if r["condition"] == "B_environmental_only")
pers = next(r for r in abl if r["condition"] == "D_persistence_baseline")
results["ablation_summary"] = {
    "environment_gain_over_historical": comb["pr_auc"] - hist_o["pr_auc"],
    "historical_gain_over_persistence": hist_o["pr_auc"] - pers["pr_auc"],
    "environment_alone_vs_persistence": env_o["pr_auc"] - pers["pr_auc"],
}
print(
    f"\n  environment adds {results['ablation_summary']['environment_gain_over_historical']:+.4f} "
    f"PR-AUC beyond history alone"
)
print(
    f"  history adds {results['ablation_summary']['historical_gain_over_persistence']:+.4f} "
    f"PR-AUC beyond persistence"
)
print(
    f"  environment ALONE vs persistence: "
    f"{results['ablation_summary']['environment_alone_vs_persistence']:+.4f}"
)

with open("reports/model2_spatial_ablation.json", "w") as _f:
    json.dump(results, _f, indent=2, default=float)
pd.DataFrame(rows + state_rows).to_csv("reports/model2_spatial_holdout.csv", index=False)
pd.DataFrame(abl).to_csv("reports/model2_information_ablation.csv", index=False)
print("\nsaved reports/model2_spatial_holdout.csv, model2_information_ablation.csv")
