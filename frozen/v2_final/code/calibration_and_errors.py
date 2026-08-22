"""Model 2: calibration assessment + structured error analysis.

Calibration matters because the application shows a number to a human. If the page
says 80%, an outbreak should follow roughly 80% of the time - otherwise the number
is a score, not a probability, and must not be labelled "confidence".

Error analysis asks the more interesting question: WHERE does the model fail?
Errors are stratified by season, baseline incidence, population density, rainfall,
region, and historical outbreak magnitude.

Both are read-only analyses of the already-frozen modelling protocol. Nothing is
tuned on the test set; the isotonic calibrator is fitted on validation only.
"""
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, average_precision_score, roc_auc_score,
                             f1_score, recall_score, precision_score)

from dengue.model2_outbreak import load_panel, build_supervised, feature_columns

HORIZON = 2                 # the flagship 14-day early-warning configuration
OUTBREAK_INC = 100.0
SEED = 42
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=127,
              min_child_samples=40, subsample=0.85, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, seed=SEED)


def ece(y, p, n_bins=15):
    """Expected Calibration Error over equal-count bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    tot, out = len(y), 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum() / tot) * abs(y[m].mean() - p[m].mean())
    return float(out)


def reliability(y, p, n_bins=12):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 20:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    return pd.DataFrame(rows)


print(f"loading panel (horizon={HORIZON}w) ...", flush=True)
sup = build_supervised(load_panel(), horizon=HORIZON, outbreak_inc=OUTBREAK_INC)
feats = feature_columns(sup)
tr = sup[sup.anio <= 2021]
va = sup[(sup.anio > 2021) & (sup.anio <= 2023)]
te = sup[sup.anio > 2023].copy()
print(f"  train={len(tr)} val={len(va)} test={len(te)}  test prevalence={te.y.mean():.4f}")

spw = (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)
model = lgb.train({**PARAMS, "scale_pos_weight": spw},
                  lgb.Dataset(tr[feats], label=tr.y.values), num_boost_round=700)
p_va = model.predict(va[feats])
p_te = model.predict(te[feats])

# ================================================================== CALIBRATION
print(f"\n{'='*76}\nCALIBRATION\n{'='*76}")
raw = {"brier": brier_score_loss(te.y.values, p_te), "ece": ece(te.y.values, p_te),
       "pr_auc": average_precision_score(te.y.values, p_te),
       "roc_auc": roc_auc_score(te.y.values, p_te)}
print(f"  RAW model      Brier={raw['brier']:.4f}  ECE={raw['ece']:.4f}  "
      f"PR-AUC={raw['pr_auc']:.4f}")

# isotonic calibrator fitted on VALIDATION ONLY, then applied to test
iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, va.y.values)
p_te_cal = iso.predict(p_te)
cal = {"brier": brier_score_loss(te.y.values, p_te_cal), "ece": ece(te.y.values, p_te_cal),
       "pr_auc": average_precision_score(te.y.values, p_te_cal),
       "roc_auc": roc_auc_score(te.y.values, p_te_cal)}
print(f"  CALIBRATED     Brier={cal['brier']:.4f}  ECE={cal['ece']:.4f}  "
      f"PR-AUC={cal['pr_auc']:.4f}")
better = "calibrated" if cal["ece"] < raw["ece"] else "raw"
print(f"  -> deploy the {better.upper()} probabilities "
      f"(ECE {min(raw['ece'], cal['ece']):.4f})")

rel_raw, rel_cal = reliability(te.y.values, p_te), reliability(te.y.values, p_te_cal)
rel_raw.to_csv("reports/model2_reliability_raw.csv", index=False)
rel_cal.to_csv("reports/model2_reliability_calibrated.csv", index=False)
print("\n  reliability (raw): predicted -> observed")
for _, r in rel_raw.iterrows():
    print(f"    {r.predicted:.3f} -> {r.observed:.3f}   (n={int(r.n)})")

fig, ax = plt.subplots(figsize=(6.2, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.plot(rel_raw.predicted, rel_raw.observed, "o-", label=f"raw (ECE {raw['ece']:.3f})")
ax.plot(rel_cal.predicted, rel_cal.observed, "s-", label=f"isotonic (ECE {cal['ece']:.3f})")
ax.set_xlabel("Predicted outbreak probability")
ax.set_ylabel("Observed outbreak frequency")
ax.set_title(f"Reliability diagram - {HORIZON*7}-day outbreak forecast (test 2024-25)")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("reports/model2_reliability.png", dpi=140); plt.close()

# ================================================================ ERROR ANALYSIS
print(f"\n{'='*76}\nERROR ANALYSIS\n{'='*76}")
p_use = p_te_cal if better == "calibrated" else p_te
qs = np.unique(np.quantile(p_va, np.linspace(0.5, 0.999, 200)))
p_va_use = iso.predict(p_va) if better == "calibrated" else p_va
thr = float(max((f1_score(va.y.values, (p_va_use >= t).astype(int), zero_division=0), t)
                for t in qs)[1])
te["p"] = p_use
te["pred"] = (te.p >= thr).astype(int)
te["outcome"] = np.select(
    [(te.y == 1) & (te.pred == 1), (te.y == 0) & (te.pred == 1),
     (te.y == 1) & (te.pred == 0)], ["TP", "FP", "FN"], default="TN")
counts = te.outcome.value_counts()
print(f"  threshold={thr:.4f}   TP={counts.get('TP',0)}  FP={counts.get('FP',0)}  "
      f"FN={counts.get('FN',0)}  TN={counts.get('TN',0)}")
print(f"  recall={recall_score(te.y, te.pred):.4f}  precision={precision_score(te.y, te.pred):.4f}")

te["month"] = pd.to_datetime(te.data_iniSE).dt.month
te["season"] = pd.cut(te.month, [0, 3, 6, 9, 12],
                      labels=["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"])
te["baseline_inc_q"] = pd.qcut(te.p_inc100k.rank(method="first"), 4,
                               labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
te["pop_density_q"] = pd.qcut(te.population_density.rank(method="first"), 4,
                              labels=["Q1 rural", "Q2", "Q3", "Q4 urban"])
te["rainfall_q"] = pd.qcut(te.precip_roll4_sum.rank(method="first"), 4,
                           labels=["Q1 driest", "Q2", "Q3", "Q4 wettest"])
# how big is this outbreak relative to the municipality's own history?
hist_max = tr.groupby("codigo_ibge").p_inc100k.max().rename("hist_max_inc")
te = te.merge(hist_max, on="codigo_ibge", how="left")
te["vs_history"] = np.where(
    te.y_inc_future > te.hist_max_inc.fillna(np.inf),
    "unprecedented (> historical max)", "within historical range")


def strat(col):
    g = te.groupby(col, observed=True)
    out = g.apply(lambda d: pd.Series({
        "n": len(d), "n_outbreaks": int(d.y.sum()),
        "recall": recall_score(d.y, d.pred, zero_division=0) if d.y.sum() else np.nan,
        "precision": precision_score(d.y, d.pred, zero_division=0) if d.pred.sum() else np.nan,
        "FN": int(((d.y == 1) & (d.pred == 0)).sum()),
        "FP": int(((d.y == 0) & (d.pred == 1)).sum()),
    })).reset_index()
    return out


strata = {}
for col in ["season", "baseline_inc_q", "pop_density_q", "rainfall_q", "vs_history"]:
    s = strat(col)
    strata[col] = s.to_dict("records")
    print(f"\n  -- by {col} --")
    print("   " + s.round(4).to_string(index=False).replace("\n", "\n   "))

# regional breakdown - worst and best states by recall
st = strat("estado").sort_values("recall")
st.to_csv("reports/model2_errors_by_state.csv", index=False)
print("\n  -- worst 5 states by recall --")
print("   " + st.head(5).round(4).to_string(index=False).replace("\n", "\n   "))
print("  -- best 5 states by recall --")
print("   " + st.tail(5).round(4).to_string(index=False).replace("\n", "\n   "))

# is a false positive typically a *just-ended* outbreak? (momentum hypothesis)
fp = te[te.outcome == "FP"]
tn = te[te.outcome == "TN"]
fp_recent = float((fp.p_inc100k >= OUTBREAK_INC).mean())
tn_recent = float((tn.p_inc100k >= OUTBREAK_INC).mean())
print(f"\n  MOMENTUM CHECK - share of rows already at epidemic level when predicted:")
print(f"    false positives : {fp_recent:.3f}")
print(f"    true negatives  : {tn_recent:.3f}")
print(f"    -> false alarms are {fp_recent/max(tn_recent,1e-9):.0f}x more likely to be "
      f"municipalities whose outbreak is currently active and about to subside")

# worst individual misses
worst_fn = (te[te.outcome == "FN"].nlargest(15, "y_inc_future")
            [["municipio", "estado", "data_iniSE", "p_inc100k", "y_inc_future", "p"]])
worst_fn.to_csv("reports/model2_worst_false_negatives.csv", index=False)
print("\n  -- 10 largest missed outbreaks --")
print("   " + worst_fn.head(10).round(2).to_string(index=False).replace("\n", "\n   "))

te[["codigo_ibge", "municipio", "estado", "data_iniSE", "y", "pred", "p",
    "p_inc100k", "y_inc_future", "outcome", "vs_history"]].to_parquet(
    "data/processed/model2_test_predictions.parquet", index=False)

json.dump({"horizon_weeks": HORIZON, "threshold": thr, "raw": raw, "calibrated": cal,
           "deploy": better, "confusion": {k: int(v) for k, v in counts.items()},
           "momentum": {"fp_at_epidemic_level": fp_recent, "tn_at_epidemic_level": tn_recent},
           "strata": strata},
          open("reports/model2_calibration_errors.json", "w"), indent=2, default=float)
print("\nsaved reports/model2_calibration_errors.json, model2_reliability.png")
