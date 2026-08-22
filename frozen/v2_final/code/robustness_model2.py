"""Verification suite for Model 2.

A single train/test split can flatter a model. This runs:
  1. Rolling-origin backtest  - retrain each year, always predict the next unseen year.
  2. Horizon sweep            - 2 / 4 / 8 weeks ahead.
  3. Threshold sweep          - what counts as an "outbreak".
  4. Shuffled-label control   - performance must collapse to chance.
"""
import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, precision_score, f1_score)
from lightgbm import LGBMClassifier
from dengue.model2_outbreak import load_panel, build_supervised, feature_columns

def mk(spw):
    return LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=127,
                          min_child_samples=40, subsample=0.85, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=spw,
                          random_state=42, n_jobs=-1, verbose=-1)

def tune_thr(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 150)))
    return max((f1_score(y, (p>=t).astype(int), zero_division=0), t) for t in qs)[1]

def evaluate(ytr, Xtr, yva, Xva, yte, Xte):
    m = mk((ytr==0).sum()/max((ytr==1).sum(),1)).fit(Xtr, ytr)
    pv, pt = m.predict_proba(Xva)[:,1], m.predict_proba(Xte)[:,1]
    thr = tune_thr(yva, pv); yh = (pt>=thr).astype(int)
    return {"roc_auc":roc_auc_score(yte,pt),"pr_auc":average_precision_score(yte,pt),
            "accuracy":accuracy_score(yte,yh),"recall":recall_score(yte,yh,zero_division=0),
            "precision":precision_score(yte,yh,zero_division=0),
            "f1":f1_score(yte,yh,zero_division=0),"n_test":len(yte),
            "test_prevalence":float(yte.mean()),
            "trivial_acc":float(max(yte.mean(),1-yte.mean()))}

print("loading panel ...", flush=True)
panel = load_panel()
results = {}

# ---------- 1. rolling-origin backtest ----------
print("\n=== 1. ROLLING-ORIGIN BACKTEST (horizon=4, outbreak>=100/100k) ===", flush=True)
sup = build_supervised(panel, horizon=4, outbreak_inc=100.0)
feats = feature_columns(sup)
rows=[]
for test_year in range(2019, 2026):
    tr = sup[sup.anio <= test_year-2]; va = sup[sup.anio == test_year-1]; te = sup[sup.anio == test_year]
    if len(te)==0 or te.y.sum()<10 or va.y.sum()<5: continue
    r = evaluate(tr.y.values, tr[feats], va.y.values, va[feats], te.y.values, te[feats])
    r["test_year"]=test_year; rows.append(r)
    print(f"  {test_year}: PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} acc={r['accuracy']:.4f} "
          f"(trivial {r['trivial_acc']:.4f}) recall={r['recall']:.4f} prec={r['precision']:.4f}", flush=True)
bt = pd.DataFrame(rows); results["rolling_origin"]=rows
print(f"  MEAN over years: PR-AUC={bt.pr_auc.mean():.4f}  acc={bt.accuracy.mean():.4f}  recall={bt.recall.mean():.4f}")

# ---------- 2. horizon sweep ----------
print("\n=== 2. HORIZON SWEEP (test 2024-25) ===", flush=True)
rows=[]
for h in [2,4,8]:
    s = build_supervised(panel, horizon=h, outbreak_inc=100.0); f=feature_columns(s)
    tr=s[s.anio<=2021]; va=s[(s.anio>2021)&(s.anio<=2023)]; te=s[s.anio>2023]
    r=evaluate(tr.y.values,tr[f],va.y.values,va[f],te.y.values,te[f]); r["horizon"]=h; rows.append(r)
    print(f"  h={h:>2}w: PR-AUC={r['pr_auc']:.4f} acc={r['accuracy']:.4f} (trivial {r['trivial_acc']:.4f}) "
          f"recall={r['recall']:.4f} prec={r['precision']:.4f}", flush=True)
results["horizons"]=rows

# ---------- 3. outbreak-threshold sweep ----------
print("\n=== 3. OUTBREAK THRESHOLD SWEEP (horizon=4) ===", flush=True)
rows=[]
for inc in [50.0,100.0,300.0]:
    s = build_supervised(panel, horizon=4, outbreak_inc=inc); f=feature_columns(s)
    tr=s[s.anio<=2021]; va=s[(s.anio>2021)&(s.anio<=2023)]; te=s[s.anio>2023]
    r=evaluate(tr.y.values,tr[f],va.y.values,va[f],te.y.values,te[f]); r["outbreak_inc"]=inc; rows.append(r)
    print(f"  >={inc:>5.0f}/100k: PR-AUC={r['pr_auc']:.4f} acc={r['accuracy']:.4f} "
          f"(trivial {r['trivial_acc']:.4f}) recall={r['recall']:.4f} prev={r['test_prevalence']:.4f}", flush=True)
results["thresholds"]=rows

# ---------- 4. shuffled-label control ----------
print("\n=== 4. SHUFFLED-LABEL CONTROL (must collapse to chance) ===", flush=True)
s=build_supervised(panel,horizon=4,outbreak_inc=100.0); f=feature_columns(s)
tr=s[s.anio<=2021].copy(); va=s[(s.anio>2021)&(s.anio<=2023)].copy(); te=s[s.anio>2023].copy()
rng=np.random.default_rng(0)
r=evaluate(rng.permutation(tr.y.values),tr[f],rng.permutation(va.y.values),va[f],te.y.values,te[f])
print(f"  shuffled-train: PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
      f"(chance PR-AUC ~= {te.y.mean():.4f})")
results["shuffled_control"]=r

json.dump(results, open("reports/model2_robustness.json","w"), indent=2, default=float)
print("\nsaved reports/model2_robustness.json")
