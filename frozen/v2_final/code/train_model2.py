import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss,
                             confusion_matrix)
from dengue.model2_outbreak import (load_panel, build_supervised, temporal_split,
                                    feature_columns, seasonal_climatology,
                                    HORIZON, OUTBREAK_INC, MIN_POP)

def report(y, p, thr, name):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0,1]).ravel()
    return {"model": name, "threshold": float(thr),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y))>1 else np.nan,
            "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, yh),
            "recall": recall_score(y, yh, zero_division=0),
            "precision": precision_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "specificity": tn/max(tn+fp,1), "brier": brier_score_loss(y, p),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def best_threshold(y, p):
    """Pick the F1-optimal threshold on validation - never on test."""
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 200)))
    scores = [(f1_score(y, (p>=t).astype(int), zero_division=0), t) for t in qs]
    return max(scores)[1]

print(f"Config: horizon={HORIZON}w  outbreak>={OUTBREAK_INC}/100k  min_pop={MIN_POP}")
df = load_panel()
print("panel:", df.shape, "| municipios:", df.codigo_ibge.nunique(),
      "|", df.data_iniSE.min().date(), "->", df.data_iniSE.max().date())
sup = build_supervised(df)
feats = feature_columns(sup)
print("supervised:", sup.shape, "| features:", len(feats), "| outbreak rate: %.4f" % sup.y.mean())

tr, va, te = temporal_split(sup)
for n, d in [("train <=2021", tr), ("val 2022-23", va), ("test 2024-25", te)]:
    print(f"  {n:14} n={len(d):>8}  outbreak_rate={d.y.mean():.4f}")
if len(te)==0 or te.y.sum()==0:
    raise SystemExit("empty/degenerate test split")

Xtr, ytr = tr[feats], tr.y.values
Xva, yva = va[feats], va.y.values
Xte, yte = te[feats], te.y.values
rows = []

# ---- trivial baselines (must be beaten) ----
rows.append(report(yte, te.baseline_persistence.values, 0.5, "BASELINE_persistence"))
clim = seasonal_climatology(tr, te)
rows.append(report(yte, clim, best_threshold(yva, seasonal_climatology(tr, va)), "BASELINE_climatology"))

# ---- gradient boosting ----
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
spw = (ytr==0).sum()/max((ytr==1).sum(),1)
models = {
 "LightGBM": LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=127,
                            min_child_samples=40, subsample=0.85, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=spw,
                            random_state=42, n_jobs=-1, verbose=-1),
 "XGBoost": XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=8,
                          min_child_weight=10, subsample=0.85, colsample_bytree=0.8,
                          reg_lambda=5.0, scale_pos_weight=spw, tree_method="hist",
                          eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0),
}
fitted = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pv, pt = m.predict_proba(Xva)[:,1], m.predict_proba(Xte)[:,1]
    thr = best_threshold(yva, pv)
    rows.append(report(yte, pt, thr, name))
    fitted[name] = (m, thr)
    print(f"  fitted {name}")

res = pd.DataFrame(rows)
res.to_csv("reports/model2_results.csv", index=False)
cols = ["model","roc_auc","pr_auc","accuracy","recall","precision","f1","specificity","brier"]
print("\n=== TEST SET (2024-2025, never seen in training) ===")
print(res[cols].to_string(index=False))

best = res[res.model.isin(fitted)].sort_values("pr_auc", ascending=False).iloc[0]["model"]
m, thr = fitted[best]
joblib.dump({"model": m, "threshold": thr, "features": feats,
             "horizon": HORIZON, "outbreak_inc": OUTBREAK_INC}, "models/model2_outbreak.joblib")
imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
imp.to_csv("reports/model2_feature_importance.csv")
print(f"\nBest by PR-AUC: {best}  -> models/model2_outbreak.joblib")
print("\nTop 15 features:\n" + imp.head(15).to_string())
json.dump(res.to_dict("records"), open("reports/model2_summary.json","w"), indent=2, default=float)
