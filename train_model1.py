import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from dengue.datasets import load_hematology_1523
from dengue.model1_screening import build_models, metrics_at, threshold_for_sensitivity, SEED

N_REPEATS, N_SPLITS = 4, 5
X, y, meta = load_hematology_1523()
print(f"Dataset : {meta['name']}   n={len(y)}  features={X.shape[1]}  prevalence={y.mean():.3f}")
print(f"Task    : {meta['task']}")
print(f"Trivial majority-class accuracy = {max(y.mean(),1-y.mean()):.4f}")
print(f"Protocol: {N_SPLITS}-fold CV x {N_REPEATS} repeats, all preprocessing inside fold\n")

def repeated_oof(model, X, y):
    """Return OOF probability matrix (n_samples, n_repeats)."""
    P = np.zeros((len(y), N_REPEATS))
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for tr, te in skf.split(X, y):
            m = clone(model).fit(X.iloc[tr], y[tr])
            P[te, r] = m.predict_proba(X.iloc[te])[:, 1]
    return P

rows, oof_store = [], {}
for name, model in build_models().items():
    P = repeated_oof(model, X, y)
    oof_store[name] = P
    per = [metrics_at(y, P[:, r], 0.5) for r in range(N_REPEATS)]
    agg = {k: (np.mean([d[k] for d in per]), np.std([d[k] for d in per]))
           for k in ["roc_auc","pr_auc","accuracy","sensitivity","specificity","balanced_accuracy","brier"]}
    pm = P.mean(1)
    thr = threshold_for_sensitivity(y, pm, 0.90)
    m90 = metrics_at(y, pm, thr)
    rows.append({"model": name, **{k: agg[k][0] for k in agg},
                 **{k+"_sd": agg[k][1] for k in agg},
                 "acc@sens90": m90["accuracy"], "spec@sens90": m90["specificity"], "thr@sens90": thr})
    print(f"{name:20} AUC={agg['roc_auc'][0]:.4f}+/-{agg['roc_auc'][1]:.4f}  "
          f"acc={agg['accuracy'][0]:.4f}+/-{agg['accuracy'][1]:.4f}  "
          f"sens={agg['sensitivity'][0]:.4f}  spec={agg['specificity'][0]:.4f}  "
          f"brier={agg['brier'][0]:.4f}")

res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
res.to_csv("reports/model1_cv_results.csv", index=False)
np.savez("reports/model1_oof.npz", y=y, **oof_store)
print("\n" + res[["model","roc_auc","pr_auc","accuracy","sensitivity","specificity","balanced_accuracy","brier"]].to_string(index=False))
best = res.iloc[0]["model"]
print(f"\nBest by ROC-AUC: {best}   (trivial baseline acc = {max(y.mean(),1-y.mean()):.4f})")
with open("reports/model1_summary.json", "w") as _f:
    json.dump({"best_model": best,
               "trivial_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
               "results": res.to_dict("records")}, _f, indent=2)
