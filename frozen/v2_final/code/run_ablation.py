import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from dengue.ablation import (load_peds, run_ablation, nested_oof, full_metrics,
                             threshold_for_sensitivity, PEDS_GROUPS, HEMA_GROUPS)
from dengue.datasets import load_hematology_1523
from dengue.audit import audit, permutation_null
from lightgbm import LGBMClassifier

out = {}

print("="*80)
print("ABLATION 1 - COMPLICATED DENGUE (needs closer monitoring), Zenodo 6476112")
Xp, yp, mp = load_peds()
print(f"n={len(yp)}  complicated={yp.sum()} ({yp.mean():.3f})  trivial_acc={max(yp.mean(),1-yp.mean()):.4f}")
rep = audit("peds", Xp, yp)
print(f"audit: {rep['verdict']}  max_single_feature_auc={rep['max_single_feature_auc']:.4f}")
pn = permutation_null(LGBMClassifier(n_estimators=200,num_leaves=7,random_state=42,n_jobs=4,verbose=-1),
                      Xp, yp, n=8, folds=5)
print(f"permutation control: real={pn['real_auc']:.4f} null={pn['null_mean']:.4f}+/-{pn['null_sd']:.4f} z={pn['z']:.1f}")
print("\nLightGBM:")
ap_l = run_ablation(Xp, yp, PEDS_GROUPS, kind="lgbm")
print("Logistic regression:")
ap_r = run_ablation(Xp, yp, PEDS_GROUPS, kind="logreg")
abl_p = pd.concat([ap_l, ap_r], ignore_index=True)
abl_p.to_csv("reports/ablation_peds.csv", index=False)
out["peds"] = {"audit": {k:v for k,v in rep.items() if k!="single_feature_auc"},
               "permutation": pn, "ablation": abl_p.to_dict("records")}

print("\n" + "="*80)
print("ABLATION 2 - DENGUE SCREENING (CBC only cohort), Mendeley 6fsrsk3mb8")
Xh, yh, mh = load_hematology_1523()
print(f"n={len(yh)}  positive={yh.sum()} ({yh.mean():.3f})  trivial_acc={max(yh.mean(),1-yh.mean()):.4f}")
print("NOTE: this cohort records NO symptoms and NO fever duration, so tiers C/D of the")
print("      requested ablation are not estimable here - that is a data limitation, not a result.")
print("\nLightGBM:")
ah_l = run_ablation(Xh, yh, HEMA_GROUPS, kind="lgbm")
abl_h = ah_l
abl_h.to_csv("reports/ablation_hematology.csv", index=False)
out["hematology"] = {"ablation": abl_h.to_dict("records")}

json.dump(out, open("reports/ablation_summary.json","w"), indent=2, default=float)

print("\n" + "="*80)
print("PEDIATRIC ABLATION (LightGBM) - full screening metrics at sensitivity>=0.90")
cols=["tier","n_features","roc_auc","pr_auc","accuracy","sensitivity","specificity","ppv","npv","brier"]
print(ap_l[cols].round(4).to_string(index=False))
print("\nHEMATOLOGY ABLATION (LightGBM)")
print(ah_l[cols].round(4).to_string(index=False))
