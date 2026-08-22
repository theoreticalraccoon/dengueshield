import sys, json; sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from xgboost import XGBClassifier
from dengue.audit import audit, permutation_null
from dengue.datasets import ALL_SCREENING

m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, verbosity=0)
rows = []
for name, loader in ALL_SCREENING.items():
    X, y, meta = loader()
    rep = audit(name, X, y)
    pn = permutation_null(m, X, y, n=8, folds=5)
    rep.update(pn)
    rows.append(rep)
    print(f"\n{'='*78}\n{name}  (n={rep['n']}, prevalence={rep['prevalence']:.3f})")
    print(f"  verdict              : {rep['verdict']}")
    print(f"  max single-feat AUC  : {rep['max_single_feature_auc']:.4f}")
    print(f"  leaky features       : {rep['leaky_features'] or 'none'}")
    print(f"  disjoint-range feats : {rep['disjoint_features'] or 'none'}")
    print(f"  CV AUC               : {pn['real_auc']:.4f}   null={pn['null_mean']:.3f}+/-{pn['null_sd']:.3f}  z={pn['z']:.1f}")
    print("  top features by solo AUC:")
    print(rep["single_feature_auc"].head(5).to_string().replace("\n", "\n     "))

summary = pd.DataFrame([{k: v for k, v in r.items() if k != "single_feature_auc"} for r in rows])
summary.to_csv("reports/dataset_audit.csv", index=False)
print(f"\n\n{'='*78}\nSUMMARY\n{summary[['dataset','n','prevalence','max_single_feature_auc','real_auc','verdict']].to_string(index=False)}")
print("\nUSABLE:", [r['dataset'] for r in rows if r['verdict']=='PASS'])
