"""Persist the complication-risk model (the 'needs closer monitoring' screener)."""
import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from dengue.ablation import (load_peds, PEDS_GROUPS, nested_oof, full_metrics,
                             threshold_for_sensitivity, SEED)

X, y, meta = load_peds()
cols = [c for g in PEDS_GROUPS.values() for c in g if c in X.columns]
X = X[cols]
print(f"n={len(y)} complicated={y.sum()} features={len(cols)}")

oof = nested_oof(X, y, kind="lgbm")
thr = threshold_for_sensitivity(y, oof, 0.90)
m = full_metrics(y, oof, thr)
print(f"nested-CV: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
      f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} NPV={m['npv']:.3f}")

final = Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                         min_child_samples=10, subsample=0.85, subsample_freq=1,
                                         colsample_bytree=0.8, reg_lambda=5.0,
                                         class_weight="balanced", random_state=SEED,
                                         n_jobs=4, verbose=-1))])
final = CalibratedClassifierCV(final, method="sigmoid", cv=5).fit(X, y)
joblib.dump({"model": final, "features": cols, "threshold_sens90": float(thr),
             "metrics": m, "groups": PEDS_GROUPS}, "models/peds_complications.joblib")
with open("reports/peds_final.json", "w") as _f:
    json.dump(m, _f, indent=2, default=float)
np.save("reports/peds_oof.npy", oof)
print("saved models/peds_complications.joblib")
