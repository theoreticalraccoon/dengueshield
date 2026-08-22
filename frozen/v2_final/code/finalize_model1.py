"""Final Model 1: tuned + calibrated screening model with honest nested evaluation."""
import sys, json, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from dengue.datasets import load_hematology_1523
from dengue.model1_screening import metrics_at, threshold_for_sensitivity, SEED

X, y, meta = load_hematology_1523()
print(f"{meta['name']}  n={len(y)}  feats={X.shape[1]}  prevalence={y.mean():.4f}")
trivial = max(y.mean(), 1-y.mean())
print(f"Trivial majority-class accuracy = {trivial:.4f}\n")

base = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("clf", LGBMClassifier(random_state=SEED, n_jobs=4, verbose=-1))])
grid = {"clf__n_estimators":[300,600,900], "clf__learning_rate":[0.01,0.02,0.05],
        "clf__num_leaves":[7,15,31,63], "clf__min_child_samples":[10,20,40,60],
        "clf__subsample":[0.7,0.85,1.0], "clf__subsample_freq":[1],
        "clf__colsample_bytree":[0.6,0.8,1.0], "clf__reg_lambda":[0,1,5,20]}

# NESTED CV: tuning happens inside each outer fold, so the reported score is unbiased.
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
for k,(tr,te) in enumerate(outer.split(X,y)):
    srch = RandomizedSearchCV(clone(base), grid, n_iter=40, scoring="roc_auc",
                              cv=StratifiedKFold(4,shuffle=True,random_state=SEED),
                              random_state=SEED, n_jobs=4, refit=True)
    srch.fit(X.iloc[tr], y[tr])
    cal = CalibratedClassifierCV(srch.best_estimator_, method="isotonic", cv=4)
    cal.fit(X.iloc[tr], y[tr])
    oof[te] = cal.predict_proba(X.iloc[te])[:,1]
    print(f"  outer fold {k}: inner-best AUC={srch.best_score_:.4f}")

m05 = metrics_at(y, oof, 0.5)
thr90 = threshold_for_sensitivity(y, oof, 0.90)
m90 = metrics_at(y, oof, thr90)
print("\n=== NESTED-CV, CALIBRATED (honest out-of-fold estimates) ===")
print(f"  ROC-AUC            {m05['roc_auc']:.4f}")
print(f"  PR-AUC             {m05['pr_auc']:.4f}")
print(f"  Brier              {m05['brier']:.4f}   (calibration quality)")
print(f"  -- at threshold 0.50 --")
print(f"  accuracy           {m05['accuracy']:.4f}  (trivial baseline {trivial:.4f})")
print(f"  sensitivity        {m05['sensitivity']:.4f}")
print(f"  specificity        {m05['specificity']:.4f}")
print(f"  -- at screening operating point (sensitivity>=0.90), thr={thr90:.3f} --")
print(f"  accuracy           {m90['accuracy']:.4f}")
print(f"  sensitivity        {m90['sensitivity']:.4f}")
print(f"  specificity        {m90['specificity']:.4f}")
print(f"  NPV                {m90['npv']:.4f}")

# calibration curve
from sklearn.calibration import calibration_curve
pt, pp = calibration_curve(y, oof, n_bins=10, strategy="quantile")
print("\n  reliability (predicted -> observed):")
for a,b in zip(pp,pt): print(f"    {a:.3f} -> {b:.3f}")

# fit final model on all data
srch = RandomizedSearchCV(clone(base), grid, n_iter=60, scoring="roc_auc",
                          cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                          random_state=SEED, n_jobs=4, refit=True).fit(X,y)
final = CalibratedClassifierCV(srch.best_estimator_, method="isotonic", cv=5).fit(X,y)
joblib.dump({"model":final,"features":list(X.columns),"threshold_sens90":float(thr90),
             "metrics_nested_cv":{"roc_auc":m05["roc_auc"],"pr_auc":m05["pr_auc"],
             "brier":m05["brier"],"accuracy@0.5":m05["accuracy"],
             "accuracy@sens90":m90["accuracy"],"specificity@sens90":m90["specificity"]}},
            "models/model1_screening.joblib")
json.dump({"trivial_baseline":float(trivial),"at_0.5":m05,"at_sens90":m90,
           "best_params":{k:str(v) for k,v in srch.best_params_.items()}},
          open("reports/model1_final.json","w"), indent=2, default=float)
np.save("reports/model1_oof_calibrated.npy", oof)
print("\nsaved models/model1_screening.joblib")
