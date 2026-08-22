"""Do models trained on the synthetic datasets transfer to real clinical data?

Uses only the 4 features common to all three cohorts: age, sex, platelet, WBC.
"""
import sys, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from lightgbm import LGBMClassifier
from dengue.datasets import (load_hematology_1523, load_vitals_1003,
                             load_bd_structured, load_bd_comprehensive)

COMMON = ["age","sex","platelet","wbc"]
def harmonise(X, y, m):
    ren = {
      "hematology_1523": {"Age":"age","Gender":"sex","Total Platelet Count(/cumm)":"platelet","Total WBC count(/cumm)":"wbc"},
      "vitals_1003":     {"Age":"age","Sex":"sex","Platelet Count":"platelet","WBC Count":"wbc"},
      "bd_structured":   {"Age":"age","Gender":"sex","Platelet Count":"platelet","WBC":"wbc"},
      "bd_comprehensive":{"Age":"age","Gender":"sex","Platelet_Count":"platelet","WBC_Count":"wbc"},
    }[m["name"]]
    d = X.rename(columns=ren)[COMMON].copy()
    d["sex"] = pd.to_numeric(d["sex"], errors="coerce").fillna(-1)
    return d.apply(pd.to_numeric, errors="coerce"), y

sets = {}
for loader in (load_hematology_1523, load_vitals_1003, load_bd_structured, load_bd_comprehensive):
    X, y, m = loader(); sets[m["name"]] = harmonise(X, y, m)

mk = lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            min_child_samples=20, random_state=42, n_jobs=4, verbose=-1)
print("Common features:", COMMON)
print("\n--- internal CV (same dataset) ---")
for name,(X,y) in sets.items():
    p = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5,shuffle=True,random_state=0), method="predict_proba")[:,1]
    print(f"  {name:18} AUC={roc_auc_score(y,p):.4f}  acc={accuracy_score(y,p>0.5):.4f}")

print("\n--- TRANSFER: train on row, test on column (AUC) ---")
names = list(sets)
tab = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    Xa, ya = sets[a]; m = mk().fit(Xa, ya)
    for b in names:
        Xb, yb = sets[b]
        if a == b:
            p = cross_val_predict(mk(), Xb, yb, cv=StratifiedKFold(5,shuffle=True,random_state=0), method="predict_proba")[:,1]
        else:
            p = m.predict_proba(Xb)[:,1]
        tab.loc[a,b] = roc_auc_score(yb, p)
print(tab.round(3).to_string())
tab.to_csv("reports/transfer_auc.csv")

print("\n--- headline: synthetic-trained -> REAL clinical data ---")
Xr, yr = sets["hematology_1523"]
for src in ["bd_structured","bd_comprehensive","vitals_1003"]:
    Xs, ys = sets[src]; m = mk().fit(Xs, ys)
    p = m.predict_proba(Xr)[:,1]; yh = (p>0.5).astype(int)
    print(f"  trained on {src:18} -> real data: AUC={roc_auc_score(yr,p):.4f} "
          f"acc={accuracy_score(yr,yh):.4f} sens={recall_score(yr,yh):.4f} "
          f"(real-data majority baseline acc={max(yr.mean(),1-yr.mean()):.4f})")
