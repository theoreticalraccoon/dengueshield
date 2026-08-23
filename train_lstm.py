import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from dengue.lstm_outbreak import SEQ_LEN, make_sequences, predict, train_lstm
from dengue.model2_outbreak import (
    DYNAMIC,
    HORIZON,
    STATIC,
    build_supervised,
    load_panel,
    temporal_split,
)

print("loading panel ...", flush=True)
sup = build_supervised(load_panel())
dyn = [
    c
    for c in [*DYNAMIC, "week_sin", "week_cos", "casos_growth_1", "casos_growth_4", "inc_roll4"]
    if c in sup.columns
]
stat = [c for c in STATIC if c in sup.columns]
print(f"dynamic={len(dyn)}  static={len(stat)}  seq_len={SEQ_LEN}  horizon={HORIZON}")

tr, va, te = temporal_split(sup)
# impute + scale, fitted on TRAIN ONLY
imp_d, sc_d = SimpleImputer(strategy="median"), StandardScaler()
imp_s, sc_s = SimpleImputer(strategy="median"), StandardScaler()
sc_d.fit(imp_d.fit_transform(tr[dyn]))
sc_s.fit(imp_s.fit_transform(tr[stat]))


def prep(d):
    d = d.copy()
    d[dyn] = sc_d.transform(imp_d.transform(d[dyn]))
    d[stat] = sc_s.transform(imp_s.transform(d[stat]))
    return d


print(
    "building sequences on the FULL contiguous panel, then splitting by label year ...", flush=True
)
# Sequences are built across the whole series so that a 2024 label may legitimately
# use 2023 weeks as input history - exactly what is available operationally. The
# split is then made on the YEAR OF THE LABEL, keeping train/val/test disjoint and
# identical in coverage to the gradient-boosting comparison.
X_all, S_all, y_all, idx_all = make_sequences(prep(sup), dyn, stat, SEQ_LEN)
anio = sup.loc[idx_all, "anio"].to_numpy()
masks = {"train": anio <= 2021, "val": (anio > 2021) & (anio <= 2023), "test": anio > 2023}
out = {}
for name, mk in masks.items():
    out[name] = (X_all[mk], S_all[mk], y_all[mk], idx_all[mk])
    print(f"  {name:5} X={out[name][0].shape} pos_rate={out[name][2].mean():.4f}", flush=True)

Xtr, Str, ytr, _ = out["train"]
Xva, Sva, yva, _ = out["val"]
Xte, Ste, yte, ite = out["test"]
print("training LSTM ...", flush=True)
model, best_ap = train_lstm(Xtr, Str, ytr, Xva, Sva, yva, epochs=40, bs=512, patience=6)

pv = predict(model, Xva, Sva)
pt = predict(model, Xte, Ste)
qs = np.unique(np.quantile(pv, np.linspace(0.5, 0.999, 200)))
thr = max((f1_score(yva, (pv >= t).astype(int), zero_division=0), t) for t in qs)[1]
yh = (pt >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, yh, labels=[0, 1]).ravel()
res = {
    "model": "PyTorch_LSTM",
    "threshold": float(thr),
    "roc_auc": roc_auc_score(yte, pt),
    "pr_auc": average_precision_score(yte, pt),
    "accuracy": accuracy_score(yte, yh),
    "recall": recall_score(yte, yh, zero_division=0),
    "precision": precision_score(yte, yh, zero_division=0),
    "f1": f1_score(yte, yh, zero_division=0),
    "specificity": tn / max(tn + fp, 1),
    "brier": brier_score_loss(yte, pt),
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn),
}
print("\n=== LSTM TEST (2024-25) ===")
for k, v in res.items():
    print(f"  {k:12} {v}")
torch.save(
    {
        "state_dict": model.state_dict(),
        "dyn": dyn,
        "stat": stat,
        "seq_len": SEQ_LEN,
        "threshold": float(thr),
        "horizon": HORIZON,
    },
    "models/model2_lstm.pt",
)
joblib.dump(
    {"imp_d": imp_d, "sc_d": sc_d, "imp_s": imp_s, "sc_s": sc_s},
    "models/model2_lstm_scalers.joblib",
)
with open("reports/model2_lstm.json", "w") as _f:
    json.dump(res, _f, indent=2, default=float)
np.savez("reports/model2_lstm_test_preds.npz", y=yte, p=pt, idx=ite)
print("\nsaved models/model2_lstm.pt")
