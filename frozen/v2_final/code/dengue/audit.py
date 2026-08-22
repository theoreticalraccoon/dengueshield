"""Data integrity auditing: detect synthetic / label-leaking datasets.

Public dengue "ML datasets" are frequently simulated or rule-generated. Training on
them yields 95-100% scores that do not transfer to patients. Every dataset used in
this project must pass this audit before it is allowed near a model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SINGLE_FEATURE_AUC_ALARM = 0.95   # one feature ~ the label => leakage or simulation
DISJOINT_ALARM = True             # class-conditional ranges must overlap


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("category").cat.codes
    return out


def single_feature_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """AUC of each feature used alone (direction-agnostic)."""
    Xn, res = _numeric(X), {}
    for c in Xn.columns:
        v = Xn[c]
        m = v.notna().values & ~pd.isna(y)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], v[m])
        res[c] = max(a, 1 - a)
    return pd.Series(res).sort_values(ascending=False)


def range_overlap(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Do positive/negative value ranges overlap? Disjoint ranges = simulated data."""
    Xn, rows = _numeric(X), []
    for c in Xn.columns:
        a, b = Xn.loc[y == 1, c].dropna(), Xn.loc[y == 0, c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        overlap = not (a.max() < b.min() or b.max() < a.min())
        rows.append({"feature": c, "pos_min": a.min(), "pos_max": a.max(),
                     "neg_min": b.min(), "neg_max": b.max(), "overlaps": overlap})
    return pd.DataFrame(rows)


def permutation_null(model, X: pd.DataFrame, y: np.ndarray, n: int = 10, folds: int = 5):
    """Compare true CV AUC against label-shuffled nulls. z>3 => signal is real."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    Xn = _numeric(X)
    p = cross_val_predict(model, Xn, y, method="predict_proba",
                          cv=StratifiedKFold(folds, shuffle=True, random_state=42))[:, 1]
    real = roc_auc_score(y, p)
    rng, nulls = np.random.default_rng(0), []
    for i in range(n):
        ys = rng.permutation(y)
        ps = cross_val_predict(model, Xn, ys, method="predict_proba",
                               cv=StratifiedKFold(folds, shuffle=True, random_state=i))[:, 1]
        nulls.append(roc_auc_score(ys, ps))
    nulls = np.array(nulls)
    z = (real - nulls.mean()) / max(nulls.std(), 1e-9)
    return {"real_auc": real, "null_mean": nulls.mean(), "null_sd": nulls.std(), "z": z}


def audit(name: str, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Full integrity report. verdict REJECT => do not train on this dataset."""
    aucs = single_feature_auc(X, y)
    ov = range_overlap(X, y)
    disjoint = ov.loc[~ov["overlaps"], "feature"].tolist() if len(ov) else []
    leaky = aucs[aucs >= SINGLE_FEATURE_AUC_ALARM].index.tolist()
    verdict = "REJECT" if (leaky or disjoint) else "PASS"
    return {"dataset": name, "n": int(len(y)), "prevalence": float(np.mean(y)),
            "max_single_feature_auc": float(aucs.iloc[0]) if len(aucs) else float("nan"),
            "leaky_features": leaky, "disjoint_features": disjoint,
            "verdict": verdict, "single_feature_auc": aucs}
