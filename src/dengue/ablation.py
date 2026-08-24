"""Feature-group ablation: how much does each category of information add?

Answers "what actually improves screening?" by adding one information category at
a time and re-running the full nested-CV protocol each time.

Post-hoc variables are excluded by construction. Treatments given *because* a
patient deteriorated (IV fluids, steroids, antibiotics) are consequences of the
outcome, not predictors available at triage, and would leak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dengue.config import SEED as SEED  # re-exported: callers import it from here
from dengue.metrics import threshold_for_sensitivity  # re-exported: callers import it from here

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# ---------------------------------------------------------------- pediatric cohort
PEDS_TARGET = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"
PEDS_POSTHOC = ["IV FLUIDS", "ANTIBIOTICS", "ANTIPYRETICS", "STEROIDS", "OUTCOME"]
PEDS_GROUPS = {
    "A_CBC": ["HB", "PCV", "LOWEST WBC", "HIGHEST WBC", "PLATELET COUNT"],
    "B_+demographics": ["AGE", "GENDER"],
    "C_+symptoms": [
        "HEADACHE",
        "MYALGIA",
        "ABDOMINAL PAIN",
        "RASH",
        "VOMITING",
        "BREATHLESSNESS",
        "BLEEDING",
        "ORGANOMEGALY",
        "BP AT ADMISSION",
    ],
    "D_+fever_day": ["DURATION OF FEVER"],
    "E_+organ_labs": ["ALT", "CREATININE", "FERRITIN"],
}


def load_peds():
    """Pediatric dengue cohort (Zenodo 6476112). Target: complicated dengue."""
    d = pd.read_excel(RAW / "zen_peds_MASTER_CHART_F1000Research.xlsx")
    d.columns = [c.strip() for c in d.columns]
    y = (d[PEDS_TARGET] == 2).astype(int).values  # 2 = complicated dengue
    X = d.drop(columns=["Serial number", PEDS_TARGET, *PEDS_POSTHOC], errors="ignore")
    # binary symptom columns are coded 1/2 -> map to 0/1 (1 = present)
    for c in PEDS_GROUPS["C_+symptoms"] + ["GENDER"]:
        if c in X and set(X[c].dropna().unique()) <= {1, 2}:
            X[c] = (X[c] == 1).astype(int)
    return (
        X,
        y,
        {
            "name": "peds_complications",
            "source": "Zenodo 6476112",
            "task": "complicated dengue (needs closer monitoring)",
        },
    )


# ------------------------------------------------------------- screening cohort
HEMA_GROUPS = {
    "A_CBC": [
        "Hemoglobin(g/dl)",
        "Neutrophils(%)",
        "Lymphocytes(%)",
        "Monocytes(%)",
        "Eosinophils(%)",
        "RBC",
        "HCT(%)",
        "MCV(fl)",
        "MCH(pg)",
        "MCHC(g/dl)",
        "RDW-CV(%)",
        "Total Platelet Count(/cumm)",
        "MPV(fl)",
        "PDW(%)",
        "PCT(%)",
        "Total WBC count(/cumm)",
    ],
    "B_+demographics": ["Age", "Gender"],
    "C_+derived_ratios": ["NLR", "PLR", "PLT_WBC_ratio", "MPV_PLT_ratio"],
    # Percentages turned into absolute counts, plus the indices built on them. Kept
    # as its own tier rather than folded into C so the ablation table shows what the
    # counts bought over the ratios alone.
    "D_+absolute_counts": [
        "neut_abs",
        "lymph_abs",
        "mono_abs",
        "LMR",
        "SII",
        "plateletcrit_calc",
        "RDW_PLT_ratio",
        "HCT_PLT_ratio",
    ],
}


# ------------------------------------------------------------------- evaluation
def _model(kind: str):
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                LGBMClassifier(random_state=SEED, n_jobs=4, verbose=-1, class_weight="balanced"),
            ),
        ]
    )


LGBM_GRID = {
    "clf__n_estimators": [100, 200, 400],
    "clf__learning_rate": [0.01, 0.03, 0.05],
    "clf__num_leaves": [3, 7, 15],
    "clf__min_child_samples": [5, 10, 20, 30],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__subsample_freq": [1],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__reg_lambda": [1, 5, 20],
}


def _tuned(X: pd.DataFrame, y: np.ndarray, kind: str, n_iter: int, folds: int, calibrate: bool):
    """Search, then calibrate - the protocol both the folds and the final fit use.

    This exists because `finalize_peds.py` did not use it. Its metrics came from
    `nested_oof`, which tunes inside every fold, while the model it actually wrote
    to models/peds_complications.joblib was a hand-written LightGBM with fixed
    hyperparameters. The reported ROC-AUC 0.874 therefore described an estimator
    that was never saved. Both paths now call this, so they cannot diverge again.
    """
    est = _model(kind)
    if kind == "lgbm":
        search = RandomizedSearchCV(
            est,
            LGBM_GRID,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=StratifiedKFold(4, shuffle=True, random_state=SEED),
            random_state=SEED,
            n_jobs=4,
            refit=True,
        )
        search.fit(X, y)
        est = search.best_estimator_
    else:
        est = clone(est).fit(X, y)

    # Too few positives to split a calibration set out of: an uncalibrated
    # estimator is honest, a calibrator fitted on 4 events is not.
    if calibrate and min(np.bincount(y)) >= 8:
        est = CalibratedClassifierCV(est, method="sigmoid", cv=folds).fit(X, y)
    return est


def nested_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    kind: str = "lgbm",
    n_splits: int = 5,
    n_repeats: int = 3,
    calibrate: bool = True,
) -> np.ndarray:
    """Out-of-fold probabilities averaged over repeats; tuning inside each outer fold."""
    P = np.zeros((len(y), n_repeats))
    for r in range(n_repeats):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=SEED + r)
        for tr, te in outer.split(X, y):
            est = _tuned(X.iloc[tr], y[tr], kind, n_iter=25, folds=4, calibrate=calibrate)
            P[te, r] = est.predict_proba(X.iloc[te])[:, 1]
    return P.mean(axis=1)


def fit_final(X: pd.DataFrame, y: np.ndarray, kind: str = "lgbm", n_iter: int = 60):
    """The deployable estimator, fitted on everything by the in-fold protocol.

    `n_iter` is larger than the folds use because this is fitted once and there is
    no outer loop paying for it.
    """
    return _tuned(X, y, kind, n_iter=n_iter, folds=5, calibrate=True)




def full_metrics(y, p, thr) -> dict:
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": float(thr),
        "accuracy": accuracy_score(y, yh),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": (sens + spec) / 2,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    kind: str = "lgbm",
    sens_target: float = 0.90,
) -> pd.DataFrame:
    """Cumulatively add feature groups, re-running nested CV at each tier."""
    rows, cols = [], []
    for label, add in groups.items():
        cols = cols + [c for c in add if c in X.columns]
        if not cols:
            continue
        p = nested_oof(X[cols], y, kind=kind)
        thr = threshold_for_sensitivity(y, p, sens_target)
        m = full_metrics(y, p, thr)
        m.update({"tier": label, "n_features": len(cols), "model": kind})
        rows.append(m)
        print(
            f"  {label:20} k={len(cols):>2}  AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
            f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)
