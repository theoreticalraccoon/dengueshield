"""Report artifacts: what they are, who produces them, and which can be rebuilt.

Four files under `reports/` were read by the app but written by nothing in the repo
- no producer had ever been committed. One of them, `cbc_population_medians.json`,
is load-bearing: the screening form fills every unentered field from it, so Quick
entry mode depended on a file no code could regenerate.

They split into two kinds, and the distinction matters:

**derived** - a pure function of inputs that are themselves committed. These can be
rebuilt on demand and verified byte-for-byte against what is on disk. Both producers
here are exact.

**snapshot** - a measurement taken over a data state that has since moved on. The
Sri Lanka panel grows every Tuesday, so `srilanka_calibration.json` (n=3091) cannot
be reproduced from today's panel, which has no slice of that size. Re-deriving it
would not restore the file, it would silently replace a historical measurement with
a different one. These are recorded, not regenerated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = "derived"
SNAPSHOT = "snapshot"
PRODUCED = "produced"  # written by a script every time that script runs


@dataclass(frozen=True)
class Artifact:
    """One file under reports/, who writes it, and whether it can be rebuilt."""

    name: str
    producer: str  # the script that writes it, or "" when none ever existed
    kind: str
    note: str = ""

    @property
    def has_producer(self) -> bool:
        return bool(self.producer)


def _a(name, producer, kind=PRODUCED, note=""):
    return Artifact(name=name, producer=producer, kind=kind, note=note)


# Every report artifact the app or freeze_v2 reads. The three hand-maintained
# lists in freeze_v2.py drifted from reality; this is the one declaration.
INVENTORY: tuple[Artifact, ...] = (
    # --- screening
    _a("model1_summary.json", "train_model1.py"),
    _a("model1_final.json", "finalize_model1.py"),
    _a("model1_oof_calibrated.npy", "finalize_model1.py"),
    _a(
        "cbc_population_medians.json",
        "derive_reports.py",
        DERIVED,
        "Cohort feature medians. Had no producer at all; the screening form fills "
        "every unentered field from it, so Quick entry mode depended on a file "
        "nothing could regenerate.",
    ),
    _a(
        "model1_operating_points.json",
        "derive_reports.py",
        DERIVED,
        "Pure function of model1_oof_calibrated.npy; no retrain needed.",
    ),
    # --- complication
    _a("peds_final.json", "finalize_peds.py"),
    _a("ablation_peds.csv", "run_ablation.py"),
    # --- Brazil outbreak
    _a("model2_summary.json", "train_model2.py"),
    _a("model2_results.csv", "train_model2.py"),
    _a("model2_feature_importance.csv", "train_model2.py"),
    _a("model2_lstm.json", "train_lstm.py"),
    _a("model2_robustness.json", "robustness_model2.py"),
    _a("model2_shap.json", "shap_model2.py"),
    _a("model2_spatial_ablation.json", "spatial_and_ablation.py"),
    _a("model2_spatial_holdout.csv", "spatial_and_ablation.py"),
    _a("model2_information_ablation.csv", "spatial_and_ablation.py"),
    _a("model2_calibration_errors.json", "calibration_and_errors.py"),
    _a("leakage_audit.json", "leakage_audit.py"),
    _a("lag_fix_impact.json", "verify_fixes.py"),
    _a(
        "model2_headtohead.csv",
        "",
        SNAPSHOT,
        "GBM vs LSTM vs baselines. No producer was ever committed, and the run it "
        "records predates the horizon-2 retrain. Recomputing would replace a "
        "historical comparison rather than restore it.",
    ),
    _a("model2_headtohead.json", "", SNAPSHOT, "Same records as the CSV; nothing reads it."),
    # --- Sri Lanka
    _a("srilanka_current_risk.csv", "finalize_srilanka.py"),
    _a("srilanka_feature_importance.csv", "finalize_srilanka.py"),
    _a("srilanka_dual_risk.csv", "finalize_emergence.py"),
    _a("srilanka_emergence.json", "finalize_emergence.py"),
    _a("srilanka_transfer.csv", "transfer_srilanka.py"),
    _a(
        "srilanka_calibration.json",
        "",
        SNAPSHOT,
        "Measured over a panel of n=3091. The panel grows every Tuesday and no "
        "slice of it is that size any more, so this cannot be reproduced.",
    ),
    # --- screening + complication detail
    _a("model1_cv_results.csv", "train_model1.py"),
    _a("model1_oof.npz", "train_model1.py"),
    _a("peds_oof.npy", "finalize_peds.py"),
    _a("ablation_hematology.csv", "run_ablation.py"),
    _a("ablation_summary.json", "run_ablation.py"),
    # --- Brazil outbreak detail
    _a("model2_lstm_test_preds.npz", "train_lstm.py"),
    _a("model2_shap_importance.csv", "shap_model2.py"),
    _a("model2_shap_drivers.csv", "shap_model2.py"),
    _a("model2_shap_summary.png", "shap_model2.py"),
    _a("model2_shap_drivers.png", "shap_model2.py"),
    _a("model2_reliability_raw.csv", "calibration_and_errors.py"),
    _a("model2_reliability_calibrated.csv", "calibration_and_errors.py"),
    _a("model2_reliability.png", "calibration_and_errors.py"),
    _a("model2_errors_by_state.csv", "calibration_and_errors.py"),
    _a("model2_worst_false_negatives.csv", "calibration_and_errors.py"),
    _a("reporting_delay_stress.csv", "leakage_audit.py"),
    _a(
        "model2_state_folds_with_events.csv",
        "",
        SNAPSHOT,
        "No producer and no consumer: nothing in the repo writes it and nothing "
        "reads it. Kept because it is committed evidence, but it is dead weight.",
    ),
    # --- transfer
    _a("srilanka_transfer.json", "transfer_srilanka.py"),
    _a("transfer_auc.csv", "transfer_test.py"),
    _a("srilanka_situation_history.csv", "refresh_situation.py"),
    # --- data integrity + provenance
    _a("dataset_audit.csv", "run_audit.py"),
    _a("data_freshness.json", "refresh_data.py"),
    _a("srilanka_latest_situation.csv", "refresh_situation.py"),
    _a("situation_freshness.json", "refresh_situation.py"),
)

BY_NAME = {a.name: a for a in INVENTORY}


def orphans() -> list[Artifact]:
    """Artifacts no script in the repo writes."""
    return [a for a in INVENTORY if not a.has_producer]


def reproducible() -> list[Artifact]:
    """Artifacts `derive_reports.py` can rebuild and verify exactly."""
    return [a for a in INVENTORY if a.kind == DERIVED]


def freezable(root: Path) -> list[Path]:
    """Report files a release should capture: declared, and present on disk.

    freeze_v2.py used `reports/*.glob("*")`, so whatever happened to be sitting in
    the directory entered the immutable release - including the weekly refresh's
    freshness stamps and the experiments' output. A release should contain what it
    declares.
    """
    return [p for a in INVENTORY if (p := root / a.name).exists()]


# ------------------------------------------------------------------ producers


def population_medians(X: pd.DataFrame) -> dict[str, float]:
    """Cohort medians used to fill unentered fields on the screening form.

    Taken over the engineered feature matrix, not the raw columns, so the ratio
    features are medians-of-ratios rather than ratios-of-medians. That is what the
    model was trained against and what the committed file contains.
    """
    return {str(k): float(v) for k, v in X.median().items()}


def operating_points(y: np.ndarray, p: np.ndarray, balanced_threshold: float) -> dict:
    """Screening metrics at the default and best-balanced thresholds.

    `p` is the calibrated out-of-fold prediction saved by finalize_model1.py, so
    this needs no retraining - it is a pure function of a committed array.
    """
    from dengue.model1_screening import metrics_at

    at_half = metrics_at(y, p, 0.5)
    return {
        "roc_auc": at_half["roc_auc"],
        "pr_auc": at_half["pr_auc"],
        "brier": at_half["brier"],
        "at_0.5": at_half,
        "best_balanced": metrics_at(y, p, balanced_threshold),
    }


def best_balanced_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The threshold maximising balanced accuracy, searched over observed scores.

    Candidates are the observed scores themselves, not a rounded grid: the
    committed threshold is an exact prediction value and rounding moves it.

    Scored by counting rather than by calling `metrics_at` per candidate - that
    would recompute ROC-AUC, PR-AUC and Brier at all ~1500 thresholds, none of
    which depend on the threshold. Ties resolve to the lowest score, matching a
    left-to-right scan of the ascending candidates.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    if pos.size == 0 or neg.size == 0:
        return 0.5

    candidates = np.unique(p)
    # A case is predicted positive when its score is >= the threshold.
    true_pos = pos.size - np.searchsorted(pos, candidates, side="left")
    true_neg = np.searchsorted(neg, candidates, side="left")
    balanced = (true_pos / pos.size + true_neg / neg.size) / 2
    return float(candidates[int(balanced.argmax())])


# ------------------------------------------------------------------ compare


def _round(obj, places=10):
    """Normalise floats so a rebuild is compared at the precision JSON carries."""
    if isinstance(obj, dict):
        return {k: _round(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, places) for v in obj]
    if isinstance(obj, (int, float, np.floating, np.integer)):
        return round(float(obj), places)
    return obj


def differences(rebuilt: dict, path: Path, places: int = 10) -> list[str]:
    """Keys where a rebuilt artifact disagrees with the file on disk.

    An empty list means the producer reproduces the committed file exactly.
    """
    if not path.exists():
        return [f"{path.name}: absent"]
    on_disk = json.loads(path.read_text())
    a, b = _round(rebuilt, places), _round(on_disk, places)

    out: list[str] = []

    def walk(x, y, trail=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x:
                    out.append(f"{trail}{k}: only on disk")
                elif k not in y:
                    out.append(f"{trail}{k}: only rebuilt")
                else:
                    walk(x[k], y[k], f"{trail}{k}.")
        elif x != y:
            out.append(f"{trail.rstrip('.')}: rebuilt {x!r} != on disk {y!r}")

    walk(a, b)
    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
