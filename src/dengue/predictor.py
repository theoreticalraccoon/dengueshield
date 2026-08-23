"""One interface over four saved models that agree on nothing else.

The bundles are `joblib.dump`ed dicts, and each was written by a different script
with a different idea of the contract:

    model1_screening      threshold_sens90   metrics_nested_cv   predict_proba
    peds_complications    threshold_sens90   metrics             predict_proba
    srilanka_outbreak     threshold          -- absent --        predict
    srilanka_emergence    threshold          metrics             predict

So every caller had to remember which key this particular bundle uses, whether its
estimator is a scikit-learn probability model or a bare LightGBM Booster, and how
to order the feature columns. That knowledge was spread across app.py, the finalize
scripts and freeze_v2.py, which resorted to probing with `hasattr(est, "num_trees")`.

The seam is real rather than hypothetical: there are genuinely two adapters behind
it, `predict_proba(X)[:, 1]` and `predict(X)`, and they are not interchangeable.

Screening additionally needs the inference-time half of its feature engineering:
unentered fields fall back to cohort medians, and the engineered ratios are then
rebuilt from whatever was actually entered - using `datasets.clinical_ratios`, the
same function training used, rather than a second copy of the arithmetic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from dengue.config import MODELS, REPORTS
from dengue.datasets import clinical_ratios

# Bundle key aliases, most specific first. A bundle carries exactly one of each.
THRESHOLD_KEYS = ("threshold_sens90", "threshold")
METRICS_KEYS = ("metrics", "metrics_nested_cv")

SCREENING = "screening"
PEDS = "peds"
SL_CONTINUATION = "sl_continuation"
SL_EMERGENCE = "sl_emergence"

FILENAMES = {
    SCREENING: "model1_screening.joblib",
    PEDS: "peds_complications.joblib",
    SL_CONTINUATION: "srilanka_outbreak.joblib",
    SL_EMERGENCE: "srilanka_emergence.joblib",
}


def _screening_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the engineered CBC ratios exactly as load_hematology_1523 does."""
    return clinical_ratios(
        frame,
        "Total Platelet Count(/cumm)",
        "Total WBC count(/cumm)",
        "Neutrophils(%)",
        "Lymphocytes(%)",
        "MPV(fl)",
    )


@dataclass(frozen=True)
class Prediction:
    """A probability, the operating point it is judged against, and the verdict."""

    probability: float
    threshold: float
    flag: bool

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")


def _first(bundle: dict, keys, default=None):
    for k in keys:
        if k in bundle:
            return bundle[k]
    return default


def probabilities(model: Any, X: pd.DataFrame):
    """Positive-class probabilities, whichever kind of estimator this is.

    scikit-learn classifiers expose predict_proba; a native LightGBM Booster's
    `predict` already returns the positive-class probability for a binary
    objective, and it has no predict_proba at all.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


@dataclass(frozen=True)
class Predictor:
    """A saved model plus everything a caller needs to use it correctly."""

    name: str
    model: Any
    features: list[str]
    threshold: float
    metrics: dict
    defaults: dict
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill unsupplied inputs, rebuild engineered features, order columns."""
        if self.defaults:
            filled = {**self.defaults}
            filled.update({k: v for k, v in frame.iloc[0].items() if pd.notna(v)})
            frame = pd.DataFrame([filled])
        if self.derive is not None:
            frame = self.derive(frame)

        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: missing features {missing}")
        return frame[self.features]

    def predict_frame(self, frame: pd.DataFrame):
        """Probabilities for a frame already carrying every model feature."""
        return probabilities(self.model, frame[self.features])

    def predict_one(self, inputs: dict) -> Prediction:
        """Score a single case given whatever fields the caller has."""
        X = self.prepare(pd.DataFrame([dict(inputs)]))
        p = float(probabilities(self.model, X)[0])
        return Prediction(probability=p, threshold=self.threshold, flag=p >= self.threshold)


def from_bundle(name: str, bundle: dict, defaults: dict | None = None) -> Predictor:
    """Normalise one saved bundle into a Predictor."""
    threshold = _first(bundle, THRESHOLD_KEYS)
    if threshold is None:
        raise KeyError(f"{name}: bundle has no threshold under any of {THRESHOLD_KEYS}")

    return Predictor(
        name=name,
        model=bundle["model"],
        features=list(bundle["features"]),
        threshold=float(threshold),
        metrics=dict(_first(bundle, METRICS_KEYS, {}) or {}),
        defaults=defaults or {},
        derive=_screening_ratios if name == SCREENING else None,
    )


def load(name: str) -> Predictor | None:
    """Load a predictor by name, or None when its artifact is not present.

    Absent models are a legitimate state - a clean checkout ships no trained
    models - so this returns None rather than raising, and callers render their
    own "run finalize_x.py" message.
    """
    if name not in FILENAMES:
        raise KeyError(f"unknown predictor {name!r}; expected one of {sorted(FILENAMES)}")

    path = MODELS / FILENAMES[name]
    if not path.exists():
        return None

    defaults = {}
    if name == SCREENING:
        medians = REPORTS / "cbc_population_medians.json"
        if medians.exists():
            defaults = json.loads(medians.read_text())

    return from_bundle(name, joblib.load(path), defaults)


def load_all() -> dict[str, Predictor | None]:
    return {name: load(name) for name in FILENAMES}


def save_bundle(path, *, model, features, threshold: float, metrics: dict | None = None, **extra):
    """Write a bundle in the canonical shape.

    Writing goes through here so new artifacts cannot invent a fifth spelling of
    the contract. `threshold` is the canonical key; `from_bundle` still reads the
    legacy `threshold_sens90` for the two bundles that already use it.
    """
    bundle = {
        "model": model,
        "features": list(features),
        "threshold": float(threshold),
        "metrics": dict(metrics or {}),
        **extra,
    }
    joblib.dump(bundle, path)
    return bundle
