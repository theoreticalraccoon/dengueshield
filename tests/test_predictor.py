"""One interface over four bundles that agree on nothing else."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue import predictor as P


class FakeProba:
    """A scikit-learn style classifier: predict_proba, positive class in column 1."""

    def predict_proba(self, X):
        p = np.asarray(X.iloc[:, 0], dtype=float)
        return np.column_stack([1 - p, p])


class FakeBooster:
    """A native LightGBM Booster: predict returns the positive probability."""

    def predict(self, X):
        return np.asarray(X.iloc[:, 0], dtype=float)


def bundle(model=None, threshold_key="threshold", **extra):
    b = {"model": model or FakeProba(), "features": ["a", "b"], threshold_key: 0.4}
    b.update(extra)
    return b


# ------------------------------------------------------------------ adapters


def test_both_estimator_kinds_are_scored():
    X = pd.DataFrame({"a": [0.25, 0.75]})
    assert list(P.probabilities(FakeProba(), X)) == [0.25, 0.75]
    assert list(P.probabilities(FakeBooster(), X)) == [0.25, 0.75]


def test_booster_has_no_predict_proba():
    """The reason the seam is real rather than hypothetical."""
    assert not hasattr(FakeBooster(), "predict_proba")


# ------------------------------------------------------------------ normalising


@pytest.mark.parametrize("key", ["threshold", "threshold_sens90"])
def test_either_threshold_key_is_accepted(key):
    assert P.from_bundle("x", bundle(threshold_key=key)).threshold == 0.4


def test_missing_threshold_is_an_error_naming_both_keys():
    b = bundle()
    del b["threshold"]
    with pytest.raises(KeyError) as e:
        P.from_bundle("x", b)
    assert "threshold_sens90" in str(e.value)


@pytest.mark.parametrize("key", ["metrics", "metrics_nested_cv"])
def test_either_metrics_key_is_accepted(key):
    assert P.from_bundle("x", bundle(**{key: {"roc_auc": 0.9}})).metrics == {"roc_auc": 0.9}


def test_absent_metrics_is_an_empty_dict_not_none():
    """srilanka_outbreak.joblib genuinely carries no metrics key."""
    assert P.from_bundle("x", bundle()).metrics == {}


# ------------------------------------------------------------------ preparing


def test_features_are_reordered_to_the_trained_order():
    p = P.from_bundle("x", bundle())
    out = p.prepare(pd.DataFrame([{"b": 2.0, "a": 1.0, "ignored": 9.0}]))
    assert list(out.columns) == ["a", "b"]


def test_missing_features_are_named():
    p = P.from_bundle("x", bundle())
    with pytest.raises(KeyError, match="b"):
        p.prepare(pd.DataFrame([{"a": 1.0}]))


def test_defaults_fill_only_what_the_caller_omitted():
    p = P.from_bundle("x", bundle(), defaults={"a": 10.0, "b": 20.0})
    out = p.prepare(pd.DataFrame([{"a": 1.0}]))
    assert out.iloc[0].to_dict() == {"a": 1.0, "b": 20.0}


def test_derive_runs_after_filling():
    """Engineered features must be rebuilt from what was entered, not defaulted."""
    p = P.Predictor(
        name="x",
        model=FakeProba(),
        features=["a", "ratio"],
        threshold=0.5,
        metrics={},
        defaults={"a": 2.0, "ratio": 999.0},
        derive=lambda f: f.assign(ratio=f.a * 10),
    )
    assert p.prepare(pd.DataFrame([{"a": 3.0}])).iloc[0]["ratio"] == 30.0


# ------------------------------------------------------------------ predicting


def test_predict_one_flags_at_the_threshold():
    p = P.from_bundle("x", bundle())  # threshold 0.4
    assert p.predict_one({"a": 0.40, "b": 0}).flag is True
    assert p.predict_one({"a": 0.39, "b": 0}).flag is False


def test_prediction_rejects_an_impossible_probability():
    with pytest.raises(ValueError, match="out of range"):
        P.Prediction(probability=1.4, threshold=0.5, flag=True)


# ------------------------------------------------------------------ loading


def test_unknown_name_is_rejected():
    with pytest.raises(KeyError, match="unknown predictor"):
        P.load("not_a_model")


def test_absent_artifact_loads_as_none(monkeypatch, tmp_path):
    """A clean checkout ships no trained models; the app renders a banner."""
    monkeypatch.setattr(P, "MODELS", tmp_path)
    assert P.load(P.SCREENING) is None


@pytest.mark.parametrize("name", sorted(P.FILENAMES))
def test_real_bundles_normalise(name):
    pytest.importorskip("lightgbm")
    p = P.load(name)
    if p is None:
        pytest.skip(f"{name} artifact not present")
    assert 0.0 < p.threshold < 1.0
    assert p.features
    assert isinstance(p.metrics, dict)


def test_screening_fills_from_cohort_medians():
    pytest.importorskip("lightgbm")
    p = P.load(P.SCREENING)
    if p is None or not p.defaults:
        pytest.skip("screening model or medians not present")
    # Quick entry mode supplies 6 of 22 features; the rest come from the cohort.
    quick = {
        "Age": 32,
        "Gender": 1,
        "Total Platelet Count(/cumm)": 145_000,
        "Total WBC count(/cumm)": 4_800,
        "Hemoglobin(g/dl)": 14.2,
        "HCT(%)": 43.0,
    }
    assert 0.0 <= p.predict_one(quick).probability <= 1.0


def test_screening_ratios_follow_the_entered_values():
    """The ratios are rebuilt from input, not left at the cohort median."""
    pytest.importorskip("lightgbm")
    p = P.load(P.SCREENING)
    if p is None or not p.defaults:
        pytest.skip("screening model or medians not present")
    low = p.prepare(pd.DataFrame([{"Total Platelet Count(/cumm)": 20_000}]))
    high = p.prepare(pd.DataFrame([{"Total Platelet Count(/cumm)": 400_000}]))
    assert low.iloc[0]["PLT_WBC_ratio"] < high.iloc[0]["PLT_WBC_ratio"]
    assert low.iloc[0]["PLT_WBC_ratio"] != p.defaults["PLT_WBC_ratio"]
