"""Assertions about the artifacts actually on disk, not about the code that makes them.

Two defects motivated this file, and both were invisible to every other test:

  * `reports/model1_final.json` carried a screening operating point with
    specificity 0.0 - a model that flags every patient - while the bundle beside
    it carried a working one. Nothing compared them.
  * `finalize_peds.py` reported metrics from a tuned nested-CV and deployed a
    hand-written estimator with fixed hyperparameters, so the published ROC-AUC
    described a model that was never saved.

Neither is a bug in a function. Both are a published number and a shipped artifact
disagreeing, which is only checkable against the artifacts themselves. Every test
here skips when its artifact is absent, because a clean checkout ships none of them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dengue import predictor
from dengue.config import REPORTS


def _load(name):
    pytest.importorskip("lightgbm")
    p = predictor.load(name)
    if p is None:
        pytest.skip(f"{name} artifact not present")
    return p


def _json(name):
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text())


def _npy(name):
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return np.load(path)


def _screening_labels():
    """The screening cohort's labels, or a skip when the raw dataset is absent.

    This file already skips whenever a MODEL artifact is missing; the raw clinical
    cohort needed the same treatment and did not have it. `data/raw/` is not in the
    repo - the patient-level datasets are deliberately not published - so the tests
    built on it passed on a developer machine and failed on the first clean
    checkout, which is exactly the class of failure CI exists to catch and a poor
    way to find out.
    """
    from dengue.datasets import load_hematology_1523

    try:
        return load_hematology_1523()[1]
    except FileNotFoundError as missing:
        pytest.skip(f"screening cohort not present in this checkout ({missing.filename})")


# ------------------------------------------------------------------ screening


def test_the_screening_operating_point_excludes_somebody():
    """A threshold below every score flags all 1,511 patients and screens nothing.

    This is what reports/model1_final.json recorded: sensitivity 1.0, specificity
    0.0, accuracy exactly the prevalence. It is a degenerate operating point, not a
    conservative one.
    """
    p = _load(predictor.SCREENING)
    oof = _npy("model1_oof_calibrated.npy")
    flagged = oof >= p.threshold
    assert flagged.mean() < 1.0, "the threshold flags every patient"
    assert flagged.mean() > 0.0, "the threshold flags nobody"


def test_the_screening_threshold_still_meets_its_sensitivity_target():
    p = _load(predictor.SCREENING)
    y = _screening_labels()
    oof = _npy("model1_oof_calibrated.npy")
    if len(y) != len(oof):
        pytest.skip("cohort and saved out-of-fold predictions are different vintages")

    flagged = oof >= p.threshold
    sensitivity = flagged[y == 1].mean()
    specificity = (~flagged)[y == 0].mean()
    assert sensitivity >= 0.90, f"operating point misses too much: sens={sensitivity:.3f}"
    assert specificity > 0.0, "specificity 0.0 means the threshold rules nothing out"


def test_the_published_screening_metrics_describe_the_shipped_threshold():
    """reports/ and the bundle must agree on what the operating point is."""
    p = _load(predictor.SCREENING)
    final = _json("model1_final.json")
    published = final.get("at_sens90", {}).get("threshold")
    if published is None:
        pytest.skip("model1_final.json has no at_sens90 block")
    assert published == pytest.approx(p.threshold, rel=1e-6), (
        "reports/model1_final.json describes a different operating point from the "
        "one models/model1_screening.joblib actually uses"
    )


# ------------------------------------------------------------------ complication


def test_the_published_complication_metrics_reproduce_from_the_saved_predictions():
    """peds_final.json must be recomputable from peds_oof.npy at its own threshold."""
    from dengue.ablation import full_metrics, load_peds

    published = _json("peds_final.json")
    oof = _npy("peds_oof.npy")
    y = load_peds()[1]
    if len(y) != len(oof):
        pytest.skip("cohort and saved out-of-fold predictions are different vintages")

    recomputed = full_metrics(y, oof, published["threshold"])
    for key in ("roc_auc", "pr_auc", "sensitivity", "specificity", "npv", "tp", "fn"):
        assert recomputed[key] == pytest.approx(published[key], rel=1e-9), key


def test_the_deployed_complication_model_was_tuned_not_hand_written():
    """The metrics come from a tuned protocol, so the artifact must be tuned too.

    Checked structurally rather than by score: the saved estimator has to be a
    calibrated wrapper around a searched pipeline, which is what
    `ablation.fit_final` produces and what the hand-written LightGBM was not.
    """
    p = _load(predictor.PEDS)
    inner = getattr(p.model, "calibrated_classifiers_", None)
    assert inner, "the deployed model is not calibrated"
    pipeline = inner[0].estimator
    assert hasattr(pipeline, "named_steps"), "expected the impute -> clf pipeline"
    assert "clf" in pipeline.named_steps


# ------------------------------------------------------------------ every bundle


@pytest.mark.parametrize("name", sorted(predictor.FILENAMES))
def test_no_bundle_ships_a_threshold_at_an_extreme(name):
    """0.0 or 1.0 means the operating point collapsed, whichever model it is."""
    p = _load(name)
    assert 0.0 < p.threshold < 1.0


# ------------------------------------------------------------------ Sri Lanka


@pytest.mark.parametrize("name", [predictor.SL_CONTINUATION, predictor.SL_EMERGENCE])
def test_the_sri_lanka_bundles_ship_a_calibrator(name):
    """Trained with scale_pos_weight, so a raw score is not a probability."""
    p = _load(name)
    assert p.calibrator is not None, (
        f"{name} has no calibrator; its scores are not probabilities and the "
        "dashboard bands and operating point are drawn on the wrong scale"
    )


@pytest.mark.parametrize("name", [predictor.SL_CONTINUATION, predictor.SL_EMERGENCE])
def test_the_sri_lanka_bundles_ship_a_seed_ensemble(name):
    """A single booster is a draw from a distribution ~0.02 PR-AUC wide.

    Checked through the estimator's structure rather than its exact class, because
    emergence now deploys a `BlendedEmergence` whose two MEMBERS are seed ensembles
    (see dengue.blend). The property being defended is unchanged - nothing here may
    ship one booster - so the test asserts that property rather than a type name.
    """
    from dengue.blend import BlendedEmergence
    from dengue.validation import SeedEnsemble

    p = _load(name)
    members = (
        [p.model.classifier, p.model.regressor]
        if isinstance(p.model, BlendedEmergence)
        else [p.model]
    )
    for member in members:
        assert isinstance(member, SeedEnsemble), f"{name} deploys one seed, not an average"
        assert len(member) > 1


@pytest.mark.parametrize(
    "artifact", ["srilanka_continuation.json", "srilanka_emergence.json"]
)
def test_calibration_did_not_improve_the_ranking(artifact):
    """The guard against a calibrator that saw test labels.

    Isotonic cannot reverse a pair, so it cannot rank better than the raw score. It
    can only merge pairs into ties, which costs a little. A calibrated PR-AUC above
    the raw one is not a better model; it is leakage.
    """
    m = _json(artifact)
    raw, calibrated = m.get("pr_auc"), m.get("pr_auc_calibrated")
    if raw is None or calibrated is None:
        pytest.skip(f"{artifact} predates the calibration split")
    assert calibrated <= raw + 1e-12


@pytest.mark.parametrize(
    "artifact", ["srilanka_continuation.json", "srilanka_emergence.json"]
)
def test_calibration_actually_improved_calibration(artifact):
    m = _json(artifact)
    if "ece_uncalibrated" not in m:
        pytest.skip(f"{artifact} predates calibration")
    assert m["ece"] < m["ece_uncalibrated"]
    assert m["brier"] < m["brier_uncalibrated"]


def test_a_forecast_never_claims_certainty():
    """Isotonic's end steps are unregularised; 1.0 on a dashboard is a lie."""
    from dengue.calibration import CERTAINTY_CAP

    p = _load(predictor.SL_CONTINUATION)
    forecasts = REPORTS / "srilanka_current_risk.csv"
    if not forecasts.exists():
        pytest.skip("no current forecast")
    import pandas as pd

    risk = pd.read_csv(forecasts).risk
    assert risk.max() <= 1.0 - CERTAINTY_CAP + 1e-9, "a district is forecast at 100%"
    assert risk.min() >= 0.0
    assert p.calibrator is not None
