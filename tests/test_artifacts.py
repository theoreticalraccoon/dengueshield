"""Report artifacts have producers, and the producers reproduce them exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dengue import artifacts
from dengue.config import REPORTS

# ------------------------------------------------------------------ comparison


def test_identical_has_no_differences(tmp_path):
    obj = {"a": 1.0, "nested": {"b": 2.5}}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(obj))
    assert artifacts.differences(obj, p) == []


def test_absent_file_is_reported(tmp_path):
    assert artifacts.differences({"a": 1}, tmp_path / "gone.json") == ["gone.json: absent"]


def test_value_mismatch_names_the_key(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"outer": {"inner": 1.0}}))
    (diff,) = artifacts.differences({"outer": {"inner": 2.0}}, p)
    assert "outer.inner" in diff


def test_missing_and_extra_keys_are_reported(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"kept": 1, "only_disk": 2}))
    diffs = artifacts.differences({"kept": 1, "only_rebuilt": 3}, p)
    assert any("only_disk: only on disk" in d for d in diffs)
    assert any("only_rebuilt: only rebuilt" in d for d in diffs)


def test_float_noise_below_precision_is_not_a_difference(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 0.12345678901234}))
    assert artifacts.differences({"a": 0.12345678901239}, p, places=10) == []


# ------------------------------------------------------------------ producers


def test_population_medians_covers_every_feature():
    X = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 20.0]})
    assert artifacts.population_medians(X) == {"a": 2.0, "b": 15.0}


def test_best_balanced_threshold_is_an_observed_score():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert artifacts.best_balanced_threshold(y, p) in set(p)


# ------------------------------------------------------------------ the orphans


def _cohort():
    pytest.importorskip("lightgbm")
    from dengue.datasets import load_hematology_1523

    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "mendeley_1523.csv"
    if not raw.exists():
        pytest.skip("screening cohort not present")
    return load_hematology_1523()


# ------------------------------------------------------------------ inventory


def test_inventory_has_no_duplicate_names():
    names = [a.name for a in artifacts.INVENTORY]
    assert len(names) == len(set(names))


def test_every_artifact_has_a_producer_or_an_explanation():
    """An artifact with no producer must say why, or it is just unexplained."""
    for a in artifacts.INVENTORY:
        assert a.has_producer or a.note, f"{a.name} has neither a producer nor a note"


def test_orphans_are_the_known_set():
    """Exactly these have no producer. A new one should be a deliberate decision.

    cbc_population_medians.json and model1_operating_points.json were orphans too
    until derive_reports.py was written; they are absent here because they now
    have one.
    """
    assert {a.name for a in artifacts.orphans()} == {
        "model2_headtohead.csv",
        "model2_headtohead.json",
        "srilanka_calibration.json",
        "model2_state_folds_with_events.csv",
    }


def test_derived_artifacts_name_their_rebuilder():
    for a in artifacts.reproducible():
        assert a.producer == "derive_reports.py"


def test_inventory_covers_everything_on_disk():
    """A future freeze copies the declared set, so an undeclared file is dropped."""
    if not REPORTS.exists():
        pytest.skip("reports/ not present")
    on_disk = {p.name for p in REPORTS.glob("*") if p.is_file()}
    if not on_disk:
        pytest.skip("reports/ is empty")
    assert on_disk - set(artifacts.BY_NAME) == set(), "undeclared report artifacts"


def test_freezable_only_returns_existing_declared_files(tmp_path):
    (tmp_path / "dataset_audit.csv").write_text("x")
    (tmp_path / "not_declared.csv").write_text("x")
    names = {p.name for p in artifacts.freezable(tmp_path)}
    assert names == {"dataset_audit.csv"}


# ------------------------------------------------------------------ the orphans


def test_population_medians_reproduce_the_committed_file():
    """cbc_population_medians.json is load-bearing: Quick entry mode fills from it."""
    target = REPORTS / "cbc_population_medians.json"
    if not target.exists():
        pytest.skip("artifact not present")
    X, _, _ = _cohort()
    assert artifacts.differences(artifacts.population_medians(X), target) == []


def test_operating_points_reproduce_the_committed_file():
    target = REPORTS / "model1_operating_points.json"
    oof = REPORTS / "model1_oof_calibrated.npy"
    if not (target.exists() and oof.exists()):
        pytest.skip("artifact not present")
    _, y, _ = _cohort()
    p = np.load(oof)
    thr = artifacts.best_balanced_threshold(y, p)
    assert artifacts.differences(artifacts.operating_points(y, p, thr), target) == []
