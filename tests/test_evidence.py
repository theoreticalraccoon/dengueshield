"""Named access to reports/. The app reads nothing raw, so this is its contract."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dengue import evidence
from dengue.config import REPORTS

# ------------------------------------------------------------------ absence


def test_every_accessor_survives_an_empty_reports_dir(tmp_path):
    """A clean checkout has no reports/. The app must degrade, not raise."""
    assert evidence.forecasts(tmp_path) is None
    assert evidence.history(tmp_path) is None
    assert evidence.robustness(tmp_path) == {}
    assert evidence.calibration_errors(tmp_path) == {}
    assert evidence.spatial_holdout(tmp_path) is None
    assert evidence.transfer(tmp_path) is None
    assert evidence.dataset_audit(tmp_path) is None
    assert evidence.head_to_head(tmp_path) is None
    assert evidence.validation_battery(tmp_path) == []
    assert evidence.generalisation_gap(tmp_path) == (None, None)
    assert evidence.calibration_gap(tmp_path) == (None, None)


def test_headline_metrics_reports_unknown_rather_than_guessing(tmp_path):
    tasks = evidence.headline_metrics(tmp_path)
    assert set(tasks) == {"screening", "complication", "continuation", "emergence"}
    for t in tasks.values():
        assert not t.known
        assert "n/a" in t.format()


# ------------------------------------------------------------------ derived pairs


def test_generalisation_gap_is_unseen_minus_seen(tmp_path):
    pd.DataFrame(
        {
            "condition": [
                "A_temporal_seen_municipalities",
                "B_spatiotemporal_unseen_municipalities",
            ],
            "pr_auc": [0.960, 0.950],
        }
    ).to_csv(tmp_path / "model2_spatial_holdout.csv", index=False)
    unseen, gap = evidence.generalisation_gap(tmp_path)
    assert unseen == pytest.approx(0.950)
    assert gap == pytest.approx(-0.010)


def test_calibration_gap_is_negative_when_calibration_helps(tmp_path):
    (tmp_path / "model2_calibration_errors.json").write_text(
        json.dumps({"raw": {"ece": 0.019}, "calibrated": {"ece": 0.005}})
    )
    ece, gain = evidence.calibration_gap(tmp_path)
    assert ece == pytest.approx(0.005)
    assert gain == pytest.approx(-0.014)  # lower ECE is better


# ------------------------------------------------------------------ the battery


def test_battery_skips_rows_whose_artifact_is_missing(tmp_path):
    """Partial reports/ yields a partial table, never a crash or a stale literal."""
    (tmp_path / "model2_robustness.json").write_text(
        json.dumps({"horizons": [{"horizon": 2, "pr_auc": 0.96}, {"horizon": 4, "pr_auc": 0.92}]})
    )
    rows = evidence.validation_battery(tmp_path)
    assert len(rows) == 1
    label, result = rows[0]
    assert label == "Horizon sweep (2 / 4 weeks)"
    assert result == "0.960 / 0.920"


def test_battery_flags_contamination_rather_than_reassuring(tmp_path):
    (tmp_path / "leakage_audit.json").write_text(
        json.dumps({"contamination_checks": {"casos_lag_1": True, "casos_lag_2": False}})
    )
    (_, result), = evidence.validation_battery(tmp_path)
    assert result == "CONTAMINATION DETECTED"


def test_battery_against_the_real_reports():
    if not (REPORTS / "model2_robustness.json").exists():
        pytest.skip("reports/ not populated")
    rows = dict(evidence.validation_battery())
    assert any("Temporal holdout" in k for k in rows)
    assert any("Shuffled-label control" in k for k in rows)
    # The control must collapse; if it ever reads high the panel is contaminated.
    control = next(v for k, v in rows.items() if "Shuffled-label" in k)
    assert "collapses to 0.1" in control
