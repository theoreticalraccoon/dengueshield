"""Named access to reports/. The app reads nothing raw, so this is its contract."""

from __future__ import annotations

import json
import os

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


# ------------------------------------------------------------------ which country


def test_the_headline_continuation_row_is_the_deployed_sri_lanka_model(tmp_path):
    """It used to be Brazil, in a table beside Sri Lankan emergence.

    Both figures came from artifacts, so the fix that made emergence honest looked
    like it covered this too. Nothing checked they came from the same country.
    """
    (tmp_path / "srilanka_continuation.json").write_text(
        json.dumps({"pr_auc": 0.7253, "baseline_persistence_pr_auc": 0.4688})
    )
    (tmp_path / "model2_summary.json").write_text(
        json.dumps([{"model": "LightGBM", "pr_auc": 0.9599}])
    )
    cont = evidence.headline_metrics(tmp_path)["continuation"]
    assert cont.score == pytest.approx(0.7253)
    assert cont.baseline == pytest.approx(0.4688)
    assert "Sri Lank" in cont.asked_of


def test_a_brazil_artifact_alone_leaves_the_continuation_row_unknown(tmp_path):
    """Absent is honest; borrowing another country's number is not."""
    (tmp_path / "model2_summary.json").write_text(
        json.dumps([{"model": "LightGBM", "pr_auc": 0.9599}])
    )
    assert not evidence.headline_metrics(tmp_path)["continuation"].known


def test_brazil_is_still_reported_but_labelled_as_an_experiment(tmp_path):
    (tmp_path / "model2_summary.json").write_text(
        json.dumps(
            [
                {"model": "LightGBM", "pr_auc": 0.9599},
                {"model": "BASELINE_persistence", "pr_auc": 0.7577},
            ]
        )
    )
    br = evidence.brazil_continuation(tmp_path)
    assert br.score == pytest.approx(0.9599)
    assert br.baseline == pytest.approx(0.7577)
    assert "not deployed" in br.asked_of


def test_calibration_status_is_read_not_asserted(tmp_path):
    """The app said "not calibrated" as a literal; it now asks the artifacts."""
    assert evidence.srilanka_calibration(tmp_path) == {}
    (tmp_path / "srilanka_emergence.json").write_text(
        json.dumps({"ece": 0.039, "ece_uncalibrated": 0.292})
    )
    assert evidence.srilanka_calibration(tmp_path) == {"emergence": (0.292, 0.039)}


# ------------------------------------------------------------------ cache key


def test_artifact_version_changes_when_an_artifact_changes(tmp_path):
    """The app's cached loaders key on this; a stale key is a stale dashboard.

    The rewrite is aged explicitly rather than just performed twice in a row. Two
    writes inside one filesystem timestamp tick can land on the same `st_mtime_ns`,
    and this test used to fail intermittently on exactly that - reliably under a
    full-suite run, where the machine is warm enough to get both writes into one
    tick, and never when run alone. Setting the time makes the test assert the
    contract (a modified file invalidates the key) instead of asserting that the
    clock happened to tick.
    """
    models, reports, proc = tmp_path / "m", tmp_path / "r", tmp_path / "p"
    for d in (models, reports, proc):
        d.mkdir()
    artifact = reports / "srilanka_dual_risk.csv"
    artifact.write_text("district,emergence_risk\nX,0.7\n")

    before = evidence.artifact_version(models, reports, proc)
    assert before == evidence.artifact_version(models, reports, proc)  # stable

    artifact.write_text("district,emergence_risk\nX,0.1\n")
    stat = artifact.stat()
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert evidence.artifact_version(models, reports, proc) != before


def test_artifact_version_notices_a_resize_within_the_same_tick(tmp_path):
    """Size is in the key too, so a same-instant rewrite of different length counts.

    Modification time alone would miss this, and a weekly refresh that rewrote a
    forecast in under a filesystem tick would leave the dashboard serving the old
    one out of cache.
    """
    models, reports, proc = tmp_path / "m", tmp_path / "r", tmp_path / "p"
    for d in (models, reports, proc):
        d.mkdir()
    artifact = reports / "srilanka_dual_risk.csv"
    artifact.write_text("district,emergence_risk\nX,0.7\n")
    before = evidence.artifact_version(models, reports, proc)

    stat = artifact.stat()
    artifact.write_text("district,emergence_risk\nX,0.7\nY,0.2\nZ,0.9\n")
    # Pin the timestamp back to what it was: only the size differs now.
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert evidence.artifact_version(models, reports, proc) != before


def test_artifact_version_notices_a_new_artifact(tmp_path):
    reports = tmp_path / "r"
    reports.mkdir()
    before = evidence.artifact_version(tmp_path / "m", reports, tmp_path / "p")
    (reports / "srilanka_continuation.json").write_text("{}")
    assert evidence.artifact_version(tmp_path / "m", reports, tmp_path / "p") != before


def test_artifact_version_survives_a_clean_checkout(tmp_path):
    """No models and no reports is a legitimate state, not a crash."""
    assert evidence.artifact_version(tmp_path / "m", tmp_path / "r", tmp_path / "p") == ()
