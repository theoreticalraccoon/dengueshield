"""The app renders and both prediction buttons work.

These run the real Streamlit script against the real artifacts, so they are slow
(a few seconds per screen) and they depend on models/ and reports/ being present.
They are skipped rather than failed when the artifacts are absent, because a clean
checkout without the trained models is a legitimate state - CI refreshes data, it
does not retrain.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from dengue import risk

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DUAL = ROOT / "reports" / "srilanka_dual_risk.csv"

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

SCREENS = ["Patient assessment", "Outbreak forecast", "About the models"]


def app(screen):
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.radio[0].set_value(screen).run()
    return at


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("screen", SCREENS)
def test_screen_renders(screen):
    assert_clean(app(screen))


def test_screening_button():
    at = app("Patient assessment")
    at.button(key="screen_btn").click().run()
    assert_clean(at)
    assert at.metric, "no result metric rendered"


def test_complication_button():
    at = app("Patient assessment")
    at.button(key="peds_btn").click().run()
    assert_clean(at)


def _unassessable_district():
    """A district the emergence model declines to score, from live data."""
    if not DUAL.exists():
        return None
    thr = 0.5
    bundle = ROOT / "models" / "srilanka_emergence.joblib"
    if bundle.exists():
        thr = joblib.load(bundle).get("threshold", 0.5)
    assessed = risk.assess_frame(pd.read_csv(DUAL), thr)
    quiet = assessed[assessed.triage == risk.NOT_ASSESSABLE]
    return None if quiet.empty else str(quiet.iloc[0].district)


def test_district_context_does_not_borrow_the_other_model():
    """The rendered defect: a not-assessable district must not show a number."""
    district = _unassessable_district()
    if district is None:
        pytest.skip("no not-assessable district in the current forecast")

    at = app("Patient assessment")
    picker = next(s for s in at.selectbox if s.label.startswith("District"))
    picker.set_value(district).run()
    at.button(key="screen_btn").click().run()
    assert_clean(at)

    shown = [m.value for m in at.metric]
    assert "n/a" in shown, f"expected an n/a risk for {district}, got {shown}"
