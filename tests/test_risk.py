"""The district risk rule, including the case that was wrong in three places."""

from __future__ import annotations

import pandas as pd
import pytest

from dengue import risk

THR = 0.5


def row(district="Colombo", in_outbreak=False, continuation=0.9, emergence=None):
    """One dual-risk row. `emergence=None` is the blank the app used to misread."""
    return pd.Series(
        {
            "district": district,
            "currently_in_outbreak": in_outbreak,
            "continuation_risk": continuation,
            "emergence_risk": float("nan") if emergence is None else emergence,
        }
    )


# ------------------------------------------------------------------ the rule


def test_in_outbreak_is_answered_by_continuation():
    v = risk.assess(row(in_outbreak=True, continuation=0.82), THR)
    assert v.triage == risk.OUTBREAK_NOW
    assert v.headline_risk == 0.82
    assert v.assessable
    assert v.status == "In outbreak"


def test_in_outbreak_ignores_a_blank_emergence():
    """Emergence is blank by construction for districts already in outbreak."""
    v = risk.assess(row(in_outbreak=True, continuation=0.7, emergence=None), THR)
    assert v.triage == risk.OUTBREAK_NOW
    assert v.headline_risk == 0.7


def test_quiet_and_scored_above_threshold_is_likely():
    v = risk.assess(row(continuation=0.9, emergence=0.61), THR)
    assert v.triage == risk.OUTBREAK_LIKELY
    assert v.headline_risk == 0.61  # emergence, not the 0.9 continuation


def test_quiet_and_scored_below_threshold_is_clear():
    v = risk.assess(row(continuation=0.9, emergence=0.10), THR)
    assert v.triage == risk.CLEAR
    assert v.headline_risk == 0.10


def test_threshold_is_inclusive():
    assert risk.assess(row(emergence=THR), THR).triage == risk.OUTBREAK_LIKELY
    assert risk.assess(row(emergence=THR - 1e-9), THR).triage == risk.CLEAR


# ------------------------------------------------------------------ the defect


def test_blank_emergence_is_not_assessable_not_a_continuation_risk():
    """The regression this module exists to prevent.

    A quiet district with no emergence score must not borrow the continuation
    model's number - that model is fitted on a population it is not in.
    """
    v = risk.assess(row(continuation=0.88, emergence=None), THR)
    assert v.triage == risk.NOT_ASSESSABLE
    assert v.headline_risk is None
    assert v.band is None
    assert not v.assessable
    assert v.continuation_risk == 0.88  # still readable, just not the headline


# ------------------------------------------------------------------ banding


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, "Low"),
        (0.25, "Low"),  # edges are closed at the top, as pd.cut had them
        (0.2500001, "Moderate"),
        (0.50, "Moderate"),
        (0.75, "High"),
        (0.7500001, "Very High"),
        (1.0, "Very High"),
    ],
)
def test_band_edges(p, expected):
    assert risk.band_for(p) == expected


def test_band_of_nothing_is_nothing():
    assert risk.band_for(None) is None
    assert risk.band_for(float("nan")) is None


# ------------------------------------------------------------------ the frame


def _frame():
    return pd.DataFrame(
        [
            row("Colombo", in_outbreak=True, continuation=0.95),
            row("Kandy", continuation=0.80, emergence=0.72),
            row("Galle", continuation=0.60, emergence=0.05),
            row("Mannar", continuation=0.88, emergence=None),
        ]
    )


def test_frame_and_row_cannot_drift_apart():
    """The property that keeps the table and the detail read-out honest."""
    df = _frame()
    assessed = risk.assess_frame(df, THR)
    for original, got in zip(df.itertuples(), assessed.itertuples(), strict=True):
        expected = risk.assess(original, THR)
        assert got.triage == expected.triage
        assert got.band == expected.band  # both None, or both the same string
        assert got.assessable == expected.assessable
        if expected.headline_risk is None:
            assert pd.isna(got.headline_risk)
        else:
            assert got.headline_risk == pytest.approx(expected.headline_risk)


def test_frame_does_not_mutate_its_input():
    df = _frame()
    before = df.copy()
    risk.assess_frame(df, THR)
    pd.testing.assert_frame_equal(df, before)


def test_frame_keeps_headline_numeric_even_when_all_blank():
    """An all-blank column must stay float64 or downstream formatting breaks."""
    df = pd.DataFrame([row("Mannar", emergence=None), row("Mullaitivu", emergence=None)])
    assessed = risk.assess_frame(df, THR)
    assert assessed.headline_risk.dtype == "float64"
    assert assessed.headline_risk.isna().all()


def test_triage_counts_include_empty_groups():
    counts = risk.triage_counts(risk.assess_frame(_frame(), THR))
    assert list(counts) == list(risk.TRIAGE_ORDER)
    assert counts == {
        risk.OUTBREAK_NOW: 1,
        risk.OUTBREAK_LIKELY: 1,
        risk.CLEAR: 1,
        risk.NOT_ASSESSABLE: 1,
    }


def test_reason_renders_the_threshold():
    v = risk.assess(row(emergence=0.9), THR)
    assert "50%" in v.reason(THR)
