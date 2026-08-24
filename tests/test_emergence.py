"""Emergence labelling and the long-memory features.

The history features are the ones that improved the model, so their
past-only property is the thing most worth pinning: a single unshifted
expanding window would leak the future and inflate every emergence score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue import emergence
from dengue.config import SL_INC


def panel(incidences, district="Colombo", start="2020-01-06"):
    weeks = pd.date_range(start, periods=len(incidences), freq="7D")
    return pd.DataFrame(
        {
            "district": district,
            "week_start": weeks,
            "p_inc100k": [float(v) for v in incidences],
            "population_total": 1_000_000,
        }
    )


HOT = SL_INC + 10
COLD = 0.5


# ------------------------------------------------------------------ eligibility


def test_eligible_needs_the_current_week_plus_quiet_prior_weeks():
    """quiet=2 means this week AND the two before it were all below threshold."""
    df = panel([HOT, COLD, COLD, COLD])
    mask = emergence.eligible_mask(df, quiet=2).tolist()
    #  idx0 hot · idx1 only 1 quiet before · idx2 still sees the hot week at idx0
    #  idx3 is the first week with itself + 2 clean weeks behind it
    assert mask == [False, False, False, True]


def test_a_hot_district_is_never_eligible():
    df = panel([COLD, COLD, COLD, HOT])
    assert emergence.eligible_mask(df, quiet=2).tolist()[-1] is False


# ------------------------------------------------------------------ the label


def test_label_is_max_over_the_window_not_a_single_week():
    """An outbreak that starts and is caught early still counts."""
    df = panel([COLD] * 3 + [HOT] + [COLD] * 6)
    out = emergence.label_emergence(df, horizon=4, quiet=2, keep_ineligible=True)
    # week 0 sees the HOT at index 3 inside t+1..t+4
    assert out.iloc[0].y == 1.0
    # week 5 has no hot week in t+1..t+9
    assert out.iloc[5].y == 0.0


def test_rows_without_a_full_future_window_are_unlabelled():
    df = panel([COLD] * 6)
    out = emergence.label_emergence(df, horizon=4, quiet=2, keep_ineligible=True)
    assert out.y.tail(4).isna().all()


# ------------------------------------------------------------------ history


def test_weeks_since_outbreak_counts_from_the_previous_week():
    df = panel([HOT, COLD, COLD, COLD])
    got = emergence.add_history_features(df).weeks_since_outbreak.tolist()
    # Row 0 cannot see its own hot week; rows after it count 0, 1, 2.
    assert got == [100, 0, 1, 2]


def test_history_features_never_see_the_current_row():
    """The property that makes these features honest.

    Doubling the LAST week's incidence must not change any history feature on
    that week - they are functions of strictly earlier weeks.
    """
    base = panel([COLD, HOT, COLD, COLD, COLD])
    bumped = base.copy()
    bumped.loc[bumped.index[-1], "p_inc100k"] = HOT * 10

    a = emergence.add_history_features(base).iloc[-1]
    b = emergence.add_history_features(bumped).iloc[-1]

    for col in ("weeks_since_outbreak", "hist_outbreak_rate", "hist_inc_mean"):
        assert a[col] == pytest.approx(b[col], nan_ok=True), col


def test_hist_rate_is_the_fraction_of_earlier_weeks_in_outbreak():
    df = panel([HOT, HOT, COLD, COLD])
    rate = emergence.add_history_features(df).hist_outbreak_rate.tolist()
    assert np.isnan(rate[0])  # nothing earlier
    assert rate[1] == pytest.approx(1.0)  # 1 of 1 earlier week hot
    assert rate[2] == pytest.approx(1.0)  # 2 of 2
    assert rate[3] == pytest.approx(2 / 3)  # 2 of 3


def test_every_declared_history_feature_is_produced():
    out = emergence.add_history_features(panel([COLD] * 60))
    assert set(emergence.HISTORY_FEATURES) <= set(out.columns)
