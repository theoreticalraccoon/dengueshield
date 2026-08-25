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


# ----------------------------------------------------------------- new features


def wide_panel(n_weeks=400, n_districts=3, seed=0):
    """A panel carrying every column the climate and importation blocks need."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2012-01-02", periods=n_weeks, freq="7D")
    rows = []
    for i, district in enumerate(("Colombo", "Gampaha", "Kandy")[:n_districts]):
        season = np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
        casos = rng.poisson(12 + 8 * np.maximum(season, 0) + 3 * i, n_weeks).astype(float)
        pop = 1_000_000 + 100_000 * i
        rows.append(
            pd.DataFrame(
                {
                    "district": district,
                    "week_start": weeks,
                    "casos": casos,
                    "population_total": pop,
                    "p_inc100k": casos / pop * 1e5,
                    "precip_total_semana": rng.gamma(2, 20, n_weeks) * (1 + season),
                    "tempmed": 27 + 2 * season + rng.normal(0, 0.5, n_weeks),
                    "tempmax": 32 + 2 * season + rng.normal(0, 0.5, n_weeks),
                    "tempmin": 22 + 2 * season + rng.normal(0, 0.5, n_weeks),
                    "umidmed": 75 + 5 * season + rng.normal(0, 2, n_weeks),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("cut", [200, 300, 380])
def test_no_new_feature_can_see_the_future(cut):
    """The gate on the whole accuracy push, stated as a property.

    A feature computed at week T must not change when the weeks after T are
    deleted. Everything else in this file tests a specific window; this one tests
    the invariant that makes any of the windows trustworthy, over every declared
    feature at once - including the anomaly z-scores, whose reference distribution
    is drawn from other YEARS and is the easiest place to leak by accident.
    """
    full = wide_panel()
    cut_week = sorted(full.week_start.unique())[cut]

    def build(df):
        return emergence.add_extra_features(
            emergence.add_history_features(emergence.add_base_features(df))
        )

    a = build(full)
    b = build(full[full.week_start <= cut_week])

    cols = list(emergence.EXTRA_FEATURES + emergence.HISTORY_FEATURES)
    at = a[a.week_start == cut_week].sort_values("district")[cols].to_numpy(dtype=float)
    bt = b[b.week_start == cut_week].sort_values("district")[cols].to_numpy(dtype=float)
    assert np.allclose(at, bt, equal_nan=True), "a feature changed when the future was removed"


def test_every_declared_extra_feature_is_produced():
    out = emergence.add_extra_features(
        emergence.add_history_features(emergence.add_base_features(wide_panel()))
    )
    assert set(emergence.EXTRA_FEATURES) <= set(out.columns)


def test_shipped_features_are_a_subset_of_what_is_built():
    """Nothing can be deployed that the feature builder does not produce."""
    assert set(emergence.SHIPPED_FEATURES) <= set(emergence.EXTRA_FEATURES)


def test_weeks_since_national_peak_stays_inside_its_training_range():
    """The feature must not become a proxy for the calendar.

    An expanding-MAX version of this counter is set once and never reset, so it
    climbs monotonically and every later row sits beyond anything the model was
    fitted on - and a tree cannot extrapolate. The quartile version re-arms.
    """
    out = emergence.add_national_features(wide_panel(n_weeks=600))
    early = out[out.week_start < "2018-01-01"].weeks_since_national_peak
    late = out[out.week_start >= "2018-01-01"].weeks_since_national_peak
    assert late.max() <= early.max(), "the counter is still a monotone time index"


# ------------------------------------------------------------------ lead times


def test_lead_time_is_the_weeks_until_the_district_actually_crosses():
    df = panel([COLD] * 3 + [HOT] + [COLD] * 8)
    out = emergence.label_emergence(df, horizon=4, quiet=2, keep_ineligible=True)
    # Row 0 sees the HOT at index 3, which is three weeks after it.
    assert out.iloc[0].weeks_to_crossing == 3.0
    # Row 2 sits one week before the crossing.
    assert out.iloc[2].weeks_to_crossing == 1.0


def test_lead_time_is_blank_where_nothing_emerges():
    out = emergence.label_emergence(panel([COLD] * 12), horizon=4, quiet=2, keep_ineligible=True)
    assert out.weeks_to_crossing.isna().all()


def test_lead_times_summarises_only_the_caught_outbreaks():
    df = panel([COLD] * 3 + [HOT] + [COLD] * 8)
    labelled = emergence.label_emergence(df, horizon=4, quiet=2, keep_ineligible=True).dropna(
        subset=["y"]
    )
    caught_none = emergence.lead_times(labelled, np.zeros(len(labelled), dtype=bool))
    assert caught_none == {}

    caught_all = emergence.lead_times(labelled, np.ones(len(labelled), dtype=bool))
    assert caught_all["n_caught"] == int(labelled.y.sum())
    assert caught_all["mean_weeks"] > 0
