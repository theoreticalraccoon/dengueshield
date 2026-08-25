"""The harness that decides whether a change shipped or was noise.

Two properties matter more than the rest. `rolling_origin` must never let a fold
train on its own test year, and it must not reach the locked test years without
being told to - those are the two ways a selection protocol quietly stops being
one. `compare` must be paired, because at these fold spreads an unpaired test
would confirm anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue import validation
from dengue.config import VAL_END


def panel(years):
    return pd.DataFrame({"anio": years, "y": [0.0] * len(years)})


# ------------------------------------------------------------------ folds


def test_each_fold_trains_only_on_strictly_earlier_years():
    df = panel([2018, 2019, 2020, 2021])
    for year, train, test in validation.rolling_origin(
        df, first_test_year=2019, last_test_year=2021
    ):
        assert test.anio.unique().tolist() == [year]
        assert train.anio.max() == year - 1
        assert year not in set(train.anio)


def test_folds_are_an_expanding_window():
    """Last year's surveillance does not stop existing, so training only grows."""
    df = panel(list(range(2016, 2022)))
    sizes = [len(tr) for _, tr, _ in validation.rolling_origin(df, first_test_year=2018)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_locked_test_years_are_unreachable_by_default():
    """The default upper bound is the validation boundary, not the end of the data."""
    df = panel(list(range(2018, 2026)))
    years = [y for y, _, _ in validation.rolling_origin(df, first_test_year=2019)]
    assert max(years) == VAL_END
    assert not [y for y in years if y > VAL_END]


def test_a_year_with_no_rows_is_skipped_rather_than_yielded_empty():
    df = panel([2018, 2019, 2021])  # 2020 missing entirely
    years = [
        y for y, _, _ in validation.rolling_origin(df, first_test_year=2019, last_test_year=2021)
    ]
    assert years == [2019, 2021]


def test_the_first_test_year_needs_something_to_train_on():
    df = panel([2019, 2020])
    years = [
        y for y, _, _ in validation.rolling_origin(df, first_test_year=2019, last_test_year=2020)
    ]
    assert years == [2020]  # 2019 has no earlier year


# ------------------------------------------------------------------ comparison


def test_identical_folds_are_reported_as_noise_not_as_a_tie_win():
    scores = [0.40, 0.55, 0.31, 0.62]
    c = validation.compare(scores, scores)
    assert c.mean_delta == 0.0
    assert (c.ci_low, c.ci_high) == (0.0, 0.0)
    assert not c.significant
    assert not c.helps


def test_a_consistent_per_fold_gain_survives_even_though_the_spread_is_larger():
    """The point of pairing: fold difficulty varies far more than the effect does."""
    baseline = [0.30, 0.55, 0.38, 0.61, 0.44, 0.52, 0.35, 0.49]
    candidate = [b + 0.02 for b in baseline]
    c = validation.compare(baseline, candidate)
    assert c.helps
    assert c.ci_low > 0
    assert np.std(baseline) > c.mean_delta  # spread dwarfs the effect


def test_an_inconsistent_gain_of_the_same_size_does_not_survive():
    baseline = [0.30, 0.55, 0.38, 0.61, 0.44, 0.52, 0.35, 0.49]
    candidate = [0.44, 0.41, 0.52, 0.47, 0.30, 0.66, 0.49, 0.35]  # same mean, scrambled
    c = validation.compare(baseline, candidate)
    assert not c.helps


def test_a_regression_is_flagged_as_a_regression_not_merely_as_not_shipping():
    c = validation.compare([0.5] * 6, [0.45] * 6)
    assert c.significant and not c.helps
    assert "REGRESSION" in c.format()


def test_mismatched_folds_are_rejected():
    with pytest.raises(ValueError, match="matching folds"):
        validation.compare([0.1, 0.2], [0.1, 0.2, 0.3])


def test_no_folds_is_an_error_rather_than_a_silent_zero():
    with pytest.raises(ValueError, match="no folds"):
        validation.compare([], [])


def test_one_fold_reports_its_difference_without_inventing_a_spread():
    c = validation.compare([0.4], [0.5])
    assert c.mean_delta == pytest.approx(0.1)
    assert c.ci_low == c.ci_high == pytest.approx(0.1)


def test_comparison_is_deterministic():
    a = [0.30, 0.55, 0.38, 0.61, 0.44]
    b = [0.33, 0.57, 0.36, 0.66, 0.47]
    assert validation.compare(a, b) == validation.compare(a, b)


# ------------------------------------------------------------------ calibration


def test_perfect_calibration_scores_zero():
    y = np.array([0, 1] * 200)
    p = np.full(400, 0.5)
    assert validation.ece(y, p, n_bins=4) == pytest.approx(0.0, abs=1e-9)


def test_a_confidently_wrong_model_scores_near_one():
    y = np.zeros(200)
    p = np.full(200, 0.98)
    assert validation.ece(y, p, n_bins=4) == pytest.approx(0.98, abs=1e-6)


def test_a_constant_score_still_returns_its_gap():
    """Degenerate quantile edges must not collapse to a divide-by-zero or a 0.0."""
    y = np.array([1.0] * 30 + [0.0] * 70)
    assert validation.ece(y, np.full(100, 0.6), n_bins=10) == pytest.approx(0.3)


# ------------------------------------------------------------------ seeds


def test_seed_averaging_pins_every_lightgbm_randomness_source():
    """seed alone leaves bagging and feature sampling on their defaults."""
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"f{i}" for i in range(6)])
    frame["y"] = (frame.f0 + rng.normal(scale=0.5, size=400) > 0).astype(float)
    feats = [f"f{i}" for i in range(6)]
    params = {"objective": "binary", "verbose": -1, "subsample": 0.7, "subsample_freq": 1}

    captured = []
    real_train = lgb.train

    def spy(p, *a, **k):
        captured.append(p)
        return real_train(p, *a, **k)

    lgb.train = spy
    try:
        out = validation.fit_predict_seeds(
            params, frame, frame, feats, num_boost_round=5, seeds=(1, 2, 3)
        )
    finally:
        lgb.train = real_train

    assert len(captured) == 3
    for p, s in zip(captured, (1, 2, 3), strict=True):
        assert p["seed"] == p["bagging_seed"] == p["feature_fraction_seed"] == s
    assert len(out) == len(frame)


# ------------------------------------------------------- the second resampling axis


def clustered(n_districts=20, per_district=200, effect=0.0, seed=0):
    """Labels, districts, and two scores where the second is `effect` better."""
    rng = np.random.default_rng(seed)
    district = np.repeat([f"d{i}" for i in range(n_districts)], per_district)
    y = (rng.random(n_districts * per_district) < 0.08).astype(float)
    base = rng.random(len(y)) + 0.35 * y
    cand = base + effect * y
    return y, district, base, cand


def test_fast_average_precision_matches_sklearn_without_ties():
    from sklearn.metrics import average_precision_score

    y, _, p, _ = clustered(seed=5)
    assert validation.fast_average_precision(y, p) == pytest.approx(average_precision_score(y, p))


def test_fast_average_precision_handles_a_fold_with_no_positives():
    assert validation.fast_average_precision(np.zeros(10), np.random.random(10)) == 0.0


def test_pooled_compare_resamples_districts_not_rows():
    """The interval must be reported over the number of CLUSTERS, not observations."""
    y, district, base, cand = clustered(n_districts=12)
    c = validation.pooled_compare(y, district, base, cand, n_boot=200)
    assert c.n_folds == 12
    assert c.unit == "districts"
    assert "12 districts" in c.format()


def test_pooled_compare_finds_a_real_effect_and_not_a_null_one():
    y, district, base, cand = clustered(effect=0.5, seed=1)
    assert validation.pooled_compare(y, district, base, cand, n_boot=400).helps

    y, district, base, _ = clustered(effect=0.0, seed=1)
    null = validation.pooled_compare(y, district, base, base.copy(), n_boot=400)
    assert not null.significant


def test_pooled_compare_rejects_mismatched_lengths():
    y, district, base, cand = clustered(n_districts=4, per_district=10)
    with pytest.raises(ValueError, match="matching lengths"):
        validation.pooled_compare(y, district, base, cand[:-1])


def test_cluster_ci_brackets_its_own_point_estimate():
    y, district, p, _ = clustered(n_districts=15, seed=2)
    score, lo, hi = validation.cluster_ci(y, district, p, n_boot=400)
    assert lo <= score <= hi
    assert lo < hi, "a 26-district bootstrap should not return a degenerate interval"
