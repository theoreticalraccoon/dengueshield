"""Calibrating the Sri Lanka scores without moving what they rank.

The property worth pinning hardest is that calibration is rank-preserving. If a
calibrated PR-AUC differs from the raw one, the calibrator was fitted on data it
should not have seen - and that is precisely the failure that would look like an
improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from dengue import calibration, predictor
from dengue.config import VAL_END


def _tied_isotonic(n=4000, seed=3):
    """A fitted isotonic calibrator and the scores it was fitted on."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 6, n)
    y = (rng.random(n) < np.clip(p * 0.9, 0, 1)).astype(float)
    iso = IsotonicRegression(
        out_of_bounds="clip",
        y_min=calibration.CERTAINTY_CAP,
        y_max=1.0 - calibration.CERTAINTY_CAP,
    ).fit(p, y)
    return iso, p, y


def test_plain_isotonic_loses_ranking_to_ties():
    """The defect being fixed, pinned so it cannot quietly come back."""
    iso, p, y = _tied_isotonic()
    assert average_precision_score(y, iso.predict(p)) < average_precision_score(y, p)


def test_tie_broken_isotonic_restores_the_raw_ranking_exactly():
    iso, p, y = _tied_isotonic()
    q = calibration.TieBrokenIsotonic(iso, p).predict(p)
    assert average_precision_score(y, q) == pytest.approx(average_precision_score(y, p))


def test_tie_broken_isotonic_leaves_no_ties():
    iso, p, _ = _tied_isotonic()
    q = calibration.TieBrokenIsotonic(iso, p).predict(p)
    assert len(np.unique(q)) == len(np.unique(p))


def test_tie_break_moves_probabilities_by_less_than_a_millionth():
    """It must reorder within a step without changing what the number claims."""
    iso, p, _ = _tied_isotonic()
    tb = calibration.TieBrokenIsotonic(iso, p)
    assert np.abs(tb.predict(p) - iso.predict(p)).max() <= tb.eps


def test_tie_break_never_crosses_an_isotonic_step():
    """`eps` below half the smallest gap means the mapping stays monotone."""
    iso, p, _ = _tied_isotonic()
    q = calibration.TieBrokenIsotonic(iso, p).predict(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-15)


def test_tie_break_orders_scores_outside_the_fitted_range():
    """arctan rather than a clipped ramp: extremes must still be separated.

    A test-year score above anything seen in development is exactly where a clipped
    tie-break would saturate and reintroduce the ties this class exists to remove.
    """
    iso, p, _ = _tied_isotonic()
    tb = calibration.TieBrokenIsotonic(iso, p)
    far = tb.predict(np.array([p.max() * 2, p.max() * 3]))
    assert far[1] > far[0]


def test_epsilon_survives_isotonic_levels_separated_by_float_noise():
    """The failure that shipped an inert tie-break and looked like it worked.

    Isotonic emits adjacent levels 1e-16 apart routinely. Sizing eps at half the
    smallest gap then puts it below `np.spacing` at the values involved, the nudge
    rounds off on addition, every tie survives, and the only symptom is that the
    calibrated PR-AUC never recovers. Every unit test still passed.
    """
    iso, p, _ = _tied_isotonic()
    # Force a pair of levels a single ULP apart, as the real fit produced.
    iso.y_thresholds_ = np.asarray(iso.y_thresholds_, dtype=float).copy()
    iso.y_thresholds_[1] = np.nextafter(iso.y_thresholds_[0], 1.0)

    tb = calibration.TieBrokenIsotonic(iso, p)
    assert tb.eps >= calibration.TieBrokenIsotonic.MIN_EPS
    # The nudge must actually change the number it is added to.
    q = tb.predict(p)
    assert len(np.unique(q)) == len(np.unique(p))


def test_epsilon_is_never_large_enough_to_move_a_reported_probability():
    iso, p, _ = _tied_isotonic()
    tb = calibration.TieBrokenIsotonic(iso, p)
    assert tb.eps <= calibration.TieBrokenIsotonic.MAX_EPS
    # Four decimal places is what the reports carry; nothing there may move.
    assert np.allclose(tb.predict(p), iso.predict(p), atol=1e-5)


def test_tie_broken_calibrator_survives_a_pickle_round_trip():
    """It ships inside the joblib bundle, so it has to pickle like any estimator."""
    import pickle

    iso, p, _ = _tied_isotonic()
    tb = calibration.TieBrokenIsotonic(iso, p)
    restored = pickle.loads(pickle.dumps(tb))
    assert np.allclose(restored.predict(p), tb.predict(p))


def dev_frame(years=range(2014, VAL_END + 1), per_year=260, seed=0):
    """A panel where the label really is predictable from the features."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        x = rng.normal(size=per_year)
        noise = rng.normal(scale=0.8, size=per_year)
        rows.append(
            pd.DataFrame(
                {
                    "anio": year,
                    "f0": x,
                    "f1": rng.normal(size=per_year),
                    "y": (x + noise > 0.9).astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


PARAMS = {"objective": "binary", "verbose": -1, "num_leaves": 7, "learning_rate": 0.1}


# ------------------------------------------------------------------ pass-through


def test_no_calibrator_is_a_pass_through_not_an_error():
    """Bundles written before calibration existed must behave exactly as before."""
    p = [0.1, 0.9, 0.44]
    assert calibration.apply_calibrator(None, p).tolist() == p


def test_output_is_clipped_into_the_probability_range():
    class Wild:
        def predict(self, p):
            return np.asarray(p) * 10 - 3

    out = calibration.apply_calibrator(Wild(), [0.0, 0.5, 1.0])
    assert out.min() >= 0.0 and out.max() <= 1.0


# ------------------------------------------------------------------ the fit


def test_calibration_cannot_improve_the_ranking():
    """The one-sided invariant, and the guard against a leaking calibrator.

    Isotonic never reverses a pair, so it cannot make the ranking better; it only
    merges some pairs into ties, which costs a little. A calibrated PR-AUC ABOVE
    the raw one would mean the calibrator had seen the labels it was scoring.
    """
    pytest.importorskip("lightgbm")
    dev = dev_frame()
    _iso, report = calibration.fit_isotonic_oof(
        PARAMS, dev, ["f0", "f1"], num_boost_round=40, first_test_year=2016, seeds=(1,)
    )
    assert report.pr_auc_calibrated <= report.pr_auc_raw + 1e-12
    # ...and the tie-merging cost stays small, or the calibrator is destroying signal
    assert report.pr_auc_raw - report.pr_auc_calibrated < 0.05


def test_calibration_improves_the_calibration_it_was_fitted_on():
    pytest.importorskip("lightgbm")
    dev = dev_frame()
    # scale_pos_weight is what puts the raw score off-scale in production; the
    # same distortion is what the calibrator has to undo here.
    _iso, report = calibration.fit_isotonic_oof(
        PARAMS,
        dev,
        ["f0", "f1"],
        num_boost_round=40,
        first_test_year=2016,
        seeds=(1,),
        pos_weight_fn=lambda y: (y == 0).sum() / max((y == 1).sum(), 1),
    )
    assert report.ece_raw > report.ece_calibrated
    assert report.brier_raw > report.brier_calibrated


def test_the_calibrator_never_predicts_a_row_it_was_trained_on():
    """Every calibration point comes from a fold that had not seen it."""
    pytest.importorskip("lightgbm")
    seen = []
    real = calibration.fit_predict_seeds

    def spy(params, train, test, features, **kw):
        seen.append((set(train.anio.unique()), set(test.anio.unique())))
        return real(params, train, test, features, **kw)

    calibration.fit_predict_seeds = spy
    try:
        calibration.fit_isotonic_oof(
            PARAMS, dev_frame(), ["f0", "f1"], num_boost_round=20, first_test_year=2016, seeds=(1,)
        )
    finally:
        calibration.fit_predict_seeds = real

    assert seen
    for train_years, test_years in seen:
        assert not (train_years & test_years)
        assert max(train_years) < min(test_years)


def test_locked_test_years_never_enter_the_calibrator():
    pytest.importorskip("lightgbm")
    dev = dev_frame(years=range(2014, 2026))  # deliberately includes 2024-25
    seen = []
    real = calibration.fit_predict_seeds

    def spy(params, train, test, features, **kw):
        seen.append(max(test.anio))
        return real(params, train, test, features, **kw)

    calibration.fit_predict_seeds = spy
    try:
        calibration.fit_isotonic_oof(
            PARAMS, dev, ["f0", "f1"], num_boost_round=20, first_test_year=2016, seeds=(1,)
        )
    finally:
        calibration.fit_predict_seeds = real

    assert max(seen) == VAL_END


def test_a_panel_with_no_usable_folds_is_an_error_not_an_untrained_calibrator():
    pytest.importorskip("lightgbm")
    with pytest.raises(ValueError, match="no out-of-fold predictions"):
        calibration.fit_isotonic_oof(
            PARAMS,
            dev_frame(years=[2015]),
            ["f0", "f1"],
            num_boost_round=10,
            first_test_year=2016,
            seeds=(1,),
        )


def test_scores_outside_the_fitted_range_clip_rather_than_go_nan():
    """Production sees scores development never produced; NaN there is an outage."""
    from sklearn.isotonic import IsotonicRegression

    fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        [0.2, 0.4, 0.6], [0.0, 0.5, 1.0]
    )
    out = calibration.apply_calibrator(fitted, [0.0, 1.0])
    assert not np.isnan(out).any()
    assert out.tolist() == [0.0, 1.0]


# ------------------------------------------------------------------ the seam


class Booster:
    """A native LightGBM Booster: predict returns the positive probability."""

    def predict(self, X):
        return np.asarray(X.iloc[:, 0], dtype=float)


class Halve:
    def predict(self, p):
        return np.asarray(p, dtype=float) / 2


def bundle(**extra):
    return {"model": Booster(), "features": ["a"], "threshold": 0.3, **extra}


def test_a_bundle_with_a_calibrator_returns_calibrated_scores():
    p = predictor.from_bundle("x", bundle(calibrator=Halve()))
    assert p.predict_one({"a": 0.8}).probability == pytest.approx(0.4)


def test_a_bundle_without_one_is_unchanged():
    p = predictor.from_bundle("x", bundle())
    assert p.calibrator is None
    assert p.predict_one({"a": 0.8}).probability == pytest.approx(0.8)


def test_the_flag_is_decided_on_the_calibrated_score():
    """The threshold is stored on the calibrated scale, so it must be compared there."""
    p = predictor.from_bundle("x", bundle(calibrator=Halve()))  # threshold 0.3
    assert p.predict_one({"a": 0.8}).flag is True  # 0.40 >= 0.30
    assert p.predict_one({"a": 0.4}).flag is False  # 0.20 <  0.30


def test_frame_scoring_calibrates_too():
    p = predictor.from_bundle("x", bundle(calibrator=Halve()))
    out = p.predict_frame(pd.DataFrame({"a": [0.2, 0.6]}))
    assert list(out) == pytest.approx([0.1, 0.3])


def test_calibrated_frame_scores_keep_their_order():
    p = predictor.from_bundle("x", bundle(calibrator=Halve()))
    raw = pd.DataFrame({"a": [0.9, 0.1, 0.5, 0.3]})
    y = np.array([1, 0, 1, 0])
    assert average_precision_score(y, p.predict_frame(raw)) == pytest.approx(
        average_precision_score(y, raw.a.values)
    )
