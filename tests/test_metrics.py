"""Operating points, and the two ways of spending an alert budget.

The rates here are trivial arithmetic; what is worth pinning is that they mean what
the dashboard says they mean. NPV in particular reads as a triumph (0.977) until it
is put beside the 0.935 a model that never flags anything scores, so the tests below
fix the definition rather than the impressiveness.
"""

from __future__ import annotations

import numpy as np
import pytest

from dengue import metrics


def test_operating_point_rates_come_from_the_four_counts():
    op = metrics.OperatingPoint(threshold=0.5, tp=30, fn=10, fp=60, tn=900)
    assert op.recall == pytest.approx(30 / 40)
    assert op.precision == pytest.approx(30 / 90)
    assert op.specificity == pytest.approx(900 / 960)
    assert op.npv == pytest.approx(900 / 910)
    assert op.accuracy == pytest.approx(930 / 1000)
    assert op.balanced_accuracy == pytest.approx((0.75 + 900 / 960) / 2)


def test_a_model_that_never_flags_scores_high_accuracy_and_chance_balanced_accuracy():
    """The trap the emergence model is judged against, stated as a test."""
    op = metrics.OperatingPoint(threshold=1.1, tp=0, fn=65, fp=0, tn=935)
    assert op.accuracy == pytest.approx(0.935)
    assert op.balanced_accuracy == pytest.approx(0.5)
    assert op.recall == 0.0
    # And its NPV is already high, which is why the model's must be read against it.
    assert op.npv == pytest.approx(0.935)


def test_evaluate_threshold_flags_on_greater_or_equal():
    """Must match `Prediction.flag`, or the reported metric is not what ships."""
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.9, 0.1])
    op = metrics.evaluate_threshold(y, p, 0.5)
    assert (op.tp, op.fp, op.fn, op.tn) == (2, 1, 0, 1)


def test_degenerate_counts_do_not_divide_by_zero():
    op = metrics.OperatingPoint(threshold=0.5, tp=0, fn=0, fp=0, tn=0)
    assert op.recall == 0.0
    assert op.precision == 0.0
    assert op.npv == 0.0
    assert op.flagged_per_week(0) == 0.0


# ------------------------------------------------------------------- alert budget


def test_alerts_at_recall_picks_the_cheapest_threshold_meeting_the_target():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0.0, 0.6, 90), np.linspace(0.5, 0.99, 10)])
    op = metrics.alerts_at_recall(y, p, 0.9)
    assert op.recall >= 0.9
    # Any higher threshold would drop below the target, so this is the fewest alerts.
    tighter = metrics.evaluate_threshold(y, p, op.threshold + 1e-9)
    assert tighter.recall < 0.9 or tighter.flagged == op.flagged


def test_top_k_flags_exactly_k_districts_in_every_week():
    weeks = np.repeat([1, 2, 3], 10)
    p = np.tile(np.linspace(0, 1, 10), 3)
    flag = metrics.top_k_by_week(p, weeks, 3)
    for w in (1, 2, 3):
        assert flag[weeks == w].sum() == 3


def test_top_k_picks_the_highest_scorers_within_the_week_not_across_weeks():
    """The whole point: a quiet week still gets its own alerts.

    Under a single global threshold a week where every score is low raises nothing
    and a busy week exhausts the budget. Ranking within the week is what makes a
    fixed team capacity spendable.
    """
    weeks = np.array([1, 1, 1, 2, 2, 2])
    p = np.array([0.01, 0.02, 0.03, 0.90, 0.91, 0.92])
    flag = metrics.top_k_by_week(p, weeks, 1)
    # The best of the quiet week is flagged even though it scores below everything
    # in the busy week.
    assert flag.tolist() == [False, False, True, False, False, True]


def test_top_k_larger_than_the_week_flags_everything_available():
    weeks = np.array([1, 1, 2])
    flag = metrics.top_k_by_week(np.array([0.1, 0.2, 0.3]), weeks, 5)
    assert flag.all()
