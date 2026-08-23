"""Threshold selection, in one place.

Two helpers had been copied across the tree. `threshold_for_sensitivity` existed
byte-identically in `model1_screening` and `ablation`. The F1 tuner existed as an
inline pair of lines in eleven scripts, differing only in whether the quantile grid
had 150 or 200 points and whether a degenerate-label guard was present - so the
same "tuned threshold" meant slightly different things depending on which script
produced it.

Thresholds are always chosen on validation. The test years are locked.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from dengue.config import SENSITIVITY_TARGET, THRESHOLD_GRID


def pos_weight(y) -> float:
    """scale_pos_weight: how many negatives per positive.

    Lives here rather than in `experiment` so the screening and LSTM modules can
    use it without importing the Brazil panel machinery (and duckdb with it). The
    max(..., 1) guard is preserved from the original inline expression: a fold with
    no positives degrades to the negative count rather than dividing by zero.
    """
    y = np.asarray(y)
    return (y == 0).sum() / max((y == 1).sum(), 1)


def threshold_for_sensitivity(y, p, target: float = SENSITIVITY_TARGET) -> float:
    """Highest threshold that still achieves >= target sensitivity.

    Sensitivity is monotonically non-increasing in the threshold, so we walk
    thresholds upward and keep the last one that still meets the target. This
    maximises specificity subject to the sensitivity constraint - the correct
    operating point for a screening tool, where missed cases are the costly error.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    best = 0.0
    for t in np.unique(p):
        if (p >= t)[y == 1].mean() >= target:
            best = float(t)
        else:
            break
    return best


def threshold_for_f1(y, p, grid: int = THRESHOLD_GRID) -> float:
    """Threshold maximising F1, searched over quantiles of the scores.

    Returns 0.5 when the labels are degenerate - a spatial or temporal fold can
    legitimately contain a single class, and several callers previously differed
    on whether they guarded for that.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5

    candidates = np.unique(np.quantile(p, np.linspace(0.5, 0.999, grid)))
    scored = [(f1_score(y, (p >= t).astype(int), zero_division=0), float(t)) for t in candidates]
    return max(scored)[1]
