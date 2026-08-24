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

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from dengue.config import SENSITIVITY_TARGET, THRESHOLD_GRID

__all__ = [
    "OperatingPoint",
    "alerts_at_recall",
    "evaluate_threshold",
    "pos_weight",
    "threshold_for_budget",
    "threshold_for_f1",
    "threshold_for_sensitivity",
    "top_k_by_week",
]


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


# --------------------------------------------------------------- operating points


@dataclass(frozen=True)
class OperatingPoint:
    """One threshold, its full confusion matrix, and every rate derived from it.

    The rates were previously computed ad hoc per script, and only recall,
    precision and accuracy were ever recorded. On a task with 6.5% prevalence that
    is the least informative subset available: accuracy is dominated by the
    negatives, and the two figures a health ministry actually acts on - how often a
    "clear" verdict is right (NPV), and how much of the country gets flagged - were
    not written down anywhere.

    Every rate here is a pure function of the four counts, so a caller can always
    recover any of them, and nothing downstream has to recompute a confusion matrix
    from a JSON file that only kept the ratios.
    """

    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.fp + self.tn

    @property
    def flagged(self) -> int:
        return self.tp + self.fp

    @property
    def recall(self) -> float:
        """Of the outbreaks that emerged, the share the model flagged."""
        return self.tp / self.positives if self.positives else 0.0

    @property
    def precision(self) -> float:
        return self.tp / self.flagged if self.flagged else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / self.negatives if self.negatives else 0.0

    @property
    def npv(self) -> float:
        """Of the districts called clear, the share that stayed clear.

        Quoted for the complication model already; the emergence model computes it
        nowhere despite it being the verdict most districts receive every week.
        Prevalence-inflated by construction - compare it against `trivial_npv`,
        never against 0.5.
        """
        called_clear = self.tn + self.fn
        return self.tn / called_clear if called_clear else 0.0

    @property
    def accuracy(self) -> float:
        total = self.positives + self.negatives
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """The honest counterpart to accuracy when one class is 6.5% of the data.

        A never-flag model scores 93.5% accuracy and 50% balanced accuracy, which
        is the whole point.
        """
        return (self.recall + self.specificity) / 2.0

    def flagged_per_week(self, n_weeks: int) -> float:
        return self.flagged / n_weeks if n_weeks else 0.0

    def as_dict(self, n_weeks: int | None = None) -> dict:
        out = {
            "threshold": float(self.threshold),
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "tn": self.tn,
            "recall": self.recall,
            "precision": self.precision,
            "specificity": self.specificity,
            "npv": self.npv,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
        }
        if n_weeks:
            out["flagged_per_week"] = round(self.flagged_per_week(n_weeks), 2)
        return out


def evaluate_threshold(y, p, threshold: float) -> OperatingPoint:
    """Confusion matrix at one threshold, flagging on `p >= threshold`.

    The `>=` matches `Prediction.flag` in `dengue.predictor`, so what is measured
    here is what the app does.
    """
    y = np.asarray(y).astype(int)
    flag = np.asarray(p) >= threshold
    return OperatingPoint(
        threshold=float(threshold),
        tp=int((flag & (y == 1)).sum()),
        fn=int((~flag & (y == 1)).sum()),
        fp=int((flag & (y == 0)).sum()),
        tn=int((~flag & (y == 0)).sum()),
    )


def alerts_at_recall(y, p, target: float) -> OperatingPoint:
    """The cheapest global threshold that still catches `target` of the outbreaks.

    "Cheapest" because recall is monotone non-increasing in the threshold, so the
    HIGHEST threshold meeting the target is also the one raising the fewest alerts.
    This is the headline operating question for an early-warning system with a
    fixed weekly inspection capacity: not "how accurate is it" but "to catch 90% of
    emerging outbreaks, how much of the country must we visit?"

    Measured, not selected - callers pass held-out labels to report a cost, and
    pass validation labels when they need a threshold to deploy.
    """
    return evaluate_threshold(y, p, threshold_for_sensitivity(y, p, target))


def threshold_for_budget(p, n_weeks: int, per_week: float) -> float:
    """The threshold that raises about `per_week` alerts a week on average.

    The inverse of the usual question. A threshold is a statement about the score;
    an inspection budget is a statement about staff, and the budget is the half that
    is actually fixed. This converts one into the other by taking the appropriate
    upper quantile of the scores.
    """
    p = np.asarray(p, dtype=float)
    wanted = max(round(per_week * max(n_weeks, 1)), 1)
    if wanted >= p.size:
        return float(p.min())
    return float(np.partition(p, -wanted)[-wanted])


def top_k_by_week(p, weeks, k: int) -> np.ndarray:
    """Flag the k highest-scoring districts within each week.

    A single global threshold implicitly assumes the score is comparable ACROSS
    weeks, and in a seasonal disease it is not: in a transmission season nearly
    every district clears any fixed bar, so a whole season's inspection budget is
    spent in a few weeks, while the quiet half of the year raises nothing at all.

    Ranking within the week spends a constant budget instead, which is what a team
    with fixed capacity actually has. It needs no calibration across weeks - only
    the ordering within one - so it is strictly less demanding of the model than
    the equivalent global threshold.

    Ties are broken by position, so exactly k districts are flagged per week
    whenever the week has at least k of them.
    """
    p = np.asarray(p, dtype=float)
    weeks = np.asarray(weeks)
    flag = np.zeros(len(p), dtype=bool)
    for w in np.unique(weeks):
        idx = np.flatnonzero(weeks == w)
        # argsort on the negated score puts the largest first; take the first k.
        flag[idx[np.argsort(-p[idx], kind="stable")[:k]]] = True
    return flag
