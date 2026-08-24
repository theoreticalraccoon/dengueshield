"""Deciding whether a change actually helped, rather than whether it got lucky.

Every model in this repo is selected on a single validation split. On emergence that
split is 2,191 district-weeks carrying 201 positives, and its PR-AUC moves by 0.024
when nothing changes but the LightGBM seed - because `LGBM_PARAMS` sets `seed` and
leaves `bagging_seed` and `feature_fraction_seed` on their defaults, so bagging and
feature sampling are only incidentally reproducible. A candidate feature that "gains
0.02" on that split has demonstrated nothing.

That is the mechanism by which honest work produces inflated numbers: run enough
variants against one noisy estimate and the winner is whichever variant the noise
favoured. Nothing about it feels like cheating at the time.

So selection here is:

  * over MANY folds, not one - `rolling_origin` re-fits per test year;
  * over MANY seeds, not one - `fit_predict_seeds` averages, and sets all three
    seeds so a run is actually reproducible;
  * PAIRED - `compare` bootstraps the per-fold *difference*. Two configurations
    evaluated on the same folds share those folds' difficulty, and pairing removes
    it. Comparing mean against mean throws that away: at the fold spread these
    panels show (sd ~0.09 continuation, ~0.13 emergence), an unpaired test cannot
    resolve anything smaller than about 0.05, which is larger than every real
    effect measured so far.

The test years stay locked throughout. Folds here end at the validation boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dengue.config import SEED, VAL_END

__all__ = [
    "SEEDS",
    "Comparison",
    "SeedEnsemble",
    "compare",
    "ece",
    "fit_predict_seeds",
    "rolling_origin",
    "train_seed_ensemble",
]

# Three seeds is the working default: it removes most of the single-seed lottery at
# three times the cost. Final decisions between close candidates use more.
SEEDS: tuple[int, ...] = (SEED, SEED + 1, SEED + 2)


def rolling_origin(
    df: pd.DataFrame,
    first_test_year: int,
    last_test_year: int = VAL_END,
    year_col: str = "anio",
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (test_year, train, test): train on everything strictly earlier.

    Expanding-window rather than sliding, which is what a surveillance system
    genuinely has available - last year's data does not stop existing.

    `last_test_year` defaults to the validation boundary, so the locked test years
    are not reachable through this function by accident. Selecting on them is the
    one failure this whole module exists to prevent, and it should take a
    deliberate argument to do it.
    """
    for year in range(first_test_year, last_test_year + 1):
        train = df[df[year_col] <= year - 1]
        test = df[df[year_col] == year]
        # A fold with no positives has an undefined PR-AUC; skip rather than
        # contribute a meaningless number to the mean.
        if len(test) == 0 or len(train) == 0:
            continue
        yield year, train, test


def fit_predict_seeds(
    params: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    num_boost_round: int,
    label_col: str = "y",
    seeds: Sequence[int] = SEEDS,
) -> np.ndarray:
    """Mean prediction over `seeds`, with every source of randomness pinned.

    LightGBM draws bagging and feature sampling from `bagging_seed` and
    `feature_fraction_seed`, NOT from `seed`. Setting only `seed` - as every
    production config in this repo does - leaves both on their defaults, so
    subsample=0.85 and colsample_bytree=0.8 vary run to run in ways the config
    does not describe. Setting all three is what makes a seed sweep measure seed
    sensitivity rather than measure nothing.
    """
    import lightgbm as lgb

    y = train[label_col].to_numpy()
    dataset = lgb.Dataset(train[list(features)], label=y)
    preds = []
    for s in seeds:
        booster = lgb.train(
            {**params, "seed": s, "bagging_seed": s, "feature_fraction_seed": s},
            dataset,
            num_boost_round=num_boost_round,
        )
        preds.append(booster.predict(test[list(features)]))
    return np.mean(preds, axis=0)


class SeedEnsemble:
    """Several boosters differing only by seed, averaged.

    Deployed rather than a single booster because the seed spread is not small: on
    the Sri Lanka continuation test years a single seed scores PR-AUC 0.701 where
    the three-seed average scores 0.725. Shipping one booster means shipping
    whichever draw `SEED` happened to be, and measuring an averaged model in
    development while deploying an unaveraged one measures the wrong thing.

    `predict` is the only method it needs: `predictor.probabilities` treats
    anything without `predict_proba` as a native Booster, and this satisfies that
    interface. Pickles with the bundle like any other estimator.
    """

    def __init__(self, boosters):
        self.boosters = list(boosters)
        if not self.boosters:
            raise ValueError("a seed ensemble needs at least one booster")

    def predict(self, X):
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def feature_importance(self, importance_type="gain"):
        """Summed across members, so it means the same thing as a single booster's."""
        return np.sum([b.feature_importance(importance_type) for b in self.boosters], axis=0)

    def __len__(self):
        return len(self.boosters)


def train_seed_ensemble(
    params: dict,
    train: pd.DataFrame,
    features: Sequence[str],
    *,
    num_boost_round: int,
    label_col: str = "y",
    seeds: Sequence[int] = SEEDS,
) -> SeedEnsemble:
    """Fit one booster per seed, with every randomness source pinned."""
    import lightgbm as lgb

    dataset = lgb.Dataset(train[list(features)], label=train[label_col].to_numpy())
    return SeedEnsemble(
        lgb.train(
            {**params, "seed": s, "bagging_seed": s, "feature_fraction_seed": s},
            dataset,
            num_boost_round=num_boost_round,
        )
        for s in seeds
    )


@dataclass(frozen=True)
class Comparison:
    """A paired per-fold comparison and whether it survives its own error bar."""

    mean_delta: float
    ci_low: float
    ci_high: float
    n_folds: int
    baseline_mean: float
    candidate_mean: float

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero - the ship/do-not-ship decision."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def helps(self) -> bool:
        return self.significant and self.mean_delta > 0.0

    def format(self) -> str:
        verdict = "SHIP" if self.helps else ("REGRESSION" if self.significant else "noise")
        return (
            f"{self.baseline_mean:.4f} -> {self.candidate_mean:.4f}  "
            f"delta {self.mean_delta:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"over {self.n_folds} folds  {verdict}"
        )


def compare(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Comparison:
    """Bootstrap CI on the per-fold difference. Both arrays are the SAME folds.

    Resampling folds (not rows) is the right unit: the question is whether the
    candidate would still win on a differently-drawn set of years, and years are
    what varies. Resampling rows would treat 19,000 correlated district-weeks as
    19,000 independent observations and return an interval far too tight to be
    honest.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs matching folds: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("no folds to compare")

    delta = b - a
    rng = np.random.default_rng(seed)
    # A single fold has no spread to bootstrap; report a degenerate interval rather
    # than a fabricated one.
    if a.size == 1:
        lo = hi = float(delta[0])
    else:
        idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
        means = delta[idx].mean(axis=1)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])

    return Comparison(
        mean_delta=float(delta.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n_folds=int(a.size),
        baseline_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
    )


def ece(y, p, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-count bins.

    Moved here from calibration_and_errors.py, which measured Brazil's calibration
    while nothing measured Sri Lanka's. Equal-count rather than equal-width bins
    because outbreak scores pile up near zero, and equal-width bins would put
    almost every observation in the first bin and then average over near-empty
    ones.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(y.mean() - p.mean()))

    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total, out = len(y), 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum() / total) * abs(y[m].mean() - p[m].mean())
    return float(out)
