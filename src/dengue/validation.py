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
    "cluster_ci",
    "compare",
    "ece",
    "fast_average_precision",
    "fit_predict_seeds",
    "pooled_compare",
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
    """A paired comparison and whether it survives its own error bar.

    `unit` names what was resampled to build the interval, because the two
    comparisons in this module resample different things and the distinction is
    the point: `compare` resamples YEARS (do we still win on a differently-drawn
    set of years?), `pooled_compare` resamples DISTRICTS (do we still win on a
    differently-drawn set of districts?). A change that only survives one of them
    has not been shown to generalise along the other axis.
    """

    mean_delta: float
    ci_low: float
    ci_high: float
    n_folds: int
    baseline_mean: float
    candidate_mean: float
    unit: str = "folds"

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
            f"over {self.n_folds} {self.unit}  {verdict}"
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


def pooled_compare(
    y,
    groups,
    baseline_p,
    candidate_p,
    *,
    metric=None,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Comparison:
    """Compare two arms on POOLED out-of-fold predictions, resampling districts.

    `compare` resamples folds, and with the eight to fourteen folds these panels
    admit, that interval cannot resolve anything below about 0.05 - which is larger
    than every effect that is still on the table. The limit is the number of folds,
    and there is no way to manufacture more years.

    So this is the second axis. Every fold's out-of-fold predictions are pooled into
    one score per arm, and the interval comes from resampling the 26 DISTRICTS with
    replacement. That is a cluster bootstrap: a district's weeks are strongly
    autocorrelated, so a district - not a district-week - is the independent unit,
    and resampling rows would treat 23,000 correlated observations as 23,000
    independent ones and return an interval far too tight to be honest.

    Neither axis subsumes the other. Years and districts are different ways for a
    result to be a fluke, so the ship rule is that BOTH must agree in sign and the
    pooled interval must exclude zero. That is stricter than either test alone, not
    a way around the one that was failing.

    `n_boot` defaults lower than `compare`'s 10,000 because each draw recomputes the
    metric over the whole pooled set rather than averaging fourteen numbers.
    """
    if metric is None:
        metric = fast_average_precision

    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    a = np.asarray(baseline_p, dtype=float)
    b = np.asarray(candidate_p, dtype=float)
    if not (y.shape == groups.shape == a.shape == b.shape):
        raise ValueError(
            f"pooled comparison needs matching lengths: "
            f"y={y.shape} groups={groups.shape} baseline={a.shape} candidate={b.shape}"
        )
    if y.size == 0:
        raise ValueError("no rows to compare")

    base_score = float(metric(y, a))
    cand_score = float(metric(y, b))

    unique = np.unique(groups)
    # Precomputed once: the bootstrap draws groups thousands of times, and
    # recomputing the membership mask each draw dominates the runtime otherwise.
    members = [np.flatnonzero(groups == g) for g in unique]

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in picked])
        ys = y[idx]
        # A draw that happens to contain one class only has no defined PR-AUC.
        # Skip it rather than contribute a fabricated number to the interval.
        if ys.min() == ys.max():
            continue
        deltas.append(float(metric(ys, b[idx])) - float(metric(ys, a[idx])))

    if not deltas:
        raise ValueError("every bootstrap draw was single-class; check the labels")

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return Comparison(
        mean_delta=cand_score - base_score,
        ci_low=float(lo),
        ci_high=float(hi),
        n_folds=len(unique),
        baseline_mean=base_score,
        candidate_mean=cand_score,
        unit="districts",
    )


def fast_average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, without sklearn's per-call validation overhead.

    The cluster bootstraps below evaluate this thousands of times on tens of
    thousands of rows, and sklearn's input checking dominated the runtime - a single
    six-arm search spent thirty of its thirty-five minutes here rather than in
    LightGBM.

    Identical to `average_precision_score` when no two scores are equal, which holds
    for every caller here: these are raw model scores, and the one place ties used
    to appear - isotonic calibration - is exactly what `TieBrokenIsotonic` removed.
    `_check_matches_sklearn` in the tests pins the equivalence.
    """
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    positives = tp[-1]
    return float((precision * y).sum() / positives) if positives else 0.0


def cluster_ci(
    y,
    groups,
    p,
    *,
    metric=None,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """(score, lo, hi) for ONE arm, resampling districts.

    The single-arm counterpart to `pooled_compare`, for reporting rather than
    selecting: a headline figure quoted without an interval invites the reader to
    treat 0.405 as exact when 175 positive events cannot support that.
    """
    if metric is None:
        metric = fast_average_precision

    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    p = np.asarray(p, dtype=float)

    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in picked])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        draws.append(float(metric(ys, p[idx])))

    if not draws:
        raise ValueError("every bootstrap draw was single-class; check the labels")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


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
