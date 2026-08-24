"""Emergence: will a NEW outbreak begin in a district that is currently quiet?

The counterpart to continuation, and deliberately a different question asked of a
different population. Continuation is asked of districts already at or above the
epidemic threshold; emergence is asked only of districts that are below it AND
have been below it for `QUIET_WEEKS` consecutive weeks.

That eligibility rule is the whole point. A district that dropped below the
threshold last week is still riding its own outbreak down, so scoring it for
"emergence" would mostly re-detect the outbreak it just had. Excluding it is what
makes this the harder question - and why the model scores PR-AUC ~0.58 here
against ~0.96 for continuation.

Districts the rule excludes are not scored at all. The forecast table carries NaN
for them, meaning "the question was not asked", never "low risk" - see
dengue.risk and docs/adr/0001-blank-emergence-risk.md.

This lived in experiments/emergence_v1/ while the app, the weekly refresh and CI
all depended on it. It is production; it now sits with the rest of the package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dengue.config import EMERGENCE_HORIZON, QUIET_WEEKS, SL_INC
from dengue.srilanka import add_base_features

__all__ = [
    "HISTORY_FEATURES",
    "add_base_features",
    "add_history_features",
    "eligible_mask",
    "label_emergence",
]

# A district's own epidemic cycle. Every base feature describes the last few
# weeks; none of them knows whether this district has outbreaks every year or
# once a decade, or how long it has been quiet. Selected by rolling-origin CV
# over seven folds ending 2023 - the locked test years were not consulted.
HISTORY_FEATURES = [
    "weeks_since_outbreak",
    "hist_outbreak_rate",
    "hist_inc_mean",
    "inc_vs_history",
    "inc_lag_52",
    "inc_lag_26",
]


def add_history_features(df: pd.DataFrame, inc: float = SL_INC) -> pd.DataFrame:
    """Long-memory features: how often this district burns, and how long since.

    Every statistic is computed from strictly earlier weeks - the expanding
    windows are shifted, so a row never contributes to its own feature. Verified
    with the project's shuffled-label control: PR-AUC 0.068 against a 0.065
    chance rate, i.e. no future information.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    above = (df.p_inc100k >= inc).astype(float)
    g = df.assign(_above=above).groupby("district", sort=False)

    # Weeks since the district was last at or above threshold, counted from the
    # PREVIOUS week so the current observation cannot inform its own feature.
    prev = g["_above"].shift(1).fillna(0.0)
    counter: dict[str, int] = {}
    since = []
    for district, hot in zip(df.district.values, prev.values, strict=True):
        n = 0 if hot == 1 else counter.get(district, 99) + 1
        counter[district] = n
        since.append(min(n, 200))
    df["weeks_since_outbreak"] = since

    df["hist_outbreak_rate"] = g["_above"].transform(lambda s: s.shift(1).expanding().mean())
    df["hist_inc_mean"] = g["p_inc100k"].transform(lambda s: s.shift(1).expanding().mean())
    df["inc_vs_history"] = df.p_inc100k / df.hist_inc_mean.clip(lower=0.01)
    df["inc_lag_52"] = g["p_inc100k"].shift(52)
    df["inc_lag_26"] = g["p_inc100k"].shift(26)
    return df


def eligible_mask(df: pd.DataFrame, inc: float = SL_INC, quiet: int = QUIET_WEEKS) -> pd.Series:
    """Districts below threshold now and for the previous `quiet` weeks."""
    above = (df.p_inc100k >= inc).astype(float)
    g = df.assign(_above=above).groupby("district", sort=False)
    mask = above == 0
    for k in range(1, quiet + 1):
        mask &= g["_above"].shift(k) == 0
    return mask


def label_emergence(
    df: pd.DataFrame,
    horizon: int = EMERGENCE_HORIZON,
    inc: float = SL_INC,
    quiet: int = QUIET_WEEKS,
    keep_ineligible: bool = False,
) -> pd.DataFrame:
    """Label each district-week: does incidence cross `inc` within `horizon` weeks?

    Unlike the continuation label, which reads incidence at exactly one future
    week, this takes the maximum over the whole window t+1 .. t+horizon - an
    outbreak that begins and is caught early still counts.

    `keep_ineligible=True` keeps every row and marks eligibility in a column,
    which is what scoring needs: the forecast table must carry a row per district
    even when the question does not apply to it.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    df["_above"] = (df.p_inc100k >= inc).astype(float)

    future = pd.concat([g["_above"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y"] = (future.max(axis=1) >= 1).astype(float)
    df.loc[future.isna().any(axis=1), "y"] = np.nan

    # Date-exactness guard: a gap in the series must not be read as a quiet week.
    week_then = g["week_start"].shift(-horizon)
    df.loc[(week_then - df.week_start).dt.days != 7 * horizon, "y"] = np.nan

    eligible = df._above == 0
    for k in range(1, quiet + 1):
        eligible &= g["_above"].shift(k) == 0
    df["eligible"] = eligible

    if keep_ineligible:
        return df
    return df[eligible].dropna(subset=["y"]).reset_index(drop=True)
