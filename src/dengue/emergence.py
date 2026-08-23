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

__all__ = ["add_base_features", "eligible_mask", "label_emergence"]


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
