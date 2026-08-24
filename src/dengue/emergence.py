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
    "CLIMATE_FEATURES",
    "EXTRA_FEATURES",
    "HISTORY_FEATURES",
    "IMPORTATION_FEATURES",
    "NATIONAL_FEATURES",
    "SHIPPED_BLOCKS",
    "SHIPPED_FEATURES",
    "SUSCEPTIBLE_FEATURES",
    "add_base_features",
    "add_climate_features",
    "add_extra_features",
    "add_history_features",
    "add_importation_features",
    "add_national_features",
    "add_susceptible_features",
    "eligible_mask",
    "label_emergence",
    "lead_times",
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


# ---------------------------------------------------------------------- climate
#
# Emergence draws its predictors from `srilanka.COMMON_FEATURES`, which is the
# INTERSECTION of what Brazil and Sri Lanka both measure. That intersection exists
# for the transfer experiment, and emergence does not transfer - so it has been
# paying the cost of a constraint that buys it nothing. Nothing in it looks back
# further than eight weeks, and it discards `tempmax`/`tempmin` as a pair.
#
# This matters more here than anywhere else in the project: finding 3 measured
# environment at +0.089 PR-AUC for emergence against +0.006 for continuation, a 15x
# difference. Environment is where emergence's signal is known to live.
CLIMATE_FEATURES = [
    "precip_anom_z",
    "tempmed_anom_z",
    "umid_anom_z",
    "dtr",
    "dtr_roll4",
    "precip_lag_8",
    "precip_lag_12",
    "precip_lag_16",
    "precip_roll12_sum",
    "precip_roll16_sum",
    "dry_spell_weeks",
    "dry_then_wet",
    "degree_weeks_18",
]

# Aedes aegypti transmission is negligible below roughly this temperature: the
# extrinsic incubation period lengthens past the mosquito's lifespan. Accumulated
# excess over it is a standard degree-day proxy, expressed in weeks here because
# the panel is weekly.
EIP_BASE_TEMP = 18.0

# A (district, week-of-year) cell needs this many earlier years before its anomaly
# is emitted at all, and the result is bounded afterwards. See `_woy_anomaly`.
MIN_ANOMALY_YEARS = 4
ANOMALY_CLIP = 6.0


def _woy_anomaly(df: pd.DataFrame, col: str) -> pd.Series:
    """How unusual this week's value is FOR THIS DISTRICT AT THIS TIME OF YEAR.

    The single most important idea in this block. A raw rainfall column pooled over
    26 districts confounds "a wet district" with "wetter than usual here", and a
    seven-leaf tree cannot separate those two - it has to spend splits discovering
    which districts are wet before it can say anything about departure from normal.
    Emergence is entirely a question about departure from normal.

    Grouped by (district, week-of-year) and shifted, so the reference distribution
    is that district's SAME CALENDAR WEEK in strictly earlier years. Twenty years
    of panel gives about twenty prior observations per cell.

    Two guards, both learned from what the unguarded version produced. A cell with
    one or two prior years has a near-zero sample sd, which turns an ordinary week
    into a z-score of -3842; `MIN_ANOMALY_YEARS` withholds the feature until the
    reference distribution means something, and NaN is the honest value for "not
    enough history yet" - LightGBM splits on it natively. The clip then bounds the
    genuine outliers that survive, because a tree only needs the ordering and an
    unbounded tail wastes split points.
    """
    woy = df.week_start.dt.isocalendar().week.astype(int)
    g = df.assign(_woy=woy).groupby(["district", "_woy"], sort=False)[col]
    mean = g.transform(lambda s: s.shift(1).expanding().mean())
    sd = g.transform(lambda s: s.shift(1).expanding().std())
    seen = g.transform(lambda s: s.shift(1).expanding().count())

    z = (df[col] - mean) / sd.where(sd > 1e-6)
    return z.where(seen >= MIN_ANOMALY_YEARS).clip(-ANOMALY_CLIP, ANOMALY_CLIP)


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mechanistic climate features, all strictly past-or-present.

    Present is allowed and past is required: this week's rainfall is observed at
    the time the forecast is made, so it may be used, but no window may reach
    forward. Every rolling window here is `shift(1)`-ed first, matching
    `srilanka.add_base_features`.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)

    for col, name in (
        ("precip_total_semana", "precip_anom_z"),
        ("tempmed", "tempmed_anom_z"),
        ("umidmed", "umid_anom_z"),
    ):
        df[name] = _woy_anomaly(df, col)

    # Diurnal temperature range. Both inputs were already in the panel and only
    # ever used separately; the SPREAD is the quantity with a vector biology behind
    # it - a large daily swing suppresses transmission at a given mean temperature.
    df["dtr"] = df.tempmax - df.tempmin
    df["dtr_roll4"] = g["dtr"].transform(lambda s: s.shift(1).rolling(4).mean())

    # The breeding-site chain - rain fills containers, larvae mature, adults bite,
    # humans become cases - runs longer than the eight weeks the panel could see.
    for lag in (8, 12, 16):
        df[f"precip_lag_{lag}"] = g["precip_total_semana"].shift(lag)
    for win in (12, 16):
        df[f"precip_roll{win}_sum"] = g["precip_total_semana"].transform(
            lambda s, w=win: s.shift(1).rolling(w).sum()
        )

    # Dry spell then rain. Prolonged dry weather pushes households to store water in
    # open containers, and the rain that ends the spell floods a stock of breeding
    # sites that a steadily wet district never accumulates. The threshold is the
    # district's own past-only lower quartile, so "dry" means dry for here.
    dry_cut = g["precip_total_semana"].transform(
        lambda s: s.shift(1).expanding().quantile(0.25)
    )
    is_dry = (df.precip_total_semana < dry_cut).to_numpy()
    run, counter = [], {}
    for district, dry in zip(df.district.values, is_dry, strict=True):
        n = counter.get(district, 0) + 1 if dry else 0
        counter[district] = n
        run.append(min(n, 52))
    df["dry_spell_weeks"] = run
    df["dry_then_wet"] = df.dry_spell_weeks * df.precip_anom_z.fillna(0.0)

    excess = (df.tempmed - EIP_BASE_TEMP).clip(lower=0.0)
    df["degree_weeks_18"] = df.assign(_e=excess).groupby("district", sort=False)[
        "_e"
    ].transform(lambda s: s.shift(1).rolling(8).sum())
    return df.drop(columns=["_e"], errors="ignore")


# ------------------------------------------------------------------ susceptibles
#
# The mechanism behind multi-annual dengue cycles, and the one the panel had no
# column for. A district that has just burned through its susceptibles cannot
# sustain a new outbreak however favourable the weather is; one that has been quiet
# for years has been accumulating them through births and waning cross-immunity.
# `weeks_since_outbreak` and `hist_outbreak_rate` are shadows of this quantity, not
# substitutes for it - neither knows how BIG the last epidemic was.
SUSCEPTIBLE_FEATURES = [
    "susceptible_frac",
    "susceptible_frac_z",
    "susceptible_gain_since_outbreak",
]

# Reported cases are a small fraction of infections: most dengue is inapparent or
# never reaches a notifying facility. Fixed from the literature at roughly one
# reported case per four infections rather than tuned, because tuning a constant
# that scales the whole feature against the folds is one more knob to overfit.
REPORTING_MULTIPLIER = 4.0

# Half-life of the protection that matters here. Homotypic immunity is lifelong,
# but a district becomes re-susceptible as the circulating serotype turns over,
# which in Sri Lanka has run on a multi-year cycle.
IMMUNITY_HALFLIFE_WEEKS = 156.0


def add_susceptible_features(
    df: pd.DataFrame,
    inc: float = SL_INC,
    multiplier: float = REPORTING_MULTIPLIER,
    halflife: float = IMMUNITY_HALFLIFE_WEEKS,
) -> pd.DataFrame:
    """Reconstruct a susceptible-fraction proxy per district.

    Immunity accumulates with infections and decays geometrically:

        immune_t = immune_{t-1} * decay + infections_t / population
        susceptible_t = 1 - immune_t

    Evaluated on `shift(1)`-ed cases so a district-week never contributes its own
    cases to its own feature - the same past-only discipline as
    `add_history_features`, and the property `test_history_features_never_see_the
    _current_row` pins there.

    This is a proxy and is named like one. It carries no age structure, no serotype
    detail and a reporting multiplier that is certainly wrong in the third digit.
    What it does carry, which nothing else in the panel does, is the SIZE of what
    the district has already been through.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    decay = 0.5 ** (1.0 / halflife)

    infections = (df.casos * multiplier / df.population_total.clip(lower=1)).to_numpy()
    prev_inf = df.assign(_i=infections).groupby("district", sort=False)["_i"].shift(1)
    prev_inf = prev_inf.fillna(0.0).to_numpy()

    immune, state = [], {}
    for district, add in zip(df.district.values, prev_inf, strict=True):
        m = state.get(district, 0.0) * decay + float(add)
        state[district] = min(m, 1.0)
        immune.append(state[district])

    df["susceptible_frac"] = 1.0 - np.asarray(immune)

    g = df.groupby("district", sort=False)
    mean = g["susceptible_frac"].transform(lambda s: s.shift(1).expanding().mean())
    sd = g["susceptible_frac"].transform(lambda s: s.shift(1).expanding().std())
    df["susceptible_frac_z"] = (df.susceptible_frac - mean) / sd.where(sd > 1e-9)

    # How much of the pool has been rebuilt since the district last burned. The
    # level alone cannot say that: two districts at the same susceptible fraction
    # are in very different places if one is climbing and one has always been there.
    above = (df.p_inc100k >= inc).to_numpy()
    gain, last_hot = [], {}
    for district, hot, s in zip(df.district.values, above, df.susceptible_frac.values, strict=True):
        if district in last_hot:
            gain.append(float(s) - last_hot[district])
        else:
            gain.append(np.nan)
        if hot:
            last_hot[district] = float(s)
    df["susceptible_gain_since_outbreak"] = gain
    return df


# ------------------------------------------------------------------ importation
#
# NOT the k-nearest-neighbour block that ADR 0003 rejected. That one averaged the
# four geographically closest districts' incidence, which with 26 districts is a
# quarter of the country and behaves like a national mean with extra steps.
#
# This is a different quantity: an importation pressure that is nonzero only from
# districts that are ACTUALLY IN OUTBREAK, weighted by a gravity term, plus a
# separate term for the Colombo-Gampaha commuter core the epidemic genuinely moves
# along. Four columns rather than twelve, because dilution is how the last attempt
# lost.
IMPORTATION_FEATURES = [
    "import_force",
    "import_force_lag_2",
    "import_force_roll4",
    "hub_force",
]

HUB_DISTRICTS = ("Colombo", "Gampaha")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def add_importation_features(df: pd.DataFrame, inc: float = SL_INC) -> pd.DataFrame:
    """Gravity-weighted force of infection from districts currently in outbreak.

    Uses other districts' incidence at the SAME week, which is observed when the
    forecast is made and is therefore not leakage - the surveillance system knows
    this week's national picture before it forecasts next month's.
    """
    from dengue.srilanka import DISTRICTS

    df = df.sort_values(["district", "week_start"]).copy()
    names = [d for d in DISTRICTS if d in set(df.district.unique())]
    idx = {d: i for i, d in enumerate(names)}

    # w[i, j]: pull that district j exerts on district i. Distance is softened by a
    # 25 km floor so two adjacent centroids cannot produce a near-infinite weight.
    w = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b:
                continue
            d = max(_haversine_km(*DISTRICTS[a][:2], *DISTRICTS[b][:2]), 25.0)
            w[i, j] = DISTRICTS[b][2] / d**2

    inc_wide = df.pivot_table(
        index="week_start", columns="district", values="p_inc100k", aggfunc="first"
    ).reindex(columns=names)
    hot = (inc_wide >= inc).astype(float) * inc_wide.fillna(0.0)

    # Scaled to a comfortable range; the absolute units are meaningless, only the
    # ordering and the relative size across districts and weeks matter.
    force = pd.DataFrame(hot.to_numpy() @ w.T / 1e6, index=inc_wide.index, columns=names)
    hub_cols = [d for d in HUB_DISTRICTS if d in names]
    hub_w = w[:, [idx[d] for d in hub_cols]]
    hub = pd.DataFrame(
        inc_wide[hub_cols].fillna(0.0).to_numpy() @ hub_w.T / 1e6,
        index=inc_wide.index,
        columns=names,
    )

    long = force.melt(ignore_index=False, var_name="district", value_name="import_force")
    long = long.reset_index()
    long["hub_force"] = hub.melt(ignore_index=False, value_name="hub").reset_index().hub.to_numpy()

    df = df.merge(long, on=["week_start", "district"], how="left")
    df = df.sort_values(["district", "week_start"])
    g = df.groupby("district", sort=False)
    df["import_force_lag_2"] = g["import_force"].shift(2)
    df["import_force_roll4"] = g["import_force"].transform(lambda s: s.shift(1).rolling(4).mean())
    return df


# --------------------------------------------------------------------- national
#
# Where the country is in its own epidemic cycle. Every existing feature describes
# one district in isolation, so the model cannot tell a quiet district in a quiet
# year from a quiet district in the month before a national epidemic - which is
# exactly the distinction emergence turns on.
NATIONAL_FEATURES = [
    "national_inc",
    "national_inc_z",
    "frac_districts_in_outbreak",
    "weeks_since_national_peak",
]


def add_national_features(df: pd.DataFrame, inc: float = SL_INC) -> pd.DataFrame:
    """National epidemic phase, identical for every district in a given week."""
    df = df.sort_values(["district", "week_start"]).copy()

    per_week = df.groupby("week_start").apply(
        lambda d: pd.Series(
            {
                "national_inc": d.casos.sum() / max(d.population_total.sum(), 1) * 1e5,
                "frac_districts_in_outbreak": float((d.p_inc100k >= inc).mean()),
            }
        ),
        include_groups=False,
    )
    per_week = per_week.sort_index()

    mean = per_week.national_inc.shift(1).expanding().mean()
    sd = per_week.national_inc.shift(1).expanding().std()
    per_week["national_inc_z"] = (per_week.national_inc - mean) / sd.where(sd > 1e-9)

    # Weeks since the country was last in a national surge, where "surge" is the
    # past-only upper quartile of the national series.
    #
    # The obvious version of this - weeks since the running MAXIMUM - is a trap, and
    # a quiet one. An expanding max is set once and then never beaten, so the
    # counter just climbs forever and the column becomes a proxy for the calendar.
    # Every locked-test row would then sit beyond the largest value any training row
    # ever showed, and a tree cannot extrapolate past its training range: the whole
    # test set collapses into one leaf. A quartile threshold re-arms every few
    # years, so the feature stays inside the range the model was fitted on.
    surge_cut = per_week.national_inc.shift(1).expanding().quantile(0.75)
    in_surge = (per_week.national_inc >= surge_cut).to_numpy()
    since, n = [], 260
    for surge in in_surge:
        n = 0 if surge else n + 1
        since.append(min(n, 260))
    per_week["weeks_since_national_peak"] = since

    return df.merge(per_week.reset_index(), on="week_start", how="left")


# The whole extra block, in the order the search adds them.
EXTRA_FEATURES = CLIMATE_FEATURES + SUSCEPTIBLE_FEATURES + IMPORTATION_FEATURES + NATIONAL_FEATURES

# Which of the blocks above the production model actually uses.
#
# This is deliberately a separate list from EXTRA_FEATURES, and deliberately not
# "all of them". A block is added here only after
# experiments/accuracy_v2/search_emergence_v3.py shows it clearing BOTH intervals -
# paired over folds and clustered over districts. Everything else stays in the
# module as a measured negative result, because the reason ADR 0003 exists is that
# candidate features which merely look promising are how a model gets worse while
# its development score goes up.
SHIPPED_BLOCKS: tuple[str, ...] = ()

_BLOCK_FEATURES = {
    "climate": CLIMATE_FEATURES,
    "susceptible": SUSCEPTIBLE_FEATURES,
    "importation": IMPORTATION_FEATURES,
    "national": NATIONAL_FEATURES,
}

SHIPPED_FEATURES = [c for b in SHIPPED_BLOCKS for c in _BLOCK_FEATURES[b]]


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every new feature block, applied in one call."""
    df = add_climate_features(df)
    df = add_susceptible_features(df)
    df = add_importation_features(df)
    return add_national_features(df)


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

    # How far ahead the warning would be: the FIRST week inside the window at which
    # the district crosses. The label alone cannot say this - it is a max over the
    # whole window, so a crossing at week 1 and a crossing at week 4 are the same
    # `y`. The difference is the entire operational value of an early warning, and
    # nothing in the project measured it.
    crossings = future.to_numpy()
    first = np.where(
        np.isnan(crossings).all(axis=1),
        np.nan,
        np.argmax(np.nan_to_num(crossings) >= 1, axis=1) + 1.0,
    )
    df["weeks_to_crossing"] = np.where(df.y.to_numpy() == 1.0, first, np.nan)

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


def add_regression_target(df: pd.DataFrame, horizon: int = EMERGENCE_HORIZON) -> pd.DataFrame:
    """Peak incidence over the window the binary label thresholds.

    The binary label is where most of this task's information goes to die: 23,053
    eligible district-weeks collapse into 1,090 positives, and a week that reached
    9.8 is scored identically to one that reached 0.2 while a week that hit 40
    counts the same as one that grazed 10.0. This keeps the magnitude.

    Deliberately the SAME window `label_emergence` maxes over, so a model trained on
    this differs from the classifier only in whether the outcome was thresholded -
    anything else would confound the comparison with a change of question.
    """
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    future = pd.concat([g["p_inc100k"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    df["y_max_inc"] = future.max(axis=1)
    df["y_log_inc"] = np.log1p(df.y_max_inc)
    return df


def lead_times(labelled: pd.DataFrame, flagged) -> dict:
    """How much warning the caught outbreaks actually got, in weeks.

    An early-warning system is not judged only on whether it fires but on how much
    time it buys: a flag raised the week before a district crosses is nearly
    worthless operationally, and one raised four weeks out is a vector-control
    campaign. `weeks_to_crossing` carries that distance, and this reduces it over
    the outbreaks the model caught.

    Returns an empty dict when nothing was caught, so a caller can render absence
    rather than a fabricated mean.
    """
    caught = np.asarray(flagged, dtype=bool) & (labelled.y.to_numpy() == 1.0)
    weeks = labelled.weeks_to_crossing.to_numpy()[caught]
    weeks = weeks[~np.isnan(weeks)]
    if weeks.size == 0:
        return {}

    return {
        "n_caught": int(weeks.size),
        "mean_weeks": float(weeks.mean()),
        "median_weeks": float(np.median(weeks)),
        "distribution": {str(int(k)): int((weeks == k).sum()) for k in np.unique(weeks)},
    }
