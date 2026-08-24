"""El Nino / Southern Oscillation, as a covariate the rest of the panel cannot be.

Every predictor in the Sri Lanka panel is local and short-memory: the longest window
in `srilanka.add_base_features` is eight weeks, and even the climate anomalies added
for emergence look back only as far as the same calendar week in earlier years. ENSO
is the one thing on the table with a genuinely different time constant. It modulates
the monsoon that drives the rainfall that fills the breeding sites, and it does so at
a three-to-six month lead - a horizon nothing else here can see.

docs/adr/0003 recorded it as the one lever that was neither accepted nor rejected:
"+0.009 against a fold sd of 0.13 is not a result". This module exists so it can be
settled rather than re-guessed.

**Publication lag is the trap.** The ONI value for a month is published during the
following month, so a forecast made in month M can only have read the index through
M-1. Wiring the index in by calendar month gives the model a number that did not
exist yet, and because ENSO is smooth and autocorrelated the leak is small, plausible
and entirely invisible in the score. `PUBLICATION_LAG_MONTHS` is applied to every lag
so lag 0 already means "the most recent value actually published".

Source is NOAA CPC's detrended Nino 3.4 series, with NOAA PSL as a fallback; both are
free and unauthenticated. The parsed series is cached to `data/raw/enso_nino34.csv`
and committed, for the same reason the Sri Lanka inputs are: CI has to work on a
clean checkout, and the weekly refresh must not fail because NOAA is unreachable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
ENSO_CACHE = RAW / "enso_nino34.csv"

CPC_URL = (
    "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/"
    "detrend.nino34.ascii.txt"
)
PSL_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"

# The ONI for month M is published during M+1. A forecast made in M has not seen it.
PUBLICATION_LAG_MONTHS = 1

# Months of lead. 0 is "latest published", which is already a month behind; the long
# ones cover the monsoon lead that motivates the feature at all.
ENSO_LAGS = (0, 2, 3, 6, 9, 12)

ENSO_FEATURES = [f"nino34_lag_{lag}" for lag in ENSO_LAGS] + ["nino34_trend_3m"]

__all__ = ["ENSO_FEATURES", "ENSO_LAGS", "add_enso_features", "fetch_nino34", "load_nino34"]


def _parse_cpc(text: str) -> pd.DataFrame:
    """YR MON TOTAL ClimAdjust ANOM, whitespace separated, one header line."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        rows.append({"year": int(parts[0]), "month": int(parts[1]), "anom": float(parts[4])})
    return pd.DataFrame(rows)


def _parse_psl(text: str) -> pd.DataFrame:
    """Year followed by twelve monthly values; a trailing block of notes to skip."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for month, value in enumerate(parts[1:], start=1):
            v = float(value)
            # Both PSL products use a large negative sentinel for "no observation".
            if v < -90:
                continue
            rows.append({"year": year, "month": month, "anom": v})
    return pd.DataFrame(rows)


def fetch_nino34(force: bool = False) -> pd.DataFrame:
    """Monthly Nino 3.4 anomaly, cached. Returns columns year, month, anom."""
    if ENSO_CACHE.exists() and not force:
        return pd.read_csv(ENSO_CACHE)

    for url, parse in ((CPC_URL, _parse_cpc), (PSL_URL, _parse_psl)):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = parse(r.text)
        except Exception as exc:  # any failure falls through to the next endpoint
            print(f"    enso: {url} failed ({exc})", flush=True)
            continue
        if len(df) > 240:  # a usable series is decades long, not a stub
            df = df.sort_values(["year", "month"]).reset_index(drop=True)
            ENSO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(ENSO_CACHE, index=False)
            print(f"    enso: {len(df)} months from {url.split('/')[2]}", flush=True)
            return df

    raise RuntimeError("could not obtain a Nino 3.4 series from either NOAA endpoint")


def load_nino34() -> pd.DataFrame | None:
    """The cached series, or None when it has never been fetched.

    Absent is a legitimate state - a checkout without the cache should degrade to a
    model without ENSO rather than fail - so callers get None and decide.
    """
    return pd.read_csv(ENSO_CACHE) if ENSO_CACHE.exists() else None


def add_enso_features(df: pd.DataFrame, enso: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach lagged Nino 3.4, respecting when each value was actually published.

    Every lag is measured from the latest PUBLISHED month, not the current one, so
    no row can read an index that did not exist when its forecast was made.
    """
    enso = load_nino34() if enso is None else enso
    df = df.copy()
    if enso is None or enso.empty:
        for col in ENSO_FEATURES:
            df[col] = np.nan
        return df

    # A month ordinal makes lagging arithmetic rather than calendar-aware.
    series = pd.Series(
        enso.anom.to_numpy(dtype=float),
        index=(enso.year.to_numpy() * 12 + enso.month.to_numpy()),
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    month_key = df.week_start.dt.year * 12 + df.week_start.dt.month
    published = month_key - PUBLICATION_LAG_MONTHS

    for lag in ENSO_LAGS:
        df[f"nino34_lag_{lag}"] = published.sub(lag).map(series)

    # Direction matters as well as level: a warming Pacific and a cooling one at the
    # same anomaly are different states of the system.
    df["nino34_trend_3m"] = df["nino34_lag_0"] - df["nino34_lag_3"]
    return df
