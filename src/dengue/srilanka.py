"""Sri Lanka district-week dengue panel + Brazil -> Sri Lanka transfer support.

The Sri Lankan surveillance series (denguedatahub, 2007-2026) is district x week,
which matches the Brazilian municipality x week resolution, so a genuine transfer
experiment is possible. Environmental covariates that Brazil gets from MODIS and
land-cover products do not exist here, so transfer uses the INTERSECTION of features
available in both countries - never imputed stand-ins for missing satellite layers.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

# district -> (lat, lon, population [2012 census], area_km2)
DISTRICTS = {
    "Colombo":      (6.9271, 79.8612, 2324349,  699),
    "Gampaha":      (7.0840, 80.0098, 2304833, 1387),
    "Kalutara":     (6.5854, 80.1289, 1221948, 1598),
    "Kandy":        (7.2906, 80.6337, 1375382, 1940),
    "Matale":       (7.4675, 80.6234,  484531, 1993),
    "NuwaraEliya":  (6.9497, 80.7891,  711644, 1741),
    "Galle":        (6.0535, 80.2210, 1063334, 1652),
    "Hambanthota":  (6.1429, 81.1212,  599903, 2609),
    "Matara":       (5.9549, 80.5550,  814048, 1283),
    "Jaffna":       (9.6615, 80.0255,  583882, 1025),
    "Kilinochchi":  (9.3803, 80.3770,  113510, 1279),
    "Mannar":       (8.9810, 79.9044,   99570, 1996),
    "Vavuniya":     (8.7514, 80.4971,  172115, 1967),
    "Mullaitivu":   (9.2670, 80.8142,   92238, 2617),
    "Batticaloa":   (7.7102, 81.6924,  526567, 2854),
    "Ampara":       (7.2917, 81.6747,  649402, 4415),
    "Trincomalee":  (8.5874, 81.2152,  379541, 2727),
    "Kurunegala":   (7.4863, 80.3647, 1618465, 4816),
    "Puttalam":     (8.0362, 79.8283,  762396, 3072),
    "Anuradhapura": (8.3114, 80.4037,  860575, 7179),
    "Polonnaruwa":  (7.9403, 81.0188,  406088, 3293),
    "Badulla":      (6.9934, 81.0550,  815405, 2861),
    "Monaragala":   (6.8728, 81.3509,  451058, 5639),
    "Ratnapura":    (6.6828, 80.3992, 1088007, 3275),
    "Kegalle":      (7.2513, 80.3464,  840648, 1693),
    "Kalmune":      (7.4098, 81.8203,  106780,  100),   # Kalmunai MOH area
}

WEATHER_CACHE = RAW / "srilanka_weather_daily.parquet"

DAILY_VARS = [
    "precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
    "temperature_2m_min", "relative_humidity_2m_mean",
    "relative_humidity_2m_max", "relative_humidity_2m_min",
]


def _from_nasa_power(name, lat, lon, start, end) -> pd.DataFrame:
    """Fallback source: NASA POWER daily reanalysis (free, no key, no burst limit).

    Open-Meteo throttles aggressively for 20-year multi-variable pulls. POWER
    exposes the same underlying quantities, so the columns are renamed to match.
    """
    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=300,
                     params={"parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M",
                             "community": "AG", "latitude": lat, "longitude": lon,
                             "start": start.replace("-", ""), "end": end.replace("-", ""),
                             "format": "JSON"})
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    d = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    d = d.replace(-999.0, np.nan).sort_index()
    out = pd.DataFrame({
        "time": d.index,
        "precipitation_sum": d["PRECTOTCORR"].values,
        "temperature_2m_mean": d["T2M"].values,
        "temperature_2m_max": d["T2M_MAX"].values,
        "temperature_2m_min": d["T2M_MIN"].values,
        "relative_humidity_2m_mean": d["RH2M"].values,
    })
    # POWER exposes only mean RH; use it for all three so downstream columns exist
    out["relative_humidity_2m_max"] = out["relative_humidity_2m_mean"]
    out["relative_humidity_2m_min"] = out["relative_humidity_2m_mean"]
    out["district"] = name
    out["source"] = "nasa_power"
    return out


# Every district must come from the SAME reanalysis product. Mixing Open-Meteo and
# NASA POWER across districts would inject a source effect that the model could
# mistake for a real geographic signal, so one source is used throughout.
PREFER_NASA_POWER = True


def _fetch_one(name, lat, lon, start, end, max_tries=4) -> pd.DataFrame:
    """One district, from a single consistent reanalysis source."""
    cache = RAW / "sl_weather_parts" / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if PREFER_NASA_POWER:
        d = _from_nasa_power(name, lat, lon, start, end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
        return d
    delay = 5.0
    for _ in range(max_tries):
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive", timeout=180,
            params={"latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "daily": ",".join(DAILY_VARS), "timezone": "Asia/Colombo"})
        if r.status_code == 200:
            d = pd.DataFrame(r.json()["daily"])
            d["district"] = name
            d["source"] = "open_meteo"
            cache.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(cache, index=False)
            return d
        if r.status_code in (429, 500, 502, 503):
            time.sleep(delay)
            delay = min(delay * 1.6, 40)
            continue
        r.raise_for_status()
    print(f"      {name}: Open-Meteo throttled, using NASA POWER", flush=True)
    d = _from_nasa_power(name, lat, lon, start, end)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache, index=False)
    return d


def fetch_weather(start="2006-01-01", end="2026-08-01", pause=3.0) -> pd.DataFrame:
    """Daily ERA5 reanalysis per district centroid via Open-Meteo (free, no key).

    Each district is cached separately so an interrupted or rate-limited run
    resumes instead of refetching everything.
    """
    if WEATHER_CACHE.exists():
        return pd.read_parquet(WEATHER_CACHE)
    frames = []
    for name, (lat, lon, _, _) in DISTRICTS.items():
        cached = (RAW / "sl_weather_parts" / f"{name}.parquet").exists()
        d = _fetch_one(name, lat, lon, start, end)
        frames.append(d)
        print(f"    weather {name:14} {len(d)} days{' (cached)' if cached else ''}", flush=True)
        if not cached:
            time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"])
    WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(WEATHER_CACHE, index=False)
    return out


def load_cases() -> pd.DataFrame:
    d = pd.read_csv(RAW / "srilanka_weekly_district.csv")
    d["start_date"] = pd.to_datetime(d["start.date"], format="%m/%d/%Y", errors="coerce")
    d = d.dropna(subset=["start_date"])
    d["cases"] = pd.to_numeric(d["cases"], errors="coerce").fillna(0)
    # snap to ISO Monday so weekly joins line up across districts
    d["week_start"] = d["start_date"] - pd.to_timedelta(d["start_date"].dt.weekday, unit="D")
    return (d.groupby(["district", "week_start"], as_index=False)
              .agg(casos=("cases", "sum")))


def _rain_days(s):
    return int((s >= 1).sum())


def build_panel() -> pd.DataFrame:
    """District x week panel with weather aggregated onto the same weekly grid."""
    cases = load_cases()
    w = fetch_weather().copy()
    w["week_start"] = w["time"] - pd.to_timedelta(w["time"].dt.weekday, unit="D")
    wk = (w.groupby(["district", "week_start"])
            .agg(precip_total_semana=("precipitation_sum", "sum"),
                 precip_media_semana=("precipitation_sum", "mean"),
                 precip_max_semana=("precipitation_sum", "max"),
                 dias_lluvia_semana=("precipitation_sum", _rain_days),
                 tempmed=("temperature_2m_mean", "mean"),
                 tempmax=("temperature_2m_max", "max"),
                 tempmin=("temperature_2m_min", "min"),
                 umidmed=("relative_humidity_2m_mean", "mean"),
                 umidmax=("relative_humidity_2m_max", "max"),
                 umidmin=("relative_humidity_2m_min", "min"))
            .reset_index())
    df = cases.merge(wk, on=["district", "week_start"], how="inner")
    meta = pd.DataFrame([{"district": k, "population_total": v[2], "area_km2": v[3]}
                         for k, v in DISTRICTS.items()])
    df = df.merge(meta, on="district", how="left")
    df["population_density"] = df.population_total / df.area_km2
    df["p_inc100k"] = df.casos / df.population_total * 1e5
    df["anio"] = df.week_start.dt.year
    return df.sort_values(["district", "week_start"]).reset_index(drop=True)


def add_features(df: pd.DataFrame, horizon: int = 2,
                 outbreak_inc: float = 100.0) -> pd.DataFrame:
    """Lags, rolls and the forward-shifted outbreak label - mirrors the Brazil pipeline."""
    df = df.sort_values(["district", "week_start"]).copy()
    g = df.groupby("district", sort=False)
    for L in (1, 2, 4, 8):
        df[f"casos_lag_{L}"] = g["casos"].shift(L)
    df["casos_roll4_mean"] = g["casos"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["casos_roll8_mean"] = g["casos"].transform(lambda s: s.shift(1).rolling(8).mean())
    df["casos_growth_1"] = df.casos / df.casos_lag_1.clip(lower=1)
    df["casos_growth_4"] = df.casos / df.casos_lag_4.clip(lower=1)
    df["inc_roll4"] = df.casos_roll4_mean / df.population_total.clip(lower=1) * 1e5
    for c in ("precip_total_semana", "tempmed", "umidmed"):
        df[f"{c}_lag_1"] = g[c].shift(1)
        df[f"{c}_lag_4"] = g[c].shift(4)
    df["precip_roll4_sum"] = g["precip_total_semana"].transform(
        lambda s: s.shift(1).rolling(4).sum())
    df["precip_roll8_sum"] = g["precip_total_semana"].transform(
        lambda s: s.shift(1).rolling(8).sum())
    df["tempmed_roll4_mean"] = g["tempmed"].transform(
        lambda s: s.shift(1).rolling(4).mean())
    df["tempmed_roll8_mean"] = g["tempmed"].transform(
        lambda s: s.shift(1).rolling(8).mean())
    wk = df.week_start.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * wk / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * wk / 52.0)

    df["y_inc_future"] = g["p_inc100k"].shift(-horizon)
    df["y"] = (df.y_inc_future >= outbreak_inc).astype(float)
    dtf = g["week_start"].shift(-horizon)
    df.loc[(dtf - df.week_start).dt.days != 7 * horizon, "y"] = np.nan
    df["baseline_persistence"] = (df.p_inc100k >= outbreak_inc).astype(float)
    return df.dropna(subset=["y"]).reset_index(drop=True)


# features present in BOTH countries - the only honest basis for transfer
COMMON_FEATURES = [
    "casos", "p_inc100k", "casos_lag_1", "casos_lag_2", "casos_lag_4", "casos_lag_8",
    "casos_roll4_mean", "casos_roll8_mean", "casos_growth_1", "casos_growth_4", "inc_roll4",
    "precip_total_semana", "precip_media_semana", "precip_max_semana", "dias_lluvia_semana",
    "precip_total_semana_lag_1", "precip_total_semana_lag_4",
    "precip_roll4_sum", "precip_roll8_sum",
    "tempmed", "tempmax", "tempmin", "tempmed_lag_1", "tempmed_lag_4",
    "tempmed_roll4_mean", "tempmed_roll8_mean",
    "umidmed", "umidmax", "umidmin", "umidmed_lag_1", "umidmed_lag_4",
    "population_total", "population_density", "week_sin", "week_cos",
]
