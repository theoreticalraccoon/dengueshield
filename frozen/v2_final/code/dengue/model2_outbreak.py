"""Model 2 - spatiotemporal dengue outbreak forecasting.

Predicts, for each municipality-week t, whether an OUTBREAK will be underway at
t + HORIZON weeks, using only information available at time t.

Non-negotiables enforced here:
  * Strictly temporal train/val/test split. A random split leaks the future and
    inflates every metric.
  * The target is shifted forward within each municipality; features are never
    taken from the future.
  * PR-AUC and recall are primary. With ~5-10% outbreak weeks, accuracy is
    almost meaningless on its own - predicting "no outbreak" always scores >90%.
  * Trivial baselines (persistence, seasonal climatology) are reported alongside,
    because a model that cannot beat persistence has learned nothing.
"""
from __future__ import annotations
import numpy as np, pandas as pd, duckdb
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
FULL = RAW / "dataset_multimodal_v8.parquet"
BENCH = RAW / "dataset_benchmark_v1.parquet"

HORIZON = 4                 # weeks ahead (~1 month early warning)
OUTBREAK_INC = 100.0        # cases per 100k per week => epidemic level
MIN_POP = 50_000            # municipalities where surveillance counts are stable

STATIC = ["elev_mean","pct_tree_cover","pct_builtup","pct_cropland","pct_water",
          "pct_grassland","population_density","population_total","median_age",
          "sex_ratio","water_supply_pct","sewage_pct","garbage_collection_pct","area_km2"]
DYNAMIC = ["casos","p_inc100k","Rt","tempmin","tempmed","tempmax","umidmin","umidmed","umidmax",
           "precip_total_semana","precip_media_semana","precip_max_semana","dias_lluvia_semana",
           "NDVI_mean","NDVI_std","EVI_mean","LST_Day_mean","LST_Night_mean","nino34","soi",
           "casos_lag_1","casos_lag_2","casos_lag_4","casos_lag_8","Rt_lag_1","Rt_lag_2","Rt_lag_4",
           "precip_total_semana_lag_1","precip_total_semana_lag_4","tempmed_lag_1","tempmed_lag_4",
           "umidmed_lag_1","umidmed_lag_4","casos_roll4_mean","casos_roll8_mean",
           "precip_roll4_sum","precip_roll8_sum","tempmed_roll4_mean","tempmed_roll8_mean"]


def load_panel(path: Path | None = None, min_pop: int = MIN_POP) -> pd.DataFrame:
    """Load the municipality-week panel, restricted to municipalities with stable counts."""
    path = path or (FULL if FULL.exists() else BENCH)
    cols = ["codigo_ibge","municipio","estado","data_iniSE","anio","semana","pop"] + DYNAMIC + STATIC
    con = duckdb.connect()
    avail = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{path.as_posix()}'").fetchall()}
    cols = [c for c in dict.fromkeys(cols) if c in avail]
    df = con.execute(f"""
        SELECT {','.join('"'+c+'"' for c in cols)} FROM '{path.as_posix()}'
        WHERE pop >= {min_pop} ORDER BY codigo_ibge, data_iniSE
    """).df()
    con.close()
    return df


def build_supervised(df: pd.DataFrame, horizon: int = HORIZON,
                     outbreak_inc: float = OUTBREAK_INC) -> pd.DataFrame:
    """Attach the forward-shifted outbreak label and seasonality; drop unusable rows."""
    df = df.sort_values(["codigo_ibge", "data_iniSE"]).copy()
    g = df.groupby("codigo_ibge", sort=False)

    # target: is incidence at t+horizon at or above the epidemic threshold?
    df["y_inc_future"] = g["p_inc100k"].shift(-horizon)
    df["y"] = (df["y_inc_future"] >= outbreak_inc).astype("float")

    # contiguity guard: the t+h row must really be h weeks later, not across a gap
    df["_dt_future"] = g["data_iniSE"].shift(-horizon)
    ok = (df["_dt_future"] - df["data_iniSE"]).dt.days == 7 * horizon
    df.loc[~ok, "y"] = np.nan

    # baselines available at time t
    df["baseline_persistence"] = (df["p_inc100k"] >= outbreak_inc).astype(float)

    # seasonality (dengue is strongly seasonal; give the model the calendar)
    wk = df["data_iniSE"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * wk / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * wk / 52.0)

    # growth dynamics
    df["casos_growth_1"] = df["casos"] / df["casos_lag_1"].clip(lower=1)
    df["casos_growth_4"] = df["casos"] / df["casos_lag_4"].clip(lower=1)
    df["inc_roll4"] = df["casos_roll4_mean"] / df["pop"].clip(lower=1) * 1e5

    return df.dropna(subset=["y"]).drop(columns=["_dt_future"])


def feature_columns(df: pd.DataFrame) -> list[str]:
    extra = ["week_sin","week_cos","casos_growth_1","casos_growth_4","inc_roll4"]
    cols = [c for c in DYNAMIC + STATIC + extra if c in df.columns]
    return cols


def temporal_split(df: pd.DataFrame, train_end=2021, val_end=2023):
    """Chronological split. Test years are strictly after everything the model saw."""
    tr = df[df["anio"] <= train_end]
    va = df[(df["anio"] > train_end) & (df["anio"] <= val_end)]
    te = df[df["anio"] > val_end]
    return tr, va, te


def seasonal_climatology(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Baseline 2: historical outbreak rate for this municipality-week-of-year."""
    tr = train.copy()
    tr["woy"] = tr["data_iniSE"].dt.isocalendar().week.astype(int)
    rate = tr.groupby(["codigo_ibge", "woy"])["y"].mean()
    glob = tr["y"].mean()
    t = target.copy()
    t["woy"] = t["data_iniSE"].dt.isocalendar().week.astype(int)
    idx = pd.MultiIndex.from_arrays([t["codigo_ibge"], t["woy"]])
    return rate.reindex(idx).fillna(glob).to_numpy()
