"""Loaders for every candidate dengue dataset, returning (X, y, meta)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _clin_ratios(
    X: pd.DataFrame,
    plt_c: str,
    wbc_c: str,
    neu: str | None = None,
    lym: str | None = None,
    mpv: str | None = None,
) -> pd.DataFrame:
    """Clinically motivated haematological ratios used in dengue literature."""
    X = X.copy()
    X["PLT_WBC_ratio"] = X[plt_c] / X[wbc_c].clip(lower=1)
    if neu and lym:
        X["NLR"] = X[neu] / X[lym].clip(lower=1)  # neutrophil-lymphocyte ratio
        X["PLR"] = X[plt_c] / X[lym].clip(lower=1)  # platelet-lymphocyte ratio
    if mpv:
        X["MPV_PLT_ratio"] = X[mpv] / (X[plt_c] / 1000).clip(lower=1e-6)
    return X


def load_hematology_1523():
    """Mendeley 6fsrsk3mb8 - Jamalpur, Bangladesh. 19 CBC variables, real clinical data."""
    d = pd.read_csv(RAW / "mendeley_1523.csv").drop_duplicates().reset_index(drop=True)
    y = (d["Result"].str.lower() == "positive").astype(int).values
    X = d.drop(columns=["Result"])
    X["Gender"] = (X["Gender"].str.lower() == "male").astype(int)
    X = _clin_ratios(
        X,
        "Total Platelet Count(/cumm)",
        "Total WBC count(/cumm)",
        "Neutrophils(%)",
        "Lymphocytes(%)",
        "MPV(fl)",
    )
    return (
        X,
        y,
        {
            "name": "hematology_1523",
            "source": "Mendeley 6fsrsk3mb8",
            "task": "dengue vs other febrile illness (CBC only)",
        },
    )


def load_vitals_1003():
    """Mendeley xrsbyjs24t - vital signs + blood parameters."""
    d = (
        pd.read_csv(RAW / "mendeley_clinical.csv")
        .dropna(subset=["Final Output"])
        .reset_index(drop=True)
    )
    y = d["Final Output"].astype(int).values
    X = d.drop(columns=["Final Output"])
    X["Sex"] = X["Sex"].map({"Male": 0, "Female": 1, "Child": 2}).fillna(-1).astype(int)
    return X, y, {"name": "vitals_1003", "source": "Mendeley xrsbyjs24t"}


def load_bd_structured():
    """Mendeley 673swz9tb4 - symptoms + platelet/WBC, no serology."""
    d = pd.read_csv(RAW / "bd_structured_Dengue_clinical_dataset.csv")
    y = (d["Outcome"].str.lower() == "positive").astype(int).values
    X = d.drop(columns=["Outcome", "Id"])
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X["Gender"] = (X["Gender"].str.lower() == "male").astype(int)
    X["Location"] = X["Location"].astype("category").cat.codes
    return X, y, {"name": "bd_structured", "source": "Mendeley 673swz9tb4"}


def load_bd_comprehensive():
    """Mendeley zdtc3n6xv2 - demographics, serology, symptoms, geography."""
    d = pd.read_csv(RAW / "bd_comprehensive_dataset.csv")
    y = d["Outcome"].astype(int).values
    X = d.drop(columns=["Outcome"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X, y, {"name": "bd_comprehensive", "source": "Mendeley zdtc3n6xv2"}


ALL_SCREENING = {
    "hematology_1523": load_hematology_1523,
    "vitals_1003": load_vitals_1003,
    "bd_structured": load_bd_structured,
    "bd_comprehensive": load_bd_comprehensive,
}
