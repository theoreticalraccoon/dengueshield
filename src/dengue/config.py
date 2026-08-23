"""Paths and experiment constants, in one place.

Every one of these was previously re-declared per script - `SEED = 42` in 16 files,
the 2021/2023 split boundary across 37 lines, the data root as
`Path(__file__).resolve().parents[2]` in six modules. That scatter is not cosmetic:
`HORIZON` drifted to two different values, and `models/model2_outbreak.joblib` was
saved at a horizon no published report describes.

Constants that genuinely differ between the two surveillance systems carry a country
prefix. Brazil and Sri Lanka do NOT share an epidemic threshold - 100 cases per 100k
per week is an epidemic in Brazil and would be unheard of in Sri Lanka, where the
calibrated threshold is 9.9 - so a single `OUTBREAK_INC` would be a bug waiting to
happen.
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------ paths
# parents[2] is the repo root: src/dengue/config.py -> src/dengue -> src -> .
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# ------------------------------------------------------------------ protocol
SEED = 42

# Strictly temporal. Train on everything up to TRAIN_END, tune on the years after
# it up to VAL_END, and never touch what follows until the result is final.
TRAIN_END = 2021
VAL_END = 2023

# Thresholds are chosen on validation only. The test years are locked.
SENSITIVITY_TARGET = 0.90  # clinical models are tuned for sensitivity, not F1
THRESHOLD_GRID = 200  # quantile points searched when maximising F1

# ------------------------------------------------------------------ Brazil
BR_HORIZON = 2  # weeks ahead - the flagship 14-day early warning
BR_INC = 100.0  # cases per 100k per week => epidemic level
BR_MIN_POP = 50_000  # municipalities where surveillance counts are stable

# ------------------------------------------------------------------ Sri Lanka
SL_HORIZON = 2  # continuation: will an existing outbreak persist 14 days?
SL_INC = 9.9  # Sri Lanka-calibrated epidemic threshold (per 100k/week)

EMERGENCE_HORIZON = 4  # emergence asks a wider question: 1-4 weeks out
QUIET_WEEKS = 2  # consecutive weeks below threshold before a district is eligible

# ------------------------------------------------------------------ estimator
# The shared LightGBM configuration. Scripts that deliberately vary it should say
# so at the call site rather than copying the whole dict.
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
    "seed": SEED,
}

# Sri Lanka has ~26 districts against Brazil's thousands of municipalities, so the
# trees have to be smaller or they memorise the panel.
LGBM_PARAMS_SL = {**LGBM_PARAMS, "num_leaves": 63, "reg_lambda": 5.0}
