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

# Emergence operating point. Accuracy is the wrong dial here: predicting "no new
# outbreak" forever scores 93.5% and catches nothing. What matters is how many
# emerging outbreaks are caught, and the price in false alarms. Measured on the
# locked 2024-25 test set, over 26 districts:
#
#   target   recall   precision   districts flagged per week
#   0.50     54%      35%         2.2
#   0.60     66%      25%         3.7
#   0.70     73%      20%         5.3      <- default
#   0.80     83%      14%         8.5
#   0.90     94%       8%        17.3      (two thirds of the country, every week)
#
# These are read off the CALIBRATED score, so the thresholds in
# reports/srilanka_emergence.json are now probabilities (0.16 at the default target)
# rather than the raw ~0.65 they used to be. The recalls are unchanged to within a
# point; what moved is the scale they are read from.
#
# The threshold itself is always chosen on validation; only the target lives here.
EMERGENCE_SENSITIVITY_TARGET = 0.70

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
#
# This was 63, which was the right reasoning and not nearly enough of it. Nothing had
# ever searched it. Over 8 rolling-origin folds with a paired bootstrap on the
# per-fold difference (experiments/accuracy_v2/search_continuation.py), every
# lower-capacity configuration beat 63 and none of the higher ones did:
#
#   num_leaves=31   +0.0023  [-0.0008, +0.0058]   (not resolvable)
#   num_leaves=15   +0.0108  [+0.0056, +0.0160]
#   num_leaves=7    +0.0160  [+0.0079, +0.0235]
#   1000 rounds     -0.0076  [-0.0094, -0.0057]   (a regression)
#
# 15 leaves at 250 rounds is the best of them (+0.0191, continuation CV PR-AUC
# 0.7911 -> 0.8101). It is not distinguishable from 7 leaves at 500; what IS
# distinguishable is that 63 was too many. See docs/adr/0004-sri-lanka-capacity.md.
LGBM_PARAMS_SL = {**LGBM_PARAMS, "num_leaves": 15, "reg_lambda": 5.0}
