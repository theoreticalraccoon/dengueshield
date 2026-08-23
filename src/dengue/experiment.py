"""One configured Brazil run, so six analysis scripts stop rebuilding it.

Every one of `shap_model2`, `spatial_and_ablation`, `calibration_and_errors`,
`leakage_audit`, `verify_fixes` and `robustness_model2` opened with the same four
statements:

    sup   = build_supervised(load_panel(), horizon=..., outbreak_inc=...)
    feats = feature_columns(sup)
    tr, va, te = <the 2021/2023 split, re-inlined>
    spw   = (tr.y.values == 0).sum() / max((tr.y.values == 1).sum(), 1)

and then each carried its own copy of the LightGBM parameters. The split alone
appeared across ~37 lines, the class-imbalance weight in 16 places, the seed in 16.
`temporal_split` already existed in model2_outbreak and only three of thirteen
callers used it.

That scatter is not stylistic. `HORIZON` diverged to two different values and the
saved model ended up describing a different experiment from the published reports
(docs/adr/0002-brazil-horizon.md). A `Run` is the whole configuration in one
object, so a script states only what it varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from dengue.config import BR_HORIZON, BR_INC, LGBM_PARAMS, SEED, TRAIN_END, VAL_END
from dengue.metrics import pos_weight as pos_weight  # re-exported for callers here
from dengue.model2_outbreak import build_supervised, feature_columns, load_panel


def split(df: pd.DataFrame, train_end: int = TRAIN_END, val_end: int = VAL_END):
    """Strictly temporal train / validation / test.

    A random split leaks the future and inflates every metric. Thresholds are
    chosen on validation; the test years stay locked.
    """
    return (
        df[df.anio <= train_end],
        df[(df.anio > train_end) & (df.anio <= val_end)],
        df[df.anio > val_end],
    )


@dataclass(frozen=True)
class Run:
    """A configured experiment: the data, the split, and the fit settings."""

    sup: pd.DataFrame
    feats: list[str]
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame
    horizon: int
    outbreak_inc: float
    params: dict = field(default_factory=dict)

    @property
    def spw(self) -> float:
        """Class-imbalance weight from the training years only."""
        return pos_weight(self.tr.y.values)

    def fit_params(self, **overrides) -> dict:
        """LightGBM parameters with the imbalance weight already applied."""
        return {**self.params, "scale_pos_weight": self.spw, **overrides}

    def describe(self) -> str:
        return (
            f"horizon={self.horizon}w  outbreak>={self.outbreak_inc}/100k  "
            f"train={len(self.tr):,} val={len(self.va):,} test={len(self.te):,}  "
            f"features={len(self.feats)}"
        )


def setup(
    panel: pd.DataFrame | None = None,
    horizon: int = BR_HORIZON,
    outbreak_inc: float = BR_INC,
    seed: int = SEED,
) -> Run:
    """Build the supervised frame and split it. Pass `panel` to reuse one you hold.

    Loading two full Brazil panels in one process exhausts memory and the process
    dies with exit code 0, printing nothing - so a script that needs several
    configurations should load the panel once and pass it in.
    """
    if panel is None:
        panel = load_panel()

    sup = build_supervised(panel, horizon=horizon, outbreak_inc=outbreak_inc)
    tr, va, te = split(sup)
    return Run(
        sup=sup,
        feats=feature_columns(sup),
        tr=tr,
        va=va,
        te=te,
        horizon=horizon,
        outbreak_inc=outbreak_inc,
        params={**LGBM_PARAMS, "seed": seed},
    )
