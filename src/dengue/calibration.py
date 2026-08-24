"""Making the Sri Lanka scores mean what the dashboard says they mean.

Both Sri Lanka models train with `scale_pos_weight` - roughly 14 negatives per
positive for continuation, 23 for emergence. That is the right call for ranking:
without it the boosting objective barely notices the positive class. But it
deliberately reweights the likelihood, so what comes out is a score with the right
ORDER and the wrong SCALE. It is systematically too high.

The app has been honest about it and says so in prose ("Predicted probabilities,
not calibrated ones"), but two things downstream still read the raw number as if it
were a probability: the Low/Moderate/High/Very High bands in
`finalize_srilanka.current_forecast`, and the emergence operating point, which is
picked by walking thresholds until sensitivity hits a target.

Fixing it needs care about WHICH predictions the calibrator is fitted on. Both
finalize scripts refit the production model on train+validation, so a calibrator
fitted on validation-only predictions from the train-only model would map from a
distribution the shipped model does not produce. `fit_isotonic_oof` therefore fits
on rolling-origin OUT-OF-FOLD predictions across the whole development period:
every point is predicted by a model that had not seen it, and the mapping covers
the same span of years the production model is trained on.

Isotonic rather than Platt because the distortion `scale_pos_weight` introduces is
monotone but not sigmoid-shaped, and there are thousands of development rows to
fit it on rather than the couple of hundred where isotonic starts overfitting.

Isotonic regression is monotone non-decreasing, so it never REVERSES a pair. It can
merge distinct scores onto the same step, which turns an ordered pair into a tie, and
ties cost a little PR-AUC and ROC-AUC. So the invariant is one-sided:

    calibrated PR-AUC <= raw PR-AUC, always, by a small margin.

That makes it a usable guard. Calibration cannot manufacture rank quality, so a
calibrated score that ranks BETTER than the raw one is not a better model - it is a
calibrator that saw labels it should not have. Both finalize scripts record the raw
figure next to the calibrated one for exactly this check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from dengue.config import VAL_END
from dengue.validation import SEEDS, ece, fit_predict_seeds, rolling_origin

__all__ = ["CERTAINTY_CAP", "CalibrationReport", "apply_calibrator", "fit_isotonic_oof"]

# How close to 0 and 1 a calibrated forecast is allowed to get. See fit_isotonic_oof.
CERTAINTY_CAP = 0.001


@dataclass(frozen=True)
class CalibrationReport:
    """Before and after, on the out-of-fold predictions the calibrator was fitted on."""

    ece_raw: float
    ece_calibrated: float
    brier_raw: float
    brier_calibrated: float
    pr_auc_raw: float
    pr_auc_calibrated: float
    n: int

    def as_dict(self) -> dict:
        return {
            "ece_raw": self.ece_raw,
            "ece_calibrated": self.ece_calibrated,
            "brier_raw": self.brier_raw,
            "brier_calibrated": self.brier_calibrated,
            "pr_auc_raw": self.pr_auc_raw,
            "pr_auc_calibrated": self.pr_auc_calibrated,
            "n_calibration_rows": self.n,
        }

    def format(self) -> str:
        return (
            f"calibration on {self.n} out-of-fold rows: "
            f"ECE {self.ece_raw:.4f} -> {self.ece_calibrated:.4f}, "
            f"Brier {self.brier_raw:.4f} -> {self.brier_calibrated:.4f} "
            f"(PR-AUC {self.pr_auc_raw:.4f} -> {self.pr_auc_calibrated:.4f}, "
            f"must not rise)"
        )


def fit_isotonic_oof(
    params: dict,
    dev: pd.DataFrame,
    features: list[str],
    *,
    num_boost_round: int,
    first_test_year: int,
    last_test_year: int = VAL_END,
    label_col: str = "y",
    seeds=SEEDS,
    pos_weight_fn=None,
) -> tuple[IsotonicRegression, CalibrationReport]:
    """Isotonic calibrator fitted on rolling-origin out-of-fold predictions.

    `dev` must be the development frame only - training plus validation years. The
    locked test years must not appear in it, and `rolling_origin` will not reach
    them by default in any case.

    `pos_weight_fn(y)` supplies the per-fold class weight, which has to be
    recomputed per fold rather than fixed: the outbreak rate varies year to year,
    and a weight computed once on the full development set would leak the later
    folds' prevalence into the earlier ones.
    """
    oof_p: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for _year, train, test in rolling_origin(
        dev, first_test_year=first_test_year, last_test_year=last_test_year
    ):
        fold_params = dict(params)
        if pos_weight_fn is not None:
            fold_params["scale_pos_weight"] = pos_weight_fn(train[label_col].to_numpy())
        p = fit_predict_seeds(
            fold_params,
            train,
            test,
            features,
            num_boost_round=num_boost_round,
            label_col=label_col,
            seeds=seeds,
        )
        oof_p.append(p)
        oof_y.append(test[label_col].to_numpy())

    if not oof_p:
        raise ValueError("no out-of-fold predictions; check first_test_year against the panel")

    p = np.concatenate(oof_p)
    y = np.concatenate(oof_y)

    # out_of_bounds="clip" so a production score above anything seen in development
    # maps to the top of the fitted range rather than to NaN.
    #
    # The bounds stop short of 0 and 1 deliberately. Isotonic's end steps are
    # unregularised: if every development row in the top bin was an outbreak, the
    # fitted value is exactly 1.0, and the dashboard then prints "100%" - a claim of
    # certainty about the future from a few hundred observations. With ~10k
    # development rows nothing finer than about 1e-4 is resolvable anyway, so
    # CERTAINTY_CAP is a floor and ceiling on what the data can support, not a
    # cosmetic clamp.
    iso = IsotonicRegression(
        out_of_bounds="clip", y_min=CERTAINTY_CAP, y_max=1.0 - CERTAINTY_CAP
    ).fit(p, y)
    q = iso.predict(p)

    report = CalibrationReport(
        ece_raw=ece(y, p),
        ece_calibrated=ece(y, q),
        brier_raw=float(brier_score_loss(y, np.clip(p, 0, 1))),
        brier_calibrated=float(brier_score_loss(y, np.clip(q, 0, 1))),
        pr_auc_raw=float(average_precision_score(y, p)),
        pr_auc_calibrated=float(average_precision_score(y, q)),
        n=len(y),
    )
    return iso, report


def apply_calibrator(calibrator, p):
    """Map raw scores through a calibrator, or pass them through when there is none.

    Bundles written before calibration existed carry no calibrator, and must keep
    behaving exactly as they did - so the absent case is a pass-through, not an
    error.
    """
    if calibrator is None:
        return np.asarray(p, dtype=float)
    return np.clip(np.asarray(calibrator.predict(np.asarray(p, dtype=float))), 0.0, 1.0)
