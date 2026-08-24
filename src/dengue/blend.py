"""Two models of the same outbreak, averaged: the one change that cleared both bars.

Emergence has resisted nine documented attempts to improve it - spatial neighbours,
capacity re-tuning, extra context rows, a discrete-time hazard, Brazil fine-tuning
(docs/adr/0003), and then four fresh feature blocks and three reformulations
(experiments/accuracy_v2/search_emergence_v3.py and _v4.py). This is what worked, and
it is not a feature.

The two members answer the same question from opposite ends:

  the CLASSIFIER   asks "does incidence cross 9.9 within four weeks?" - the shipped
                   formulation, trained on 1,090 positives out of 23,053 rows;
  the REGRESSION   asks "how high does incidence get in those four weeks?" - the same
                   window, the same features, but never thresholded, so every one of
                   the 23,053 rows carries a target instead of one row in twenty-one.

The classifier sees where the boundary is and almost nothing about magnitude. The
regression sees magnitude everywhere and nothing about where the boundary matters.
Averaging them beat either alone by more than either beat production:

                              per fold (14)              per district (26)
  regression alone      +0.0120 [-0.0213, +0.0491]   +0.0175 [-0.0096, +0.0475]
  blend                 +0.0337 [+0.0067, +0.0671]   +0.0237 [+0.0050, +0.0462]

ADR 0003 rejected ensembling for screening, and that is not in tension with this.
There the members were one strong learner and two weak ones, and averaging dragged
the strong one down. Here they are two comparably good models of the same phenomenon
that make DIFFERENT errors, which is the case where averaging is supposed to work.

**The bridge is the part to be careful with.** The two members emit incompatible
scales - a `scale_pos_weight`-distorted probability and a predicted log incidence -
so they cannot simply be added. The first attempt rank-averaged them within each
fold, which reads fine per fold and destroys the pooled comparison: fourteen folds
each rank-normalised to span 0 to 1, so a quiet year and an epidemic year come out
looking identical and the pooled PR-AUC collapses for a reason that has nothing to do
with the model. It scored +0.0272 per fold and -0.0669 pooled, and taking either
number at face value would have been wrong.

So the regression is mapped through an isotonic fitted on the TRAINING rows only.
That is a scale bridge, not a calibration: being monotone it cannot change any
per-fold ranking metric at all, and its only job is to make one year's scores mean
the same as another's.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from dengue.metrics import pos_weight
from dengue.validation import SEEDS, train_seed_ensemble

__all__ = ["BLEND_WEIGHT", "BlendedEmergence", "fit_blend"]

# An even average. Not tuned: with 14 folds and a fold sd of 0.13 the harness cannot
# resolve the difference between 0.5 and 0.6, so fitting the weight would be reading
# noise, and a weight chosen that way is one more thing to overfit.
BLEND_WEIGHT = 0.5


class BlendedEmergence:
    """A classifier and a regression of the same window, averaged as probabilities.

    Satisfies the only interface `predictor.probabilities` needs - a `predict(X)`
    returning positive-class probabilities - so it drops into the existing bundle,
    the existing calibrator and the existing app without any of them knowing.
    Pickles like any other estimator.
    """

    def __init__(self, classifier, regressor, bridge, weight: float = BLEND_WEIGHT):
        self.classifier = classifier
        self.regressor = regressor
        self.bridge = bridge
        self.weight = float(weight)

    def predict(self, X):
        clf = np.asarray(self.classifier.predict(X), dtype=float)
        reg = np.asarray(self.bridge.predict(self.regressor.predict(X)), dtype=float)
        return self.weight * clf + (1.0 - self.weight) * reg

    def feature_importance(self, importance_type: str = "gain"):
        """Summed across both members, so it reads like a single booster's.

        The two are trained on the same columns in the same order, which is what
        makes adding them meaningful.
        """
        return self.classifier.feature_importance(importance_type) + self.regressor.feature_importance(
            importance_type
        )


def fit_blend(
    train: pd.DataFrame,
    features: Sequence[str],
    *,
    params: dict,
    num_boost_round: int,
    seeds: Sequence[int] = SEEDS,
    label_col: str = "y",
    reg_label_col: str = "y_log_inc",
    weight: float = BLEND_WEIGHT,
) -> BlendedEmergence:
    """Fit both members and the scale bridge between them, on `train` alone.

    Each member is a seed ensemble for the same reason the single model was: one
    booster is a draw from a distribution about 0.02 PR-AUC wide, and deploying an
    unaveraged model while measuring an averaged one measures the wrong thing.
    """
    features = list(features)

    classifier = train_seed_ensemble(
        {**params, "scale_pos_weight": pos_weight(train[label_col].to_numpy())},
        train,
        features,
        num_boost_round=num_boost_round,
        seeds=seeds,
    )
    # No class weighting: there are no classes. Every row carries a magnitude.
    regressor = train_seed_ensemble(
        {**params, "objective": "regression"},
        train,
        features,
        num_boost_round=num_boost_round,
        label_col=reg_label_col,
        seeds=seeds,
    )

    # Fitted on in-sample training predictions, which would be wrong for a
    # calibration and is right for a bridge: it is monotone, so it changes no
    # ranking, and the alternative - an out-of-fold bridge - would map from a
    # distribution the shipped regressor does not produce.
    bridge = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0 - 1e-4)
    bridge.fit(regressor.predict(train[features]), train[label_col].to_numpy(dtype=float))

    return BlendedEmergence(classifier, regressor, bridge, weight)
