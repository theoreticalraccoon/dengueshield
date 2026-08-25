"""Does anything move emergence? Feature blocks, measured on two resampling axes.

`search_emergence.py` asked whether emergence could be given more ROWS, and answered
no: extra context rows, a hazard reformulation and both together all landed inside
the noise (docs/adr/0003). This asks whether it can be given more COLUMNS - which
that ADR also answered no to, for the one block anyone had tried.

Two things are different this time.

**The instrument.** The previous sweep ran eight folds from 2016 and compared them
with a paired bootstrap over folds. At a fold sd of ~0.13 that interval cannot
resolve anything below about 0.05, and every candidate left is smaller than that -
which is why ENSO sits in the ADR as "+0.009, unresolved" rather than accepted or
rejected. Folds start at 2010 here, and every arm is additionally compared on POOLED
out-of-fold predictions with the interval taken over districts. Years and districts
are different ways for a result to be a fluke; an arm ships only if both agree.

**The columns.** The rejected block was k-nearest-neighbour incidence, which with 26
districts is a quarter of the country and duplicates information the district's own
history already carries. These blocks are not that:

  climate       departure from this district's own week-of-year normal, plus the
                8-16 week rainfall memory the panel never had and the diurnal range
                it was throwing away
  susceptible   a reconstructed susceptible pool - the mechanism behind multi-annual
                dengue cycles, and the one thing that knows how BIG the last
                epidemic was
  importation   gravity-weighted force of infection from districts actually in
                outbreak, plus the Colombo-Gampaha commuter core
  national      where the country is in its own epidemic cycle

Evaluation is PR-AUC on the eligible rows of the held-out year, identical across
arms. The locked test years are never touched.

    .venv/Scripts/python.exe experiments/accuracy_v2/search_emergence_v3.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from dengue.config import LGBM_PARAMS_SL, SEED
from dengue.emergence import (
    CLIMATE_FEATURES,
    HISTORY_FEATURES,
    IMPORTATION_FEATURES,
    NATIONAL_FEATURES,
    SUSCEPTIBLE_FEATURES,
    add_base_features,
    add_extra_features,
    add_history_features,
    label_emergence,
)
from dengue.metrics import pos_weight
from dengue.srilanka import COMMON_FEATURES, build_panel
from dengue.validation import compare, fit_predict_seeds, pooled_compare, rolling_origin

PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600

# 2010 rather than 2016: the fold count is the binding constraint on what this can
# resolve, and 2007-09 are excluded only because they carry too few positives to
# give a defined PR-AUC.
FIRST_TEST_YEAR = 2010

# Five rather than three. The seed lottery is the noise floor this whole module
# exists to get under, and averaging more draws lowers it at linear cost.
SEEDS = (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)

BLOCKS = {
    "climate": CLIMATE_FEATURES,
    "susceptible": SUSCEPTIBLE_FEATURES,
    "importation": IMPORTATION_FEATURES,
    "national": NATIONAL_FEATURES,
}


def fold_run(panel: pd.DataFrame, feats: list[str]) -> tuple[list[float], pd.DataFrame]:
    """Per-fold PR-AUC and the stacked out-of-fold predictions, from one set of fits.

    Both comparison axes are computed from this single pass: `compare` needs the
    per-fold numbers, `pooled_compare` needs the raw predictions with the district
    each belongs to.
    """
    labelled = label_emergence(panel)
    scores: list[float] = []
    frames: list[pd.DataFrame] = []

    for year, train, test in rolling_origin(labelled, first_test_year=FIRST_TEST_YEAR):
        if test.y.sum() == 0:
            continue
        params = {**PARAMS, "scale_pos_weight": pos_weight(train.y.values)}
        p = fit_predict_seeds(params, train, test, feats, num_boost_round=ROUNDS, seeds=SEEDS)
        scores.append(float(average_precision_score(test.y.values, p)))
        frames.append(
            pd.DataFrame({"district": test.district.values, "y": test.y.values, "p": p, "year": year})
        )

    return scores, pd.concat(frames, ignore_index=True)


def main() -> int:
    panel = add_extra_features(add_history_features(add_base_features(build_panel())))
    base_feats = [c for c in COMMON_FEATURES if c in panel.columns]
    base_feats += [c for c in HISTORY_FEATURES if c in panel.columns]

    base_scores, base_oof = fold_run(panel, base_feats)
    print(f"folds={len(base_scores)}  base features={len(base_feats)}")
    print(
        f"  A production                    per-fold {np.mean(base_scores):.4f}  "
        f"pooled {average_precision_score(base_oof.y, base_oof.p):.4f}  "
        f"ROC {roc_auc_score(base_oof.y, base_oof.p):.4f}\n",
        flush=True,
    )

    arms = [("all", list(BLOCKS)), *[(name, [name]) for name in BLOCKS]]
    rows = []
    for label, names in arms:
        feats = list(base_feats)
        for n in names:
            feats += [c for c in BLOCKS[n] if c in panel.columns]

        scores, oof = fold_run(panel, feats)
        if len(scores) != len(base_scores):
            print(f"  {label:30} skipped: {len(scores)} folds vs {len(base_scores)}")
            continue

        by_fold = compare(base_scores, scores)
        by_district = pooled_compare(base_oof.y, base_oof.district, base_oof.p, oof.p)
        ships = by_fold.helps and by_district.helps

        print(f"  {label:30} folds     {by_fold.format()}")
        print(f"  {'':30} districts {by_district.format()}")
        print(f"  {'':30} -> {'SHIP' if ships else 'no'}\n", flush=True)

        rows.append(
            {
                "arm": label,
                "blocks": names,
                "n_features": len(feats),
                "per_fold_mean": by_fold.candidate_mean,
                "per_fold_delta": by_fold.mean_delta,
                "per_fold_ci": [by_fold.ci_low, by_fold.ci_high],
                "pooled_pr_auc": by_district.candidate_mean,
                "pooled_delta": by_district.mean_delta,
                "pooled_ci": [by_district.ci_low, by_district.ci_high],
                "pooled_roc_auc": float(roc_auc_score(oof.y, oof.p)),
                "ships": bool(ships),
            }
        )

    out = Path(__file__).parent / "search_emergence_v3.json"
    out.write_text(
        json.dumps(
            {
                "protocol": (
                    f"rolling-origin {FIRST_TEST_YEAR}..validation, {len(SEEDS)} seeds, "
                    "paired bootstrap over folds AND cluster bootstrap over districts; "
                    "an arm ships only if both intervals exclude zero in the same direction"
                ),
                "n_folds": len(base_scores),
                "baseline": {
                    "per_fold_mean": float(np.mean(base_scores)),
                    "pooled_pr_auc": float(average_precision_score(base_oof.y, base_oof.p)),
                    "pooled_roc_auc": float(roc_auc_score(base_oof.y, base_oof.p)),
                },
                "arms": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"survives BOTH tests: {[r['arm'] for r in rows if r['ships']] or 'none'}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
