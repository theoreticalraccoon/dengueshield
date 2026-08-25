"""How should a fixed weekly inspection budget be spent?

Every score in this project is turned into a decision by ONE global threshold, and
that choice has never been examined. It carries a strong assumption: that a score of
0.3 in February means what a score of 0.3 in July means. In a seasonal disease it
does not. During transmission season most districts clear any fixed bar at once, so a
season's inspection capacity is consumed in a handful of weeks, and in the quiet half
of the year the same threshold raises nothing at all - including in the weeks when a
new outbreak is actually starting somewhere.

A public-health team does not have a threshold. It has a number of districts it can
visit per week. Ranking within the week spends exactly that, every week, and needs
only the ordering inside one week rather than calibration across a year - strictly
less than the global threshold demands of the model.

This asks the operational question directly:

    to catch 90% of emerging outbreaks, how many districts must be visited a week?

Both policies are scored on the same out-of-fold predictions from the same model, so
the comparison isolates the policy. Development folds only; the locked years are
scored once, later, in finalize_emergence.py.

    .venv/Scripts/python.exe experiments/accuracy_v2/alert_policy.py
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

from dengue.config import LGBM_PARAMS_SL, SEED
from dengue.emergence import (
    HISTORY_FEATURES,
    add_base_features,
    add_extra_features,
    add_history_features,
    label_emergence,
)
from dengue.metrics import evaluate_threshold, pos_weight, top_k_by_week
from dengue.srilanka import COMMON_FEATURES, build_panel
from dengue.validation import fit_predict_seeds, rolling_origin

PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600
FIRST_TEST_YEAR = 2010
SEEDS = (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)
TARGET = 0.90


def oof_predictions(labelled: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    frames = []
    for _year, train, test in rolling_origin(labelled, first_test_year=FIRST_TEST_YEAR):
        if test.y.sum() == 0:
            continue
        p = fit_predict_seeds(
            {**PARAMS, "scale_pos_weight": pos_weight(train.y.values)},
            train,
            test,
            feats,
            num_boost_round=ROUNDS,
            seeds=SEEDS,
        )
        frames.append(
            pd.DataFrame(
                {
                    "week_start": test.week_start.values,
                    "district": test.district.values,
                    "y": test.y.values,
                    "p": p,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    panel = add_extra_features(add_history_features(add_base_features(build_panel())))
    feats = [c for c in COMMON_FEATURES if c in panel.columns]
    feats += [c for c in HISTORY_FEATURES if c in panel.columns]

    oof = oof_predictions(label_emergence(panel), feats)
    y = oof.y.to_numpy(dtype=float)
    n_weeks = oof.week_start.nunique()
    print(f"out-of-fold rows={len(oof)}  weeks={n_weeks}  positives={int(y.sum())}\n")

    # --- policy 1: one global threshold, swept over the observed scores
    global_curve = []
    for t in np.unique(np.quantile(oof.p, np.linspace(0.0, 1.0, 400))):
        op = evaluate_threshold(y, oof.p.to_numpy(), t)
        global_curve.append(
            {"alerts_per_week": op.flagged / n_weeks, "recall": op.recall, "precision": op.precision}
        )

    # --- policy 2: the k highest-scoring districts within each week
    topk_curve = []
    for k in range(1, 15):
        flag = top_k_by_week(oof.p.to_numpy(), oof.week_start.to_numpy(), k)
        op = evaluate_threshold(y, flag.astype(float), 0.5)
        topk_curve.append(
            {
                "k": k,
                "alerts_per_week": op.flagged / n_weeks,
                "recall": op.recall,
                "precision": op.precision,
            }
        )

    def cheapest(curve):
        """Fewest alerts per week that still reaches the recall target."""
        ok = [c for c in curve if c["recall"] >= TARGET]
        return min(ok, key=lambda c: c["alerts_per_week"]) if ok else None

    g, k = cheapest(global_curve), cheapest(topk_curve)
    print(f"to catch {TARGET:.0%} of emerging outbreaks:")
    print(
        f"  global threshold   {g['alerts_per_week']:5.2f} districts/week  "
        f"(recall {g['recall']:.3f}, precision {g['precision']:.3f})"
        if g
        else "  global threshold   never reaches the target"
    )
    print(
        f"  top-k within week  {k['alerts_per_week']:5.2f} districts/week  "
        f"(recall {k['recall']:.3f}, precision {k['precision']:.3f}, k={k['k']})"
        if k
        else "  top-k within week  never reaches the target"
    )
    if g and k:
        saved = g["alerts_per_week"] - k["alerts_per_week"]
        print(f"  -> {saved:+.2f} districts/week ({saved / g['alerts_per_week']:+.1%})")

    print("\nrecall at a fixed weekly budget:")
    print(f"  {'k':>3} {'alerts/wk':>10} {'top-k recall':>13} {'global recall':>14}")
    for row in topk_curve[:10]:
        # The global threshold that raises the same volume of alerts, for a like
        # for like comparison at equal cost.
        same_cost = min(global_curve, key=lambda c: abs(c["alerts_per_week"] - row["alerts_per_week"]))
        print(
            f"  {row['k']:>3} {row['alerts_per_week']:>10.2f} {row['recall']:>13.3f} "
            f"{same_cost['recall']:>14.3f}"
        )

    out = Path(__file__).parent / "alert_policy.json"
    out.write_text(
        json.dumps(
            {
                "protocol": (
                    f"out-of-fold predictions, rolling-origin {FIRST_TEST_YEAR}..validation, "
                    f"{len(SEEDS)} seeds. Development years only."
                ),
                "n_weeks": int(n_weeks),
                "target_recall": TARGET,
                "cheapest_global": g,
                "cheapest_top_k": k,
                "top_k_curve": topk_curve,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
