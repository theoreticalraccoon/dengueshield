"""Train the Sri Lanka EMERGENCE model and produce the dual district forecast.

Gives the app two complementary numbers per district:

    continuation risk  - will an existing outbreak persist?   (finalize_srilanka.py)
    emergence risk     - is a NEW outbreak about to begin?     (this script)

Emergence is only asked of districts not currently in outbreak, which is exactly
where continuation is blind. Districts in neither population are scored NaN,
meaning "not asked" - see docs/adr/0001-blank-emergence-risk.md.

Depends on models/srilanka_outbreak.joblib, so finalize_srilanka.py must run first.
That ordering is now checked rather than assumed.

    python finalize_emergence.py
"""

from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dengue import predictor
from dengue.blend import fit_blend
from dengue.calibration import apply_calibrator, fit_isotonic_oof_model
from dengue.config import (
    EMERGENCE_HORIZON,
    EMERGENCE_SENSITIVITY_TARGET,
    LGBM_PARAMS_SL,
    MODELS,
    QUIET_WEEKS,
    REPORTS,
    SEED,
    SL_INC,
    TRAIN_END,
    VAL_END,
)
from dengue.emergence import (
    HISTORY_FEATURES,
    SHIPPED_FEATURES,
    add_base_features,
    add_extra_features,
    add_history_features,
    add_regression_target,
    label_emergence,
    lead_times,
)
from dengue.metrics import (
    evaluate_threshold,
    threshold_for_budget,
    threshold_for_f1,
    threshold_for_sensitivity,
)
from dengue.srilanka import COMMON_FEATURES, DISTRICTS, build_panel
from dengue.validation import SEEDS, cluster_ci, ece, pooled_compare

# Emergence needs far shallower trees than continuation: ~18k eligible training
# rows at ~4% positives. Selected by rolling-origin CV over seven folds ending
# 2023 (cv PR-AUC 0.363 -> 0.395); the locked test years were not consulted.
PARAMS = {
    **LGBM_PARAMS_SL,
    "num_leaves": 7,
    "reg_lambda": 50.0,
    "min_child_samples": 80,
    "learning_rate": 0.02,
    "seed": SEED,
}
ROUNDS = 600

# First fold of the out-of-fold calibration sweep, matching finalize_srilanka.py.
CALIBRATION_FIRST_YEAR = 2016

# The project default, deliberately not more.
#
# Raising this to nine looked free - averaging more draws of one estimator is
# variance reduction, not selection, so it cannot be chosen wrong - but it was
# measured before being kept, and it buys nothing: over the development folds the
# blend scores 0.3824 at three seeds, 0.3799 at five and 0.3816 at nine, a spread
# of 0.0025 that is the noise it was supposed to remove. What it does cost is real.
# The blend already doubles the artifact because it holds two members, and CI
# commits this bundle every Tuesday, so nine seeds meant a 9 MB file landing in git
# history weekly instead of 3 MB.
DEPLOY_SEEDS = SEEDS


def _fit(train_rows: pd.DataFrame, feats: list[str]):
    """The deployed estimator: classifier + incidence regression, averaged.

    One definition, used for the production fit AND for every out-of-fold fold the
    calibrator is built from, so what is calibrated is what ships.
    """
    return fit_blend(
        train_rows,
        feats,
        params=PARAMS,
        num_boost_round=ROUNDS,
        seeds=DEPLOY_SEEDS,
    )


def train(panel: pd.DataFrame) -> tuple[object, dict, list[str], object]:
    """Fit on train, tune the threshold on validation, score the locked test years."""
    feats = [c for c in COMMON_FEATURES if c in panel.columns]
    feats += [c for c in HISTORY_FEATURES if c in panel.columns]
    feats += [c for c in SHIPPED_FEATURES if c in panel.columns]
    # The regression member needs the unthresholded outcome, so rows without a full
    # future window drop out of both members together.
    em = label_emergence(panel).dropna(subset=["y_log_inc"])

    tr = em[em.anio <= TRAIN_END]
    va = em[(em.anio > TRAIN_END) & (em.anio <= VAL_END)]
    te = em[em.anio > VAL_END]
    print(f"eligible district-weeks: train={len(tr)} val={len(va)} test={len(te)}")
    print(
        f"emergence rate: train={tr.y.mean():.4f} test={te.y.mean():.4f} "
        f"({int(te.y.sum())} positive test events)"
    )

    # Two members, seed-averaged, averaged with each other - see dengue.blend. The
    # single classifier this replaces is still in there as half the blend.
    dev_model = _fit(tr, feats)

    # `scale_pos_weight` is ~23 on the classifier member, so the blended score is
    # still above the event rate. The sensitivity target below is read off this
    # scale, and the app prints it as a percentage, so it has to be a probability
    # first. Fitted out-of-fold across the development years - never on test.
    dev = pd.concat([tr, va])
    calibrator, cal_report = fit_isotonic_oof_model(
        lambda rows: _fit(rows, feats),
        dev,
        feats,
        first_test_year=CALIBRATION_FIRST_YEAR,
    )
    print(f"  {cal_report.format()}")

    raw_test = dev_model.predict(te[feats])
    p_val = apply_calibrator(calibrator, dev_model.predict(va[feats]))
    p_test = apply_calibrator(calibrator, raw_test)

    # Both thresholds come from validation. F1 balances the two errors; the
    # sensitivity target says how many emerging outbreaks we insist on catching.
    thr_f1 = threshold_for_f1(va.y.values, p_val)
    thr = threshold_for_sensitivity(va.y.values, p_val, EMERGENCE_SENSITIVITY_TARGET)
    yhat = (p_test >= thr).astype(int)
    baseline = float(average_precision_score(te.y.values, te.p_inc100k.values))

    # Every operating point, so the recall/false-alarm trade is a decision the
    # reader can see rather than a single number they have to trust.
    #
    # Each now carries its full confusion matrix and the rates derived from it.
    # Recall, precision and accuracy alone are the least informative subset
    # available on a task with 6.5% prevalence: they omit specificity, and they omit
    # NPV, which is the verdict the great majority of districts receive every week
    # and the one figure here that is genuinely above 90%.
    n_weeks = max(te.week_start.nunique(), 1)
    operating_points = []
    for target in (0.50, 0.60, 0.70, 0.80, 0.90):
        t = threshold_for_sensitivity(va.y.values, p_val, target)
        op = evaluate_threshold(te.y.values, p_test, t)
        operating_points.append({"sensitivity_target": target, **op.as_dict(n_weeks)})

    # The same trade expressed as a WEEKLY INSPECTION BUDGET, which is the constraint
    # a public-health team actually works under: not "what threshold" but "we can
    # visit N districts a week - what do we catch?"
    #
    # Deliberately still a global threshold. Ranking districts WITHIN each week and
    # flagging the top k looks like the better fit for a fixed team capacity, and it
    # was measured (experiments/accuracy_v2/alert_policy.py) and it is worse at every
    # budget: reaching 90% recall costs 10.8 districts a week under top-k against
    # 7.9 under a global threshold. Emergence events are concentrated in the
    # transmission season, so spending a constant budget every week rations alerts
    # exactly when they are needed and wastes them when nothing is starting. The
    # score is comparable across weeks after all.
    budget_points = []
    for k in range(1, 9):
        t = threshold_for_budget(p_test, n_weeks, k)
        op = evaluate_threshold(te.y.values, p_test, t)
        budget_points.append(
            {
                "districts_per_week": k,
                "threshold": op.threshold,
                "actual_per_week": round(op.flagged_per_week(n_weeks), 2),
                "recall": op.recall,
                "precision": op.precision,
                "npv": op.npv,
                "specificity": op.specificity,
                "lead_time": lead_times(te, p_test >= t),
            }
        )

    # What the model actually delivers at the point it is deployed at, each figure
    # beside the number a model that never flags anything would score. Several of
    # these are above 90% and none of them were being recorded; quoting them without
    # the trivial baseline would be the same defect in the other direction, because
    # never flagging already scores 93.5% accuracy and 93.5% NPV here.
    at_deployed = evaluate_threshold(te.y.values, p_test, thr)
    trivial_npv = float(1 - te.y.mean())

    # Intervals over DISTRICTS: 175 positive events cannot support a headline quoted
    # to three decimals as though it were exact. Measured once, at the end, on the
    # locked years - a report, never an input to a choice.
    _, pr_lo, pr_hi = cluster_ci(te.y.values, te.district.values, raw_test)
    vs_persistence = pooled_compare(
        te.y.values, te.district.values, te.p_inc100k.values, raw_test
    )

    deployed = {
        "npv": at_deployed.npv,
        "specificity": at_deployed.specificity,
        "balanced_accuracy": at_deployed.balanced_accuracy,
        "accuracy": at_deployed.accuracy,
        "trivial_npv": trivial_npv,
        "pr_auc_ci": [pr_lo, pr_hi],
        "pr_auc_vs_persistence_delta": vs_persistence.mean_delta,
        "pr_auc_vs_persistence_ci": [vs_persistence.ci_low, vs_persistence.ci_high],
        "beats_persistence": bool(vs_persistence.helps),
        "lead_time": lead_times(te, p_test >= thr),
        # The headline operating question: to catch 90% of emerging outbreaks, how
        # much of the country has to be visited every week?
        "alerts_at_recall_90": next(
            (op["flagged_per_week"] for op in operating_points if op["recall"] >= 0.90),
            None,
        ),
        "districts_total": int(te.district.nunique()),
    }

    # Discrimination on the raw score, calibration on the calibrated one - see the
    # same split in finalize_srilanka.py. Recall and precision below stay on the
    # calibrated scale, because that is the scale the operating point lives on.
    metrics = {
        "pr_auc": float(average_precision_score(te.y.values, raw_test)),
        "roc_auc": float(roc_auc_score(te.y.values, raw_test)),
        # Must not EXCEED pr_auc. If it does, the calibrator saw test labels.
        "pr_auc_calibrated": float(average_precision_score(te.y.values, p_test)),
        "ece": float(ece(te.y.values, p_test)),
        "ece_uncalibrated": float(ece(te.y.values, raw_test)),
        "brier": float(brier_score_loss(te.y.values, np.clip(p_test, 0, 1))),
        "brier_uncalibrated": float(brier_score_loss(te.y.values, np.clip(raw_test, 0, 1))),
        "calibration_dev": cal_report.as_dict(),
        "recall": float(recall_score(te.y.values, yhat, zero_division=0)),
        "precision": float(precision_score(te.y.values, yhat, zero_division=0)),
        "prevalence": float(te.y.mean()),
        "n_positive": int(te.y.sum()),
        "threshold": float(thr),
        "threshold_f1": float(thr_f1),
        "sensitivity_target": EMERGENCE_SENSITIVITY_TARGET,
        "operating_points": operating_points,
        "budget_points": budget_points,
        "horizon_weeks": EMERGENCE_HORIZON,
        "outbreak_inc": SL_INC,
        "quiet_weeks": QUIET_WEEKS,
        "baseline_persistence_pr_auc": baseline,
        "trivial_never_flag_accuracy": float(1 - te.y.mean()),
        **deployed,
    }
    print(
        f"held-out test: PR-AUC={metrics['pr_auc']:.4f} "
        f"(persistence baseline {baseline:.4f}) recall={metrics['recall']:.4f}"
    )

    # Production model: refit on train+val at the validation-chosen threshold, with
    # the same calibrator - which is why it was fitted out-of-fold across these
    # years rather than on the validation split alone.
    model = _fit(dev, feats)
    return model, metrics, feats, calibrator


def dual_forecast(
    panel: pd.DataFrame, model: lgb.Booster, feats: list[str], calibrator=None
) -> pd.DataFrame:
    """One row per district: continuation risk, emergence risk, and eligibility."""
    continuation = predictor.load(predictor.SL_CONTINUATION)
    if continuation is None:
        raise SystemExit(
            "models/srilanka_outbreak.joblib is missing - run finalize_srilanka.py first."
        )

    scored = label_emergence(panel, keep_ineligible=True)
    latest = panel.week_start.max()
    recent = scored[scored.week_start >= latest - pd.Timedelta(weeks=1)].copy()

    recent["emergence_risk"] = apply_calibrator(calibrator, model.predict(recent[feats]))
    recent.loc[~recent.eligible, "emergence_risk"] = np.nan  # question not asked
    recent["currently_in_outbreak"] = recent.p_inc100k >= SL_INC
    recent["continuation_risk"] = continuation.predict_frame(recent)

    cols = [
        "district",
        "week_start",
        "casos",
        "p_inc100k",
        "currently_in_outbreak",
        "continuation_risk",
        "emergence_risk",
        "casos_roll4_mean",
        "precip_roll4_sum",
        "tempmed",
        "umidmed",
        "population_total",
    ]
    out = recent.sort_values("week_start").groupby("district", as_index=False).last()[cols]
    out["lat"] = out.district.map(lambda d: DISTRICTS[d][0])
    out["lon"] = out.district.map(lambda d: DISTRICTS[d][1])
    return out.sort_values("continuation_risk", ascending=False)


def main(panel: pd.DataFrame | None = None) -> int:
    """Takes the RAW district-week panel, same as finalize_srilanka.main.

    refresh_data.py builds it once and hands the same frame to both, so the
    contract has to be identical; this script adds the features it needs.
    """
    raw = build_panel() if panel is None else panel
    panel = add_history_features(add_base_features(raw))
    if SHIPPED_FEATURES:
        panel = add_extra_features(panel)
    # The blend's regression member needs the unthresholded outcome.
    panel = add_regression_target(panel)

    model, metrics, feats, calibrator = train(panel)

    MODELS.mkdir(parents=True, exist_ok=True)
    predictor.save_bundle(
        MODELS / "srilanka_emergence.joblib",
        model=model,
        features=feats,
        threshold=metrics["threshold"],
        metrics=metrics,
        horizon=EMERGENCE_HORIZON,
        outbreak_inc=SL_INC,
        quiet_weeks=QUIET_WEEKS,
        calibrator=calibrator,
    )

    out = dual_forecast(panel, model, feats, calibrator)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "srilanka_dual_risk.csv", index=False)
    (REPORTS / "srilanka_emergence.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nforecast week {out.week_start.max().date()}  (horizon 1-{EMERGENCE_HORIZON} weeks)")
    print(
        out[["district", "casos", "p_inc100k", "currently_in_outbreak", "emergence_risk"]]
        .round(3)
        .head(14)
        .to_string(index=False)
    )
    print("\nsaved models/srilanka_emergence.joblib, reports/srilanka_dual_risk.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
