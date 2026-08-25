"""One table: what each model scored before, what it scores now, and against what.

The point of the baseline column is that it is there. A model whose PR-AUC rises
from 0.71 to 0.73 has improved only if the trivial baseline did not rise with it,
and 86.5% of municipality-weeks being negative means most of these metrics flatter
whatever produced them. Every row therefore carries the score it has to beat.

The AFTER column is read from the artifacts on disk every run, so it cannot drift
from what the models actually did.

The BEFORE column is two different kinds of thing, and the difference matters:

  * For the discrimination rows it is historical - what the models scored when this
    work started - and no code in the repo can recompute it, because the code that
    produced it has been replaced. It is a literal with its provenance, in the same
    spirit as the snapshot artifacts in src/dengue/artifacts.py: a measurement of a
    state that has moved on is a fact to record, not a thing to regenerate.
  * For the ECE rows it is read from the artifact too. Each one records its own
    uncalibrated figure, so before and after are the same model's scores with and
    without the calibrator - the correct paired comparison, and one that cannot go
    stale.

    .venv/Scripts/python.exe summarise_improvements.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from dengue import evidence
from dengue.config import REPORTS

# Measured on the same locked test years, before any of this work. Sources:
#   screening / complication  reports/model1_final.json, peds_final.json at HEAD~
#   continuation              num_leaves=63, 500 rounds, 3-seed averaged
#   emergence                 reports/srilanka_emergence.json at HEAD~ (single seed)
# See docs/adr/0003 and 0004 for how each was obtained.
BEFORE = {
    "screening": {"roc_auc": 0.6811, "pr_auc": 0.7790},
    "complication": {"roc_auc": 0.8741, "pr_auc": 0.7311},
    "continuation": {"pr_auc": 0.7086},
    # The 2026-08-24 retrain, i.e. the state this accuracy push started from: a
    # single binary classifier, 3 seeds, isotonic calibration with ties.
    "emergence": {
        "pr_auc": 0.4054,
        "pr_auc_calibrated": 0.3584,
        "recall": 0.7314,
        "alerts_at_recall_90": 17.31,
    },
}


def _row(task, metric, before, after, baseline=None, note=""):
    delta = None if (before is None or after is None) else round(after - before, 4)
    return {
        "task": task,
        "metric": metric,
        "before": before,
        "after": after,
        "delta": delta,
        "baseline": baseline,
        "note": note,
    }


def build() -> list[dict]:
    ops = evidence.screening_operating_points() or {}
    peds = evidence.complication_metrics()
    cont = evidence.continuation_metrics()
    emg = evidence.emergence_metrics()

    rows = [
        _row(
            "screening",
            "ROC-AUC",
            BEFORE["screening"]["roc_auc"],
            ops.get("roc_auc"),
            baseline=0.5,
            note=(
                "two changes, not one: absolute cell counts, and averaging the "
                "out-of-fold predictions over 3 repeats instead of 1. The counts "
                "alone measure +0.003 at fixed hyperparameters but +0.018 when "
                "tuning is free to adapt to them (reports/ablation_hematology.csv, "
                "tier C -> D); the deployed pipeline tunes, so the larger figure is "
                "the relevant one. Neither has a CI."
            ),
        ),
        _row(
            "screening",
            "PR-AUC",
            BEFORE["screening"]["pr_auc"],
            ops.get("pr_auc"),
            baseline=0.685,
            note="baseline is the prevalence - PR-AUC starts high here by construction.",
        ),
        _row(
            "complication",
            "ROC-AUC",
            BEFORE["complication"]["roc_auc"],
            peds.get("roc_auc"),
            baseline=0.5,
            note="unchanged by design: the fix was that the DEPLOYED model is now the tuned one.",
        ),
        _row(
            "continuation",
            "PR-AUC",
            BEFORE["continuation"]["pr_auc"],
            cont.get("pr_auc"),
            baseline=0.4688,
            note="capacity cut from 63 to 15 leaves and 500 to 250 rounds; seed-averaged.",
        ),
        _row(
            "continuation",
            "ECE",
            cont.get("ece_uncalibrated"),
            cont.get("ece"),
            note="lower is better. Same model, with and without the calibrator.",
        ),
        _row(
            "emergence",
            "PR-AUC",
            BEFORE["emergence"]["pr_auc"],
            emg.get("pr_auc"),
            baseline=emg.get("baseline_persistence_pr_auc"),
            note=(
                "the blend of the binary classifier with an incidence regression "
                "(dengue.blend), the only arm of twelve to clear both the per-fold "
                "and per-district intervals. See docs/adr/0006."
            ),
        ),
        _row(
            "emergence",
            "PR-AUC of the DEPLOYED score",
            BEFORE["emergence"]["pr_auc_calibrated"],
            emg.get("pr_auc_calibrated"),
            baseline=emg.get("baseline_persistence_pr_auc"),
            note=(
                "the number the app actually serves, which is the calibrated one. It "
                "used to rank measurably worse than the model it came from because "
                "isotonic merged scores into ties; TieBrokenIsotonic removes them, so "
                "this now equals the raw figure above rather than trailing it by 0.05."
            ),
        ),
        _row(
            "emergence",
            "ECE",
            emg.get("ece_uncalibrated"),
            emg.get("ece"),
            note="lower is better. The largest calibration gain of the four.",
        ),
        _row(
            "emergence",
            "recall @ sensitivity target 0.70",
            BEFORE["emergence"]["recall"],
            emg.get("recall"),
            note="operating point read off the calibrated scale.",
        ),
        _row(
            "emergence",
            "districts flagged per week for 90% recall",
            BEFORE["emergence"]["alerts_at_recall_90"],
            emg.get("alerts_at_recall_90"),
            baseline=emg.get("districts_total"),
            note=(
                "lower is better, and this is the operational question: to catch 90% "
                "of emerging outbreaks, how much of the country must be visited? The "
                "baseline column is the number of districts there are."
            ),
        ),
    ]
    return [r for r in rows if r["after"] is not None]


def main() -> int:
    rows = build()
    if not rows:
        print("no metric artifacts on disk - run the finalize scripts first")
        return 1

    print(f"{'task':14} {'metric':34} {'before':>8} {'after':>8} {'delta':>8} {'baseline':>9}")
    for r in rows:
        base = "-" if r["baseline"] is None else f"{r['baseline']:.4f}"
        before = "-" if r["before"] is None else f"{r['before']:.4f}"
        print(
            f"{r['task']:14} {r['metric']:34} {before:>8} "
            f"{r['after']:8.4f} {r['delta']:+8.4f} {base:>9}"
        )

    out = REPORTS / "improvement_summary.json"
    out.write_text(
        json.dumps(
            {
                "note": (
                    "AFTER is read from the artifacts on disk. BEFORE is a recorded "
                    "literal for the discrimination rows and the artifact's own "
                    "uncalibrated figure for the ECE rows. Lower is better for ECE."
                ),
                "rows": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
