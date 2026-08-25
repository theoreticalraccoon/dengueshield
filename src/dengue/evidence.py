"""Named access to what the models produced, instead of a bag of filenames.

The app used to load 20 artifacts into one dict and index it by key. Every caller
then had to know the key, the schema behind it, and that any of them could be None.
Eight of the twenty were never read at all - including the NDCU situation table,
which the app loaded on every start and rendered nowhere - and nothing made that
visible, because a key nobody reads looks exactly like a key somebody reads.

Headline figures were worse: the About screen quoted them as string literals. Each
had a real source in reports/, but nothing connected the two, so retraining moved
the artifact and not the page. That is how the persistence baseline came to be
quoted from a horizon-4 run beside a horizon-2 model score.

Absence is normal here - a clean checkout ships no trained models and no reports -
so every accessor returns None or an empty result rather than raising, and the app
renders its own banner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dengue.config import MODELS, PROC, REPORTS

# ------------------------------------------------------------------ freshness


def artifact_version(
    models: Path = MODELS, reports: Path = REPORTS, proc: Path = PROC
) -> tuple[tuple[str, int, int], ...]:
    """A key that changes whenever anything the app reads changes on disk.

    Streamlit caches on the value of a function's arguments, and every cached
    loader in app.py took none - so the key was constant and the cache never
    expired for the life of the process. Retraining rewrote the artifacts and the
    running dashboard went on serving what it had read at startup.

    That is invisible in production, where a push restarts the app and clears the
    cache with it, and very visible locally: the emergence column sat at its old
    values through four retrains. Passing this to the cached loaders makes a
    changed file invalidate them, which is what the weekly refresh needs too -
    `refresh_data.py` rewrites these while the app may be running.

    Deliberately one key for all of them rather than one per loader. The files
    change together (a retrain rewrites the bundle and the forecast in the same
    run), so splitting it would buy nothing and add a way for the two to disagree.

    **What this can and cannot notice.** It stats; it does not read. Size is part of
    the key as well as modification time, so a rewrite that changes either is
    caught, and stat-ing 62 files costs microseconds where hashing the 30 MB behind
    them would cost roughly a tenth of a second on every widget interaction.

    What it cannot catch is a rewrite that changes neither - same byte count, and
    quick enough that the filesystem timestamp has not ticked. That is not a real
    risk for this app, where rewrites come from a retrain minutes or days apart, but
    it is the reason the test for this asserts a changed *stat* rather than changed
    *content*: asserting content would be asserting something a stat-based key
    cannot deliver at a price worth paying.
    """
    roots = ((models, "*.joblib"), (reports, "*"), (proc, "*.parquet"))
    out: list[tuple[str, int, int]] = []
    for root, pattern in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            out.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(out)


# ------------------------------------------------------------------ raw reads


def _csv(root: Path, name: str) -> pd.DataFrame | None:
    path = root / name
    return pd.read_csv(path) if path.exists() else None


def _json(root: Path, name: str):
    path = root / name
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------ forecasts


def forecasts(root: Path = REPORTS) -> pd.DataFrame | None:
    """The district-week dual risk table the Outbreak screen is built on."""
    return _csv(root, "srilanka_dual_risk.csv")


def history(proc: Path = PROC) -> pd.DataFrame | None:
    """District case + rainfall series behind the per-district chart."""
    path = proc / "srilanka_history.parquet"
    return pd.read_parquet(path) if path.exists() else None


def freshness(root: Path = REPORTS) -> dict:
    """When the surveillance inputs were last refreshed. Empty if never run."""
    return _json(root, "data_freshness.json") or {}


def situation_freshness(root: Path = REPORTS) -> dict:
    return _json(root, "situation_freshness.json") or {}


# ------------------------------------------------------------------ headline


@dataclass(frozen=True)
class TaskScore:
    """One of the four questions, its score, and the baseline it must beat.

    `secondary` exists because the four-row table was comparing the models on
    different axes and quietly disadvantaging the outbreak ones. Screening and
    complication were quoted on ROC-AUC; continuation and emergence on PR-AUC. Those
    are not interchangeable - PR-AUC is bounded below by prevalence, so on a task
    with 6.5% positives it starts near zero where ROC-AUC starts at 0.5 - and the
    emergence model that reads as 0.405 on one reads as 0.827 on the other. Showing
    a single number per row therefore made a presentation choice look like a finding.
    Both are reported, and neither is dropped for being the less flattering one.
    """

    question: str
    asked_of: str
    metric: str
    score: float | None
    baseline: float | None = None
    verdict: str = ""
    secondary_metric: str = ""
    secondary_score: float | None = None

    @property
    def known(self) -> bool:
        return self.score is not None

    @property
    def has_secondary(self) -> bool:
        return bool(self.secondary_metric) and self.secondary_score is not None

    def format(self) -> str:
        if self.score is None:
            return "n/a"
        out = f"{self.metric} {self.score:.3f}"
        if self.has_secondary:
            out += f" · {self.secondary_metric} {self.secondary_score:.3f}"
        return out


def _pick(rows, model: str, field: str):
    for r in rows or []:
        if r.get("model") == model:
            return r.get(field)
    return None


def headline_metrics(root: Path = REPORTS) -> dict[str, TaskScore]:
    """The four task scores, sourced from artifacts rather than restated.

    Continuation and its persistence baseline come from the same file, so they
    are always the same experiment - quoting a model from one horizon beside a
    baseline from another is the bug this replaces.
    """
    m1 = _json(root, "model1_operating_points.json") or {}
    peds = _json(root, "peds_final.json") or {}
    cont = _json(root, "srilanka_continuation.json") or {}
    emg = _json(root, "srilanka_emergence.json") or {}

    return {
        "screening": TaskScore(
            question="Does this patient appear to have dengue?",
            asked_of="one febrile patient",
            metric="ROC-AUC",
            score=m1.get("roc_auc"),
            verdict="Modest",
        ),
        "complication": TaskScore(
            question="Does this dengue patient need closer monitoring?",
            asked_of="one dengue admission",
            metric="ROC-AUC",
            score=peds.get("roc_auc"),
            verdict="Strong",
        ),
        # Sri Lanka, not Brazil. This row used to be sourced from
        # model2_summary.json - the Brazil municipality panel, PR-AUC 0.960 - and
        # sat in a table beside Sri Lankan emergence, above a screen showing Sri
        # Lankan forecasts. It is exactly the confusion the emergence row was fixed
        # for, and it survived in this one because both figures came from artifacts
        # and nothing checked that they came from the same COUNTRY. The four rows
        # describe the four deployed models; Brazil is a separate experiment and is
        # reported as one, via `brazil_continuation`.
        "continuation": TaskScore(
            question="Will this district's outbreak continue 14 days?",
            asked_of="any Sri Lankan district",
            metric="PR-AUC",
            score=cont.get("pr_auc"),
            baseline=cont.get("baseline_persistence_pr_auc"),
            verdict="Strong",
            secondary_metric="ROC-AUC",
            secondary_score=cont.get("roc_auc"),
        ),
        "emergence": TaskScore(
            question="Will a new outbreak begin here?",
            asked_of="quiet districts only",
            metric="PR-AUC",
            score=emg.get("pr_auc"),
            baseline=emg.get("baseline_persistence_pr_auc"),
            verdict="Harder",
            secondary_metric="ROC-AUC",
            secondary_score=emg.get("roc_auc"),
        ),
    }


@dataclass(frozen=True)
class Delivered:
    """One thing the model delivers, against what doing nothing would deliver.

    The baseline is not decoration. Emergence reaches NPV 0.977 and 93.7% recall on
    the locked years, and both are real - but a model that never flags anything
    scores 0.935 NPV, so quoting the first without the second turns a modest gain
    into a spectacular one. The pairing is the honesty, and it is enforced by the
    type rather than left to whoever writes the page.
    """

    label: str
    value: float
    baseline: float | None
    baseline_label: str
    note: str = ""

    @property
    def beats_baseline(self) -> bool:
        return self.baseline is None or self.value > self.baseline


def emergence_delivery(root: Path = REPORTS) -> list[Delivered]:
    """What the emergence model actually delivers at its deployed operating point.

    Sourced from the artifact rather than typed into the page - the same reason
    `headline_metrics` exists. Returns an empty list before the first retrain that
    records these fields, so the app renders nothing rather than something stale.
    """
    m = emergence_metrics(root)
    if not m:
        return []

    prevalence = m.get("prevalence")
    out: list[Delivered] = []

    if (roc := m.get("roc_auc")) is not None:
        out.append(
            Delivered(
                "Ranking quality (ROC-AUC)",
                float(roc),
                0.5,
                "coin flip",
                "the axis the two clinical models are quoted on",
            )
        )
    if (npv := m.get("npv")) is not None and m.get("trivial_npv") is not None:
        out.append(
            Delivered(
                "A district called clear stays clear (NPV)",
                float(npv),
                float(m["trivial_npv"]),
                "never flag anything",
                "inflated by how rare emergence is - read it against the baseline",
            )
        )

    # The highest-recall operating point that was actually measured, with its price.
    ops = m.get("operating_points") or []
    best = max(ops, key=lambda o: o.get("recall", 0.0)) if ops else None
    if best:
        total = m.get("districts_total")
        price = f"{best['flagged_per_week']:.1f} districts flagged a week"
        if total:
            price += f" of {total}"
        out.append(
            Delivered(
                "Emerging outbreaks caught, at maximum sensitivity",
                float(best["recall"]),
                0.0,
                "never flag anything",
                price,
            )
        )

    if (pr := m.get("pr_auc")) is not None:
        out.append(
            Delivered(
                "Precision-recall (PR-AUC)",
                float(pr),
                m.get("baseline_persistence_pr_auc"),
                "persistence",
                f"starts at {prevalence:.3f} by construction" if prevalence else "",
            )
        )
    return out


def emergence_budget(root: Path = REPORTS) -> pd.DataFrame | None:
    """Recall against a fixed weekly inspection budget, if the retrain recorded it."""
    rows = emergence_metrics(root).get("budget_points")
    return pd.DataFrame(rows) if rows else None


# ------------------------------------------------------------------ detail


def screening_operating_points(root: Path = REPORTS) -> dict | None:
    return _json(root, "model1_operating_points.json")


def complication_metrics(root: Path = REPORTS) -> dict:
    return _json(root, "peds_final.json") or {}


def brazil_continuation(root: Path = REPORTS) -> TaskScore:
    """The Brazil municipality experiment, reported as its own thing.

    Far stronger than the Sri Lankan model (PR-AUC ~0.96 against ~0.76) because it
    has thousands of municipalities rather than 26 districts. It is not what the
    app deploys, and putting it in the four-task table made it look as though it
    were - see the note in `headline_metrics`.
    """
    rows = _json(root, "model2_summary.json") or []
    return TaskScore(
        question="Will this municipality's outbreak continue 14 days?",
        asked_of="any Brazilian municipality (experiment, not deployed)",
        metric="PR-AUC",
        score=_pick(rows, "LightGBM", "pr_auc"),
        baseline=_pick(rows, "BASELINE_persistence", "pr_auc"),
        verdict="Strong",
    )


def continuation_metrics(root: Path = REPORTS) -> dict:
    """Sri Lanka continuation, including its calibration figures.

    These used to live only inside models/srilanka_outbreak.joblib, so the About
    screen had nothing to quote and asserted the calibration status in prose.
    """
    return _json(root, "srilanka_continuation.json") or {}


def emergence_metrics(root: Path = REPORTS) -> dict:
    return _json(root, "srilanka_emergence.json") or {}


def srilanka_calibration(root: Path = REPORTS) -> dict[str, tuple[float, float]]:
    """{task: (ECE before, ECE after)} for whichever Sri Lanka models report it.

    Empty when neither has been retrained since calibration was introduced, which
    is what lets the app say nothing rather than say something stale.
    """
    out = {}
    for task, metrics in (
        ("continuation", continuation_metrics(root)),
        ("emergence", emergence_metrics(root)),
    ):
        before, after = metrics.get("ece_uncalibrated"), metrics.get("ece")
        if before is not None and after is not None:
            out[task] = (float(before), float(after))
    return out


def dataset_audit(root: Path = REPORTS) -> pd.DataFrame | None:
    """Which candidate datasets passed the integrity gate, and why."""
    return _csv(root, "dataset_audit.csv")


def robustness(root: Path = REPORTS) -> dict:
    return _json(root, "model2_robustness.json") or {}


def information_ablation(root: Path = REPORTS) -> pd.DataFrame | None:
    """Leave-group-out ablation: incremental value, as opposed to attribution."""
    return _csv(root, "model2_information_ablation.csv")


def calibration_errors(root: Path = REPORTS) -> dict:
    return _json(root, "model2_calibration_errors.json") or {}


def spatial_holdout(root: Path = REPORTS) -> pd.DataFrame | None:
    """Seen vs unseen municipalities - does the model generalise geographically?"""
    return _csv(root, "model2_spatial_holdout.csv")


def generalisation_gap(root: Path = REPORTS) -> tuple[float | None, float | None]:
    """(PR-AUC on unseen municipalities, its gap vs seen ones).

    Both come from the same file, so the score and the gap are always the same
    experiment - the reason these stopped being string literals in the app.
    """
    df = spatial_holdout(root)
    if df is None or "condition" not in df.columns:
        return None, None
    by = df.set_index("condition").pr_auc
    seen = by.get("A_temporal_seen_municipalities")
    unseen = by.get("B_spatiotemporal_unseen_municipalities")
    if unseen is None:
        return None, None
    return float(unseen), (None if seen is None else float(unseen) - float(seen))


def calibration_gap(root: Path = REPORTS) -> tuple[float | None, float | None]:
    """(calibrated ECE, improvement over raw). Lower is better, so the gap is negative."""
    cal = calibration_errors(root)
    raw = (cal.get("raw") or {}).get("ece")
    fitted = (cal.get("calibrated") or {}).get("ece")
    if fitted is None:
        return None, None
    return float(fitted), (None if raw is None else float(fitted) - float(raw))


def validation_battery(root: Path = REPORTS) -> list[tuple[str, str]]:
    """Every out-of-sample check, read from the artifact that produced it.

    This table used to be eight hand-typed strings in the app. After the
    horizon-2 retrain six of them were wrong - the rolling-origin and threshold
    sweeps had been run at horizon 4. Assembling it from the reports means a
    re-run moves the page.
    """
    rows: list[tuple[str, str]] = []

    head = headline_metrics(root).get("continuation")
    summary = _json(root, "model2_summary.json")
    acc = None
    if isinstance(summary, list):
        best = next((r for r in summary if r.get("model") == "LightGBM"), None)
        acc = (best or {}).get("accuracy")
    if head and head.known:
        tail = f" · {acc:.1%} acc" if acc else ""
        rows.append(("Temporal holdout (locked 2024–25)", f"PR-AUC {head.score:.3f}{tail}"))

    rb = robustness(root)
    ro = [r["pr_auc"] for r in rb.get("rolling_origin", []) if "pr_auc" in r]
    if ro:
        rows.append(
            (
                f"Rolling-origin backtest, {len(ro)} years",
                f"mean {sum(ro) / len(ro):.3f} · worst year {min(ro):.3f}",
            )
        )

    unseen, gap = generalisation_gap(root)
    if unseen is not None and gap is not None:
        rows.append(
            ("Spatial holdout, unseen municipalities", f"{unseen:.3f} vs {unseen - gap:.3f} seen")
        )

    holdout = spatial_holdout(root)
    if holdout is not None and "condition" in holdout.columns:
        folds = holdout[holdout.condition.str.startswith("state_fold")]
        if not folds.empty:
            rows.append(
                (
                    "Leave-whole-states-out",
                    f"mean {folds.pr_auc.mean():.3f} (spread is prevalence)",
                )
            )

    hz = rb.get("horizons", [])
    if hz:
        labels = " / ".join(str(int(r["horizon"])) for r in hz)
        scores = " / ".join(f"{r['pr_auc']:.3f}" for r in hz)
        rows.append((f"Horizon sweep ({labels} weeks)", scores))

    shuffled = (rb.get("shuffled_control") or {}).get("pr_auc")
    if shuffled is not None:
        chance = (rb.get("shuffled_control") or {}).get("test_prevalence")
        note = f" (chance {chance:.3f})" if chance else ""
        rows.append(("Shuffled-label control", f"collapses to {shuffled:.3f}{note}"))

    leak = _json(root, "leakage_audit.json") or {}
    checks = leak.get("contamination_checks") or {}
    if checks:
        clean = all(bool(v) for v in checks.values() if isinstance(v, (bool, int, float)))
        rows.append(
            (
                "Temporal-leakage audit",
                "no lag matched a future value" if clean else "CONTAMINATION DETECTED",
            )
        )

    delay = _csv(root, "reporting_delay_stress.csv")
    if delay is not None and {"delay_weeks", "pr_auc"} <= set(delay.columns):
        d = delay.set_index("delay_weeks").pr_auc
        if 0 in d.index and d.index.max() > 0:
            worst = d.index.max()
            rows.append(
                (
                    f"Reporting-delay stress ({int(worst)} weeks)",
                    f"{d[0]:.3f} → {d[worst]:.3f}",
                )
            )

    return rows


def transfer(root: Path = REPORTS) -> pd.DataFrame | None:
    """Brazil -> Sri Lanka zero-shot transfer against local training."""
    return _csv(root, "srilanka_transfer.csv")


def head_to_head(root: Path = REPORTS) -> pd.DataFrame | None:
    return _csv(root, "model2_headtohead.csv")
