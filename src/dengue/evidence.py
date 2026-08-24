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

from dengue.config import PROC, REPORTS

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
    """One of the four questions, its score, and the baseline it must beat."""

    question: str
    asked_of: str
    metric: str
    score: float | None
    baseline: float | None = None
    verdict: str = ""

    @property
    def known(self) -> bool:
        return self.score is not None

    def format(self) -> str:
        return "n/a" if self.score is None else f"{self.metric} {self.score:.3f}"


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
    m2 = _json(root, "model2_summary.json") or []
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
        "continuation": TaskScore(
            question="Will this district's outbreak continue 14 days?",
            asked_of="any district",
            metric="PR-AUC",
            score=_pick(m2, "LightGBM", "pr_auc"),
            baseline=_pick(m2, "BASELINE_persistence", "pr_auc"),
            verdict="Strong",
        ),
        "emergence": TaskScore(
            question="Will a new outbreak begin here?",
            asked_of="quiet districts only",
            metric="PR-AUC",
            score=emg.get("pr_auc"),
            baseline=emg.get("baseline_persistence_pr_auc"),
            verdict="Harder",
        ),
    }


# ------------------------------------------------------------------ detail


def screening_operating_points(root: Path = REPORTS) -> dict | None:
    return _json(root, "model1_operating_points.json")


def complication_metrics(root: Path = REPORTS) -> dict:
    return _json(root, "peds_final.json") or {}


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
