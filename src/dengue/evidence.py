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


def transfer(root: Path = REPORTS) -> pd.DataFrame | None:
    """Brazil -> Sri Lanka zero-shot transfer against local training."""
    return _csv(root, "srilanka_transfer.csv")


def head_to_head(root: Path = REPORTS) -> pd.DataFrame | None:
    return _csv(root, "model2_headtohead.csv")
