"""Create frozen/v2_final/ - the immutable release for the final report.

Captures everything needed to reproduce or audit the reported numbers: models,
preprocessing pipelines, feature lists, thresholds, hyperparameters, seeds,
package versions, every metric, SHAP output, and the Sri Lanka transfer results.
Every artifact is SHA-256 hashed into MANIFEST.json.

After this runs, the models must not be modified for the final report.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "frozen" / "v2_final"
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "models").mkdir(parents=True)
(OUT / "reports").mkdir(parents=True)
(OUT / "code").mkdir(parents=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ copy artifacts
MODELS = [
    "model1_screening.joblib",
    "peds_complications.joblib",
    "model2_outbreak.joblib",
    "srilanka_outbreak.joblib",
    "model2_lstm.pt",
    "model2_lstm_scalers.joblib",
]
for m in MODELS:
    src = ROOT / "models" / m
    if src.exists():
        shutil.copy2(src, OUT / "models" / m)

for r in sorted((ROOT / "reports").glob("*")):
    if r.is_file():
        shutil.copy2(r, OUT / "reports" / r.name)

CODE = [
    "app.py",
    "run_audit.py",
    "transfer_test.py",
    "train_model1.py",
    "finalize_model1.py",
    "run_ablation.py",
    "finalize_peds.py",
    "train_model2.py",
    "robustness_model2.py",
    "train_lstm.py",
    "shap_model2.py",
    "spatial_and_ablation.py",
    "calibration_and_errors.py",
    "transfer_srilanka.py",
    "finalize_srilanka.py",
    "freeze_v2.py",
]
for c in CODE:
    if (ROOT / c).exists():
        shutil.copy2(ROOT / c, OUT / "code" / c)
shutil.copytree(
    ROOT / "src" / "dengue", OUT / "code" / "dengue", ignore=shutil.ignore_patterns("__pycache__")
)

# written deliverables travel with the release
for doc in ["REPORT.md", "README.md"]:
    if (ROOT / doc).exists():
        shutil.copy2(ROOT / doc, OUT / doc)


# --------------------------------------------------------------- model provenance
def describe(path: Path) -> dict:
    """Pull feature list, threshold and hyperparameters out of a saved bundle."""
    try:
        b = joblib.load(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if not isinstance(b, dict):
        return {"type": type(b).__name__}
    d = {"keys": sorted(b.keys())}
    for k in ("threshold", "threshold_sens90", "horizon", "outbreak_inc"):
        if k in b:
            d[k] = b[k]
    if "features" in b:
        d["n_features"] = len(b["features"])
        d["features"] = list(b["features"])
    est = b.get("model")
    try:
        if hasattr(est, "get_params"):
            d["hyperparameters"] = {k: str(v) for k, v in est.get_params(deep=False).items()}
        elif hasattr(est, "params"):
            d["hyperparameters"] = {k: str(v) for k, v in est.params.items()}
        if hasattr(est, "num_trees"):
            d["n_trees"] = est.num_trees()
    except Exception:
        pass
    if "metrics" in b:
        d["metrics"] = b["metrics"]
    if "metrics_nested_cv" in b:
        d["metrics_nested_cv"] = b["metrics_nested_cv"]
    return d


provenance = {m: describe(OUT / "models" / m) for m in MODELS if (OUT / "models" / m).exists()}


def load_json(name):
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


def load_csv(name):
    p = ROOT / "reports" / name
    return pd.read_csv(p).to_dict("records") if p.exists() else None


# ------------------------------------------------------------------- package env
try:
    pkgs = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120
    ).stdout.splitlines()
except Exception:
    pkgs = []
if not pkgs:
    pkgs = []
    for mod in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "torch",
        "shap",
        "streamlit",
        "duckdb",
        "plotly",
    ]:
        try:
            import importlib

            name = {"scikit-learn": "sklearn"}.get(mod, mod)
            pkgs.append(f"{mod}=={importlib.import_module(name).__version__}")
        except Exception:
            pass

manifest = {
    "release": "v2_final",
    "frozen_at": "2026-08-22",
    "status": "IMMUTABLE - models must not be modified for the final report",
    "environment": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
    },
    "random_seeds": {
        "global": 42,
        "model1_nested_cv": 42,
        "model2_lightgbm": 42,
        "lstm_torch": 42,
        "spatial_holdout_split": 42,
    },
    "three_distinct_tasks": {
        "A_dengue_screening": {
            "question": "Does this patient appear likely to have dengue?",
            "population": "1,511 febrile patients, 19 CBC variables (Bangladesh)",
            "roc_auc": 0.6811,
            "pr_auc": 0.7790,
            "accuracy": 0.7617,
            "sensitivity": 0.9729,
            "specificity": 0.3025,
            "brier": 0.1804,
            "trivial_baseline_accuracy": 0.6850,
            "note": "Modest. A CBC alone is not a reliable dengue diagnostic.",
        },
        "B_complication_screening": {
            "question": "Among dengue patients, who needs closer monitoring?",
            "population": "303 paediatric dengue admissions",
            "roc_auc": 0.8741,
            "pr_auc": 0.7311,
            "sensitivity": 0.9048,
            "specificity": 0.6375,
            "npv": 0.9623,
            "note": "Tuned for sensitivity; safe to rule out, not to rule in.",
        },
        "C_outbreak_forecasting": {
            "question": "Which districts will see an outbreak in the next 14 days?",
            "population": "699 Brazilian municipalities, 66,000 test municipality-weeks",
            "pr_auc": 0.9614,
            "accuracy": 0.9673,
            "recall": 0.8942,
            "trivial_baseline_accuracy": 0.8544,
            "note": "Do NOT combine with A or B into a single 'dengue accuracy'.",
        },
    },
    "model_provenance": provenance,
    "metrics": {
        "model1_operating_points": load_json("model1_operating_points.json"),
        "model1_final": load_json("model1_final.json"),
        "peds_final": load_json("peds_final.json"),
        "model2_headtohead": load_csv("model2_headtohead.csv"),
        "model2_robustness": load_json("model2_robustness.json"),
        "model2_calibration_errors": load_json("model2_calibration_errors.json"),
        "model2_spatial_ablation": load_json("model2_spatial_ablation.json"),
        "srilanka_transfer": load_json("srilanka_transfer.json"),
        "shap": load_json("model2_shap.json"),
        "dataset_audit": load_csv("dataset_audit.csv"),
        "ablation_peds": load_csv("ablation_peds.csv"),
    },
    "key_findings": [
        "Three of four public dengue screening datasets are synthetic or rule-labelled; "
        "models trained on them collapse to AUC 0.53-0.61 on real patients.",
        "Outbreak forecasting at 14 days: PR-AUC 0.961, accuracy 96.7%, recall 89.4%, "
        "beating persistence (0.760) under strict temporal holdout.",
        "Spatial holdout: unseen municipalities score PR-AUC 0.955 vs 0.962 for seen "
        "ones - the model generalises rather than memorising place identity. "
        "Whole-state holdout is harder and more variable (mean 0.872, range 0.602-0.972).",
        "Information ablation: environmental variables add only +0.006 PR-AUC beyond "
        "historical incidence, and alone (0.597) are WORSE than persistence (0.760). "
        "SHAP attribution overstates their incremental value because they correlate "
        "with recent incidence.",
        "Error analysis: recall is 0.92 where baseline incidence is already high but "
        "near zero where it is low - the model tracks epidemic trajectory rather than "
        "detecting emergence from a low base.",
        "False positives are 37x more likely than true negatives to be municipalities "
        "already at epidemic level - the model carries epidemiological momentum through "
        "outbreaks that are subsiding.",
        "Isotonic calibration cuts ECE from 0.0190 to 0.0052, so displayed "
        "probabilities are trustworthy and may be labelled 'calibrated'.",
        "Brazil->Sri Lanka zero-shot transfer FAILS (PR-AUC 0.449), worse than "
        "persistence (0.476). Local training reaches 0.708.",
    ],
}

(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))

hashes = {}
for f in sorted(OUT.rglob("*")):
    if f.is_file() and f.name != "HASHES.json":
        hashes[str(f.relative_to(OUT)).replace("\\", "/")] = {
            "sha256": sha256(f),
            "bytes": f.stat().st_size,
        }
(OUT / "HASHES.json").write_text(json.dumps(hashes, indent=2))

total = sum(v["bytes"] for v in hashes.values())
print(f"FROZEN v2_final: {len(hashes)} artifacts, {total / 1e6:.1f} MB")
print(f"  models   : {len(list((OUT / 'models').glob('*')))}")
print(f"  reports  : {len(list((OUT / 'reports').glob('*')))}")
print(f"  code     : {len(list((OUT / 'code').rglob('*.py')))} python files")
print(f"  manifest : {(OUT / 'MANIFEST.json').stat().st_size / 1024:.1f} KB")
print("  hashes   : SHA-256 for every artifact -> HASHES.json")
