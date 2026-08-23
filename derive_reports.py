"""Rebuild the report artifacts that are pure functions of committed inputs.

Four files under reports/ were read by the app and written by nothing. Two of them
are exact functions of data already in the repo and are rebuilt here:

    cbc_population_medians.json    <- the screening cohort's feature medians
    model1_operating_points.json   <- reports/model1_oof_calibrated.npy + labels

Neither needs a retrain. The other two orphans are snapshots of a data state that
has moved on and are deliberately NOT rebuilt - see src/dengue/artifacts.py.

    python derive_reports.py            # verify against what is on disk
    python derive_reports.py --write    # rebuild and write
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

import numpy as np

from dengue import artifacts
from dengue.config import REPORTS
from dengue.datasets import load_hematology_1523

OOF = REPORTS / "model1_oof_calibrated.npy"


def build() -> dict:
    """Rebuild every derivable artifact. Returns {filename: contents}."""
    X, y, meta = load_hematology_1523()
    print(f"{meta['name']}  n={len(y)}  feats={X.shape[1]}")

    out = {"cbc_population_medians.json": artifacts.population_medians(X)}

    if OOF.exists():
        p = np.load(OOF)
        thr = artifacts.best_balanced_threshold(y, p)
        print(f"best-balanced threshold {thr:.4f}")
        out["model1_operating_points.json"] = artifacts.operating_points(y, p, thr)
    else:
        print(f"skipping model1_operating_points.json - {OOF.name} absent")

    return out


def main(write: bool) -> int:
    rebuilt = build()
    failed = False

    for name, contents in rebuilt.items():
        path = REPORTS / name
        diffs = artifacts.differences(contents, path)

        if write:
            artifacts.write_json(path, contents)
            print(f"  wrote {name}" + ("  (changed)" if diffs else "  (identical)"))
        elif not diffs:
            print(f"  OK   {name} reproduces exactly")
        else:
            failed = True
            print(f"  DIFF {name}")
            for d in diffs[:10]:
                print(f"         {d}")
            if len(diffs) > 10:
                print(f"         ... and {len(diffs) - 10} more")

    if failed:
        print("\nRebuilt artifacts disagree with the committed files.")
        print("These feed MANIFEST.json; investigate before overwriting them.")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the files, not just check")
    raise SystemExit(main(ap.parse_args().write))
