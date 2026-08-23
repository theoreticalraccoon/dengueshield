"""Download the Brazil multimodal panel from Zenodo, resumably.

The 565 MB parquet behind Model 2 does not ship with the repo, so this is the
first step on a clean checkout if you intend to retrain or re-run any Brazil
analysis. The app and the weekly refresh do NOT need it - they only use the
Sri Lanka inputs, which are committed.

Resumable on purpose: the transfer routinely drops part-way, so this issues a
ranged request from wherever the local file left off and retries.

    python fetch_full.py     # safe to re-run; stops once the file is complete
"""

import pathlib
import time

import requests

u = "https://zenodo.org/api/records/22029053/files/dataset_multimodal_v8.parquet/content"
p = pathlib.Path("data/raw/dataset_multimodal_v8.parquet")
TARGET = 564861145
for attempt in range(40):
    have = p.stat().st_size if p.exists() else 0
    if have >= TARGET:
        print("COMPLETE", have)
        break
    try:
        with requests.get(
            u, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(30, 180)
        ) as r:
            if r.status_code != 206:
                print("no-resume status", r.status_code)
                time.sleep(5)
                continue
            with open(p, "ab") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    except Exception as e:
        print(f"att{attempt} {type(e).__name__} @ {p.stat().st_size / 1e6:.0f}MB", flush=True)
        time.sleep(3)
print("FINAL", p.stat().st_size)
