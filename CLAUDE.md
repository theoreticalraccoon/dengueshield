# DengueShield — project context

Dengue screening + outbreak early warning. Four models across patient and population
levels. Repo: <https://github.com/theoreticalraccoon/dengueshield> (public).

## Environment

Python **3.11** in `.venv` — 3.14 has no wheels for torch/xgboost, do not use it.

```bash
.venv/Scripts/python.exe <script>.py        # always this interpreter
/c/Python314/python -m uv pip install --python .venv/Scripts/python.exe <pkg>
.venv/Scripts/python.exe -m streamlit run app.py
```

`uv` lives under the 3.14 install, not the venv — `python -m uv` from inside the venv
fails. IDE diagnostics claiming `lightgbm`/`streamlit` are missing are false positives
(the IDE resolves against system 3.14).

## The four tasks — never merge into one "accuracy"

| | Question | Score |
|---|---|---|
| A Screening | Does this patient have dengue? | ROC-AUC **0.681**, acc 76.2% |
| B Complication | Needs closer monitoring? | ROC-AUC **0.874**, NPV 0.962 |
| C Continuation | Will an outbreak persist 14d? | PR-AUC **0.961**, acc 96.7% |
| D Emergence | Will a *new* outbreak begin? | PR-AUC **0.583**, 14.3× lift |

C and D are the same phenomenon asked of disjoint populations: D excludes every
district already in outbreak, which is exactly where C's recall lives.

## Findings that shaped the project

1. **Three of four public dengue screening datasets are synthetic.** One has body
   temperature completely disjoint between classes; another's label is reproduced at
   98.97% by `platelet < 150,000`. Models trained on them collapse to AUC 0.53–0.61 on
   real patients. `run_audit.py` blocks any dataset with a single feature at AUC ≥ 0.95
   or disjoint class ranges.
2. **SHAP attribution ≠ incremental value.** SHAP credits ~50% of the continuation
   model to environment; removing it costs **+0.006** PR-AUC and environment alone
   (0.597) is worse than persistence (0.760). Always pair attribution with a
   leave-group-out ablation.
3. **Environment matters for emergence, not continuation** — +0.089 vs +0.006, a 15×
   difference, consistent across three configs. The central scientific result.
4. **Zero-shot geographic transfer fails.** Brazil→Sri Lanka drops 0.910 → **0.449**,
   below persistence (0.476). Local training reaches 0.708.
5. **The model tracks trajectory, not emergence.** Recall 0.92 at high baseline
   incidence, ~0 at low. 54.5% of false alarms are districts already at epidemic level
   (37× the true-negative rate).

## Layout

```
app.py                      Streamlit, 3 screens (Patient / Outbreak / About)
refresh_data.py             weekly data refresh, --check for dry run
src/dengue/
  audit.py                  dataset integrity gate
  model2_outbreak.py        Brazil panel; recompute_dynamics() rebuilds lags date-exactly
  srilanka.py               SL district-week panel + NASA POWER weather
  wer.py                    Epidemiology Unit PDF parser (see gotchas)
experiments/emergence_v1/   exploratory emergence model — NOT part of frozen v2
frozen/v2_final/            immutable release, 72 artifacts, SHA-256 verified
reports/                    every metric, forecast and audit output
```

## Gotchas that cost real time

- **Bash heredocs break on apostrophes** in this environment. Writing a Python file with
  `<<'EOF'` fails with "unexpected EOF" if the content has `'` in a docstring. Use the
  Write tool for anything non-trivial.
- **`.gitignore` negation needs `data/raw/*`, not `data/raw/`.** Git does not descend
  into an ignored *directory*, so `!data/raw/foo.csv` silently never applies. This
  already bit once and left the CI refresh without its inputs.
- **The Brazil panel has 8,107 duplicate (municipality, week) rows** whose pre-computed
  lags are positional and therefore misaligned. `recompute_dynamics()` rebuilds every
  lag by joining on the calendar date. Impact was −0.0006 PR-AUC — it degraded rather
  than inflated, but do not revert it.
- **WER PDFs are typeset sideways.** Each glyph is individually placed; reconstruct
  vertical runs bottom-to-top. Two bugs already fixed: the date regex must anchor on the
  Table 1 caption (the *cover* date is one week later) and Badulla's name run picks up
  stray header digits, so numbers come only from the numeric run.
- **Loading two full Brazil panels at once exhausts memory** and the process dies with
  exit code 0, printing nothing. `del` the first one.
- Streamlit CSS targets `data-testid` hooks — stable in practice but not a public API.

## Data currency

Surveillance is refreshed from two sources: denguedatahub (structured, lags months) and
the Epidemiology Unit's WER PDFs (current to ~6 weeks). The PDF parser is validated on
every run against weeks denguedatahub already covers — **100% exact across 492
district-weeks** — and the whole WER source is discarded if agreement drops below 95%.

`.github/workflows/refresh-data.yml` runs Tuesdays, commits only when something changed.
Sri Lanka inputs (3.5 MB) ship with the repo so CI works on a clean checkout; the 1.2 GB
Brazil parquet does not and is not needed at runtime.

```bash
.venv/Scripts/python.exe refresh_data.py --check   # is anything newer?
.venv/Scripts/python.exe refresh_data.py           # apply + regenerate forecasts
```

## Conventions

- **PR-AUC and recall lead, never accuracy** — 86.5% of municipality-weeks are negative,
  so accuracy flatters everything. Always quote the trivial baseline alongside.
- **Clinical models are tuned for sensitivity**, not accuracy or F1.
- **Serology (NS1/IgM/IgG) is ground truth for the label, never a predictor.**
- Splits are strictly temporal: train ≤2021, validate 2022–23, test 2024–25. The test
  set is **locked** — thresholds come from validation only.
- Report what failed. The app's About screen exists to show the weaknesses.

## State / open items

- App verified: 3 screens render, both prediction buttons work, `ruff` clean.
- **Not yet deployed to Streamlit Cloud.** Needs a one-time browser sign-in at
  <https://share.streamlit.io/deploy?repository=theoreticalraccoon/dengueshield&branch=main&mainModule=app.py>.
  After that every `git push` auto-rebuilds; only `.streamlit/config.toml` changes may
  need a manual Reboot app.
- Sri Lanka forecasts are **not calibrated** (ECE 0.046, overconfident in the top band).
  The app says "predicted", not "calibrated", deliberately.
- The symptom/fever-day ablation was run on the *complication* cohort, not the screening
  cohort — no audited real dataset has both dengue-negative controls and symptoms. Keep
  that caveat in any write-up.
- Nothing has been validated prospectively against a live surveillance feed.
