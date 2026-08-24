# DengueShield — project context

Dengue screening + outbreak early warning. Four models across patient and population
levels. Repo: <https://github.com/theoreticalraccoon/dengueshield> (public).

## Environment

The project folder is `DengueShield`. If a path in an older note says `Dengue_Project`,
it predates the rename. The venv survives a folder rename - `pyvenv.cfg` only points at
the base interpreter - but the `.venv/Scripts/*.exe` console shims embed the old
absolute path, so always invoke tools as `python.exe -m <tool>` rather than the shim.

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
| C Continuation | Will an outbreak persist 14d? | PR-AUC **0.960**, vs persistence 0.758 |
| D Emergence | Will a *new* outbreak begin? | Brazil **0.583** (14.3× lift); Sri Lanka **0.408**, recall **74%** |

**D is two different numbers and they must not be conflated.** 0.583 / 14.3× is the
*Brazil* exploratory experiment in `experiments/emergence_v1/emergence_results.csv`
(`model_combined`, h4_inc100) and is what findings 2–3 below rest on. The model the
app actually deploys is the *Sri Lanka* emergence model, PR-AUC **0.408** against a
persistence baseline of 0.250. The About screen quoted the Brazil figure beside Sri
Lanka forecasts until the numbers were sourced from artifacts rather than typed in.

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
   (0.595) is worse than persistence (0.758). Always pair attribution with a
   leave-group-out ablation.
3. **Environment matters for emergence, not continuation** — +0.089 vs +0.006, a 15×
   difference, consistent across three configs. The central scientific result.
4. **Zero-shot geographic transfer fails.** Brazil→Sri Lanka drops 0.910 → **0.449**,
   below persistence (0.476). Local training reaches 0.708.
5. **The model tracks trajectory, not emergence.** Recall 0.92 at high baseline
   incidence, ~0 at low. 53.3% of false alarms are districts already at epidemic level
   (37.6× the true-negative rate).

## Layout

```
app.py                      Streamlit, 3 screens (Patient / Outbreak / About)
refresh_data.py             weekly data refresh; --check exits 0 current / 10 stale
finalize_srilanka.py        continuation model  -> srilanka_outbreak.joblib
finalize_emergence.py       emergence model     -> srilanka_emergence.joblib + dual risk
derive_reports.py           rebuilds the report artifacts that are pure functions
src/dengue/
  config.py                 paths + every constant: horizons, thresholds, split years, SEED
  risk.py                   the district risk rule (C vs D on disjoint populations)
  predictor.py              one interface over all four model bundles
  evidence.py               named access to reports/ (the app reads nothing raw)
  artifacts.py              artifact -> producer inventory; derived-artifact rebuilders
  experiment.py             one configured Brazil run: split, weights, params
  emergence.py              emergence label + eligibility
  metrics.py                threshold selection (sensitivity / F1)
  audit.py                  dataset integrity gate
  model2_outbreak.py        Brazil panel; recompute_dynamics() rebuilds lags date-exactly
  srilanka.py               SL panel + weather; add_base_features / add_features
  wer.py                    Epidemiology Unit PDF parser (see gotchas)
tests/                      pytest; `python -m pytest tests/ -q`
docs/adr/                   decisions that should not be re-litigated
experiments/emergence_v1/   exploratory ONLY (run_emergence.py). The production
                            emergence model was promoted out of here.
frozen/v2_final/            immutable release, 72 artifacts, SHA-256 verified
reports/                    every metric, forecast and audit output
```

**Read `CONTEXT.md` for the domain vocabulary** — district, eligible, headline risk,
assessable, triage. The modules are named after those terms.

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
.venv/Scripts/python.exe refresh_data.py --check   # exit 0 current / 10 stale
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
  Verified by `tests/test_app_smoke.py`, which runs the real script via Streamlit's
  `AppTest` — it is not a manual check any more.
- **Everything Brazil is horizon 2 as of the 2026-08-23 retrain.** `model2_outbreak`
  and the LSTM were previously saved at horizon 4 while every published report used 2.
  See `docs/adr/0002-brazil-horizon.md`.
- **Four report artifacts still have no producer** — `model2_headtohead.csv/.json`,
  `srilanka_calibration.json`, `model2_state_folds_with_events.csv`. They are
  snapshots of data states that have moved on, so they cannot be regenerated;
  `src/dengue/artifacts.py` records that rather than faking a rebuild. The other two
  orphans now have one (`derive_reports.py`, verified byte-exact).
- `refresh_situation.py` (NDCU) is now wired into `refresh_data.py` as step 4/5 and
  fails soft - the forecast path does not depend on it. The app renders its freshness
  stamp; the situation *table* itself is still not rendered anywhere. Decide whether
  to surface it or drop the source.
- **Emergence accuracy is a trap.** Predicting "no new outbreak" forever scores
  **93.5%** on the locked test set and catches nothing; the deployed model scores
  79.2% *because* it takes risks. Judge it on recall and PR-AUC, never accuracy.
  `EMERGENCE_SENSITIVITY_TARGET` in `config.py` picks the operating point - 0.70
  (default) catches 74% of emerging outbreaks at 5.3 flagged districts per week;
  0.90 catches 91% at 13.9 of 26 districts, which is most of the country.

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
