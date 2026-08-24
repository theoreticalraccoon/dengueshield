# DengueShield

Dengue screening, complication risk, and 14-day outbreak early warning.

The question this started from: can routine blood work tell you who has dengue, and can
surveillance and weather data tell you where an outbreak is coming?

The short answer is that the second half works and the first half mostly doesn't. Outbreak
forecasting holds up under strict temporal validation. Screening from a routine CBC stays
modest on real patients no matter what you throw at it. And the forecasting skill is not
what it first appears — the model is very good at reading momentum in an outbreak already
underway, and much weaker at calling a new one.

Full write-up in [REPORT.md](REPORT.md). The primary release is frozen under
`frozen/v2_final/` (72 artifacts, SHA-256 verified).

---

## Four tasks, not one accuracy figure

These get quoted separately on purpose. They ask different questions of different
populations, and averaging them would hide the two that are hard.

| | A · Screening | B · Complication | C · Continuation | D · Emergence |
|---|---|---|---|---|
| Question | Does this patient have dengue? | Does this admission need closer monitoring? | Will an existing outbreak persist 14 days? | Is a new outbreak starting? |
| Asked of | one febrile patient | one dengue admission | any Sri Lankan district | quiet districts only |
| n (test) | 1,511 | 303 | 3,247 | 2,678 |
| Headline | ROC-AUC 0.690 | sens 90.5%, NPV 96.2% | PR-AUC 0.725 | PR-AUC 0.405, recall 73% |
| Baseline | 0.685 majority class | — | 0.469 persistence | 0.250 persistence |
| Verdict | modest | strong | strong | the hard one |

Every column above is a model the app deploys. C and D are the same phenomenon asked of
disjoint populations: D excludes every district already in outbreak, which is precisely
where C's recall comes from.

**Both C and D have a second, larger number that is not this one.** The Brazil
municipality panel scores PR-AUC **0.960** for continuation (against a 0.758 persistence
baseline) and **0.583** for emergence, because it has thousands of municipalities against
Sri Lanka's 26 districts. Neither is deployed. The C row above was sourced from the Brazil
artifact until 2026-08-24 and sat beside Sri Lankan emergence in the same table — the
mistake D had already been fixed for.

### A note on accuracy for task D

Predicting "no new outbreak" every week scores **93.5%** accuracy on the test set and
catches nothing at all, because only 6.5% of district-weeks see an outbreak begin. The
deployed model scores 79.2% and catches 74% of them. Judge emergence on recall, not
accuracy — the higher accuracy number is the useless one.

## What the work turned up

**Three of four public dengue screening datasets are synthetic.** In one, body temperature
is completely disjoint between classes — dengue 38.1–40.6 °C, non-dengue 36.0–37.6 °C, no
overlap at all. In another, `platelet < 150,000` reproduces the label 98.97% of the time.
Models trained on them score above 0.99 internally and then fall to 0.53–0.61 on real
patients. `run_audit.py` now rejects any dataset with a single feature above 0.95 AUC or
disjoint class ranges.

**Attribution is not the same as value.** SHAP hands roughly half the continuation model's
reasoning to climate and land cover. Remove those features and the model loses 0.006
PR-AUC. Environment on its own scores 0.595, worse than plain persistence at 0.758. The
lesson generalises: always pair an attribution plot with a leave-group-out ablation.

**Environment matters for emergence, not continuation.** Same features, different question:
+0.006 PR-AUC for continuation, +0.089 for emergence. That 15× gap held across three
configurations and is the most interesting result here. Environmental data was not weak,
it was being asked the wrong question.

**The model reads momentum.** Recall is 0.92 where incidence is already high and close to
zero where it is low. Over half of false alarms are municipalities already at epidemic
level — 37.6× the rate among true negatives. It is tracking outbreaks that are subsiding,
not spotting ones that are starting.

**Geography transfers, countries don't.** Held-out municipalities the model has never seen
score 0.954 against 0.961 for seen ones, so it did not simply memorise place identity. But
Brazil → Sri Lanka zero-shot collapses to 0.449, below Sri Lanka's own persistence
baseline of 0.476. Training locally recovers it to 0.708.

**Calibration has to be earned.** Isotonic regression takes the Brazil model's expected
calibration error from 0.019 to 0.005. The Sri Lanka models are trained with
`scale_pos_weight`, which fixes the ranking and wrecks the scale, so they needed the same
treatment: fitted out of fold across the development years, their test ECE falls from
0.140 to 0.023 (continuation) and 0.292 to 0.039 (emergence). It costs a little PR-AUC —
isotonic merges scores it cannot separate into ties, and PR-AUC charges for ties — which
is why discrimination is reported on the raw score and calibration on the calibrated one.

## How task C was checked

Every row below is the **Brazil** municipality panel — that is where the validation
battery was run, because 66,660 test rows across thousands of municipalities support
questions the 26-district Sri Lankan panel cannot answer. It is not the deployed model;
see the note under the task table.

| Test | Result |
|---|---|
| Temporal holdout, locked 2024–25 | PR-AUC 0.960, accuracy 96.6%, recall 90.5% |
| Rolling-origin backtest, 7 years | mean 0.907, worst year 0.846 |
| Spatial holdout, unseen municipalities | 0.954 vs 0.961 seen |
| Leave-whole-states-out | mean 0.872; the spread is mostly prevalence |
| Horizon sweep, 2 / 4 / 8 weeks | 0.961 / 0.918 / 0.799 |
| Threshold sweep, 50 / 100 / 300 per 100k | 0.960 / 0.961 / 0.936 |
| Shuffled-label control | falls to 0.135 against a 0.145 chance rate |
| Temporal-leakage audit | no lag column matched a future value |
| Lag-alignment repair | 8,107 duplicate rows fixed; headline moved −0.0006 |
| Reporting-delay stress, 2 weeks | 0.961 → 0.916 |

The test years were locked before any of this and never tuned against. Thresholds come
from the 2022–23 validation block only.

## Running it

```bash
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv/Scripts/python.exe \
  numpy pandas scikit-learn xgboost lightgbm scipy matplotlib plotly \
  pyarrow duckdb requests joblib openpyxl shap streamlit pytest
python -m uv pip install --python .venv/Scripts/python.exe torch \
  --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python.exe -m streamlit run app.py
```

Three screens:

- **Patient assessment** — six values or a full CBC, plus complication risk for a patient
  already known to have dengue. Name a district and it folds in the local outbreak picture.
- **Outbreak forecast** — every district sorted into outbreak now, outbreak likely, clear,
  or not assessable, with a map and a ranked table.
- **About the models** — what it does, how well it works, where it fails, and what it was
  built on. Every number on that screen is read from `reports/`, not typed in.

Tests: `.venv/Scripts/python.exe -m pytest tests/ -q`

## Layout

```
app.py                  the Streamlit app
src/dengue/             the package everything imports
  config.py             paths, horizons, thresholds, split years, seed
  risk.py               the district risk rule
  predictor.py          one interface over all four model bundles
  evidence.py           named access to reports/
  emergence.py          emergence labelling and eligibility
  experiment.py         one configured Brazil run
tests/                  pytest
docs/adr/               decisions that shouldn't be re-argued
frozen/v2_final/        the immutable release
reports/                every metric, forecast and audit output
```

## Rebuilding the pipeline

Order matters — later scripts read what earlier ones write.

```bash
.venv/Scripts/python.exe run_audit.py              # dataset integrity gate
.venv/Scripts/python.exe transfer_test.py          # synthetic -> real transfer
.venv/Scripts/python.exe train_model1.py           # screening model comparison
.venv/Scripts/python.exe finalize_model1.py        # nested CV + calibration
.venv/Scripts/python.exe derive_reports.py --write # artifacts derived from the above
.venv/Scripts/python.exe run_ablation.py           # feature-group ablation
.venv/Scripts/python.exe finalize_peds.py          # complication model
.venv/Scripts/python.exe train_model2.py           # outbreak forecasting
.venv/Scripts/python.exe train_lstm.py             # LSTM comparison
.venv/Scripts/python.exe robustness_model2.py      # backtest, sweeps, controls
.venv/Scripts/python.exe shap_model2.py            # explainability
.venv/Scripts/python.exe spatial_and_ablation.py   # spatial holdout + info ablation
.venv/Scripts/python.exe calibration_and_errors.py # calibration + error analysis
.venv/Scripts/python.exe leakage_audit.py          # timestamp chain + delay stress
.venv/Scripts/python.exe transfer_srilanka.py      # Brazil -> Sri Lanka
.venv/Scripts/python.exe finalize_srilanka.py      # deployed continuation model
.venv/Scripts/python.exe finalize_emergence.py     # deployed emergence model
```

The Brazil parquet is 565 MB and is not in the repo; `fetch_full.py` pulls it from Zenodo
and can resume. Nothing the app or the weekly refresh does needs it.

## Keeping the data current

```bash
.venv/Scripts/python.exe refresh_data.py --check   # exit 0 current, 10 stale
.venv/Scripts/python.exe refresh_data.py           # apply and regenerate forecasts
```

Two sources feed this. denguedatahub is tidy and reliable but lags by months. The
Epidemiology Unit's Weekly Epidemiological Reports are current to within about six weeks
but arrive as sideways-typeset PDFs. The parser is re-checked on every run against weeks
denguedatahub already covers — currently 100% exact across 492 district-weeks — and the
whole WER source is discarded if agreement drops below 95%.

A GitHub Action runs this on Tuesdays and commits only when something changed.

## Data sources

- **Screening** — Mendeley [6fsrsk3mb8](https://data.mendeley.com/datasets/6fsrsk3mb8/1)
  (used); [xrsbyjs24t](https://data.mendeley.com/datasets/xrsbyjs24t/1),
  [673swz9tb4](https://data.mendeley.com/datasets/673swz9tb4/1),
  [zdtc3n6xv2](https://data.mendeley.com/datasets/zdtc3n6xv2/3) (audited and rejected).
- **Complications** — Zenodo [6476112](https://zenodo.org/records/6476112), 303 paediatric
  dengue admissions.
- **Outbreak, Brazil** — DATASET_MULTIMODAL_V8, Zenodo
  [22029053](https://zenodo.org/records/22029053), 4.7M municipality-weeks, 2010–2025.
- **Outbreak, Sri Lanka** — [denguedatahub](https://github.com/thiyangt/denguedatahub)
  district-weekly 2006–2026, weather from [NASA POWER](https://power.larc.nasa.gov/).

## Limitations

Nothing here has been validated prospectively against a live surveillance feed. The
emergence model resists improvement: spatial neighbour features, re-tuning, ensembling,
extra training rows and a discrete-time hazard reformulation all fail a paired
rolling-origin test, and Brazil transfer — zero-shot or fine-tuned — scores below training
locally. `docs/adr/0003` records each so they are not re-run. The symptom ablation was run on the complication
cohort rather than the screening cohort, because no audited real dataset has both
dengue-negative controls and symptom records. Three report artifacts predate the current
code and cannot be regenerated; `src/dengue/artifacts.py` records which and why.

---

Everything here is a screening or risk estimate, not a diagnosis. It is decision support
for prioritisation and never a trigger for automatic action. RT-PCR/NAAT and antigen
testing remain the diagnostic standard.
