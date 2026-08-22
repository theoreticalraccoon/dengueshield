# DengueShield

Dengue screening, complication risk, and 14-day outbreak early warning.

> **Research question.** Can machine learning provide reliable dengue screening from
> routine haematological data, while forecasting geographically localized dengue
> outbreaks from epidemiological and environmental data?

> **Conclusion.** Reliable dengue forecasting is achievable at geographic scale under
> strict temporal validation, whereas routine-CBC dengue screening remains substantially
> harder on real-world data. Geographic transfer also cannot be assumed, demonstrating
> the importance of local calibration and validation. Crucially, that forecasting skill
> is *epidemiological momentum*: the model tracks outbreak trajectory far better than it
> detects emergence — and environmental data, near-worthless for the former, is
> materially useful for the latter.

📄 **Full write-up: [REPORT.md](REPORT.md)** · 🔒 Primary release `frozen/v2_final/`
(72 artifacts, SHA-256 verified untouched) · 🧪 Extension `experiments/emergence_v1/`

---

## Four distinct tasks — not one "dengue accuracy"

| | A · Screening | B · Complication risk | C · Outbreak continuation | D · Outbreak emergence |
|---|---|---|---|---|
| Question | Likely to have dengue? | Needs closer monitoring? | Will an outbreak persist 14 days? | Is a **new** outbreak starting? |
| n (test) | 1,511 | 303 | 66,000 | 53,805 |
| Headline | ROC-AUC **0.681**, acc **76.2%** | Sens **90.5%**, NPV **96.2%** | PR-AUC **0.961**, acc **96.7%** | PR-AUC **0.583**, **14.3× lift** |
| Baseline | 0.685 majority | — | 0.760 persistence | 0.382 persistence |
| Verdict | Modest | Strong | Strong | Harder — the honest early-warning task |

The ≥90% target was met where it is scientifically defensible (task C) and shown *not*
to be defensible for task A. Tasks C and D ask the same question of disjoint
populations: D excludes every district already in outbreak, which is exactly where C's
recall lives.

## Five findings worth reading the report for

1. **Three of four public dengue screening datasets are synthetic.** In one, body
   temperature is *completely disjoint* between classes (dengue 38.1–40.6 °C, non-dengue
   36.0–37.6 °C). Models trained on them score 0.99+ internally and collapse to AUC
   0.53–0.61 on real patients — worse than guessing.

2. **Environment is weak for continuation but matters for emergence — the central
   finding.** SHAP attributes ~50% of the continuation model's reasoning to climate and
   land cover, yet an ablation shows it adds only **+0.006 PR-AUC** there. Ask the same
   features to predict outbreak *emergence* and they add **+0.089** — a 12–15×
   difference, consistent across three configurations. Environmental data was not
   useless; it was being asked the wrong question. (Attribution ≠ incremental value.)

3. **The outbreak model tracks trajectory, not emergence.** Recall is 0.92 where
   baseline incidence is already high and **near zero** where it is low. False positives
   are **37× more likely** than true negatives to be municipalities already at epidemic
   level — momentum carried through subsiding outbreaks.

4. **It generalises spatially but not internationally.** Unseen municipalities score
   PR-AUC 0.955 vs 0.962 for seen ones — it did not merely memorise place identity. But
   zero-shot Brazil → Sri Lanka collapses to **0.449, below persistence (0.476)**.

5. **Calibration is earned, not assumed.** Isotonic calibration cuts the Brazil model's
   ECE from 0.0190 to **0.0052**, so its outputs are honestly *calibrated probabilities*.
   The Sri Lanka model is not — it predicts ~0.81 where outbreaks occur ~0.54 of the
   time — so the app labels it *predicted*, not *calibrated*.

## Validation battery (task C)

| Test | Result |
|---|---|
| Temporal holdout (locked 2024–25) | PR-AUC 0.961, acc 96.7%, recall 89.4% |
| Rolling-origin backtest, 7 years | mean PR-AUC 0.818, **every year >90% accuracy** |
| Horizon sweep (2/4/8 weeks) | 0.961 / 0.919 / 0.813 PR-AUC |
| Threshold sweep (50/100/300 per 100k) | 0.925 / 0.919 / 0.850 PR-AUC |
| **Spatial holdout (unseen municipalities)** | **PR-AUC 0.955** vs 0.962 seen |
| Leave-whole-states-out | mean 0.872; spread is mostly a prevalence artefact (lift inverts the ordering) |
| Shuffled-label control | collapses to 0.124 vs 0.144 chance ✅ |
| Persistence baseline | 0.615 — beaten decisively |
| **Temporal-leakage audit** | no lag column matched a future value ✅ |
| Lag-alignment repair | 8,107 duplicate rows fixed; headline moved −0.0006 PR-AUC |
| Reporting-delay stress (2 weeks) | 0.961 → 0.916 — robust to realistic delay |

## Run it

```bash
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv/Scripts/python.exe \
  numpy pandas scikit-learn xgboost lightgbm scipy matplotlib plotly \
  pyarrow duckdb requests joblib openpyxl shap streamlit
python -m uv pip install --python .venv/Scripts/python.exe torch \
  --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python.exe -m streamlit run app.py
```

Three screens, the two tools first:

- **Patient assessment** — quick 6-value entry or full CBC, plus complication risk.
  Optionally name the patient's district and it folds in local outbreak context.
- **Outbreak forecast** — autonomous district forecasts showing *both* continuation
  and emergence risk, on a map and in a ranked table.
- **About the models** — four sub-tabs (what it does · how well it works · where it
  fails · data & methods). Conclusions first, raw tables behind expanders.

## Reproduce the pipeline

```bash
.venv/Scripts/python.exe run_audit.py             # dataset integrity audit
.venv/Scripts/python.exe transfer_test.py         # synthetic -> real transfer
.venv/Scripts/python.exe train_model1.py          # screening model comparison
.venv/Scripts/python.exe finalize_model1.py       # nested CV + calibration
.venv/Scripts/python.exe run_ablation.py          # feature-group ablation
.venv/Scripts/python.exe finalize_peds.py         # complication model
.venv/Scripts/python.exe train_model2.py          # outbreak forecasting
.venv/Scripts/python.exe robustness_model2.py     # backtest + sweeps + controls
.venv/Scripts/python.exe train_lstm.py            # LSTM comparison
.venv/Scripts/python.exe shap_model2.py           # explainability
.venv/Scripts/python.exe spatial_and_ablation.py  # spatial holdout + info ablation
.venv/Scripts/python.exe calibration_and_errors.py# calibration + error analysis
.venv/Scripts/python.exe transfer_srilanka.py     # Brazil -> Sri Lanka
.venv/Scripts/python.exe finalize_srilanka.py     # production SL model + forecasts
.venv/Scripts/python.exe leakage_audit.py         # timestamp chain + reporting delay
.venv/Scripts/python.exe freeze_v2.py             # immutable hashed release
.venv/Scripts/python.exe experiments/emergence_v1/run_emergence.py
.venv/Scripts/python.exe experiments/emergence_v1/finalize_emergence_srilanka.py
```

## Data sources

- **Screening** — Mendeley [6fsrsk3mb8](https://data.mendeley.com/datasets/6fsrsk3mb8/1)
  (used); [xrsbyjs24t](https://data.mendeley.com/datasets/xrsbyjs24t/1),
  [673swz9tb4](https://data.mendeley.com/datasets/673swz9tb4/1),
  [zdtc3n6xv2](https://data.mendeley.com/datasets/zdtc3n6xv2/3) (audited, rejected).
- **Complications** — Zenodo [6476112](https://zenodo.org/records/6476112), 303
  paediatric dengue admissions.
- **Outbreak (Brazil)** — DATASET_MULTIMODAL_V8, Zenodo
  [22029053](https://zenodo.org/records/22029053): 4.7M municipality-weeks, 2010–2025.
- **Outbreak (Sri Lanka)** — [denguedatahub](https://github.com/thiyangt/denguedatahub)
  district-weekly 2006–2026; weather from [NASA POWER](https://power.larc.nasa.gov/).

---

**Clinical framing.** All outputs are screening and risk estimates, **not diagnoses**.
The system is decision support for prioritisation, never autonomous action. RT-PCR/NAAT
and antigen/serological testing remain the clinical diagnostic standard.
