# DengueShield

Dengue screening, complication risk, and 14-day outbreak early warning.

> **Research question.** Can machine learning provide reliable dengue screening from
> routine haematological data, while forecasting geographically localized dengue
> outbreaks from epidemiological and environmental data?

> **Conclusion.** Reliable dengue forecasting is achievable at geographic scale under
> strict temporal validation, whereas routine-CBC dengue screening remains substantially
> harder on real-world data. Geographic transfer also cannot be assumed, demonstrating
> the importance of local calibration and validation.

📄 **Full write-up: [REPORT.md](REPORT.md)** · 🔒 Immutable release: `frozen/v2_final/`
(72 artifacts, SHA-256 verified)

---

## Three distinct tasks — not one "dengue accuracy"

| | A · Dengue screening | B · Complication risk | C · Outbreak forecasting |
|---|---|---|---|
| Question | Does this patient appear likely to have dengue? | Among dengue patients, who needs closer monitoring? | Which districts will see an outbreak in 14 days? |
| Unit | one febrile patient | one dengue admission | one municipality-week |
| n | 1,511 | 303 | 66,000 test rows |
| Headline | ROC-AUC **0.681**, acc **76.2%** | Sens **90.5%**, spec **63.7%**, NPV **96.2%** | PR-AUC **0.961**, acc **96.7%**, recall **89.4%** |
| Verdict | Modest | Strong | Strong |

The ≥90% target was met where it is scientifically defensible (task C) and shown *not*
to be defensible for task A.

## Five findings worth reading the report for

1. **Three of four public dengue screening datasets are synthetic.** In one, body
   temperature is *completely disjoint* between classes (dengue 38.1–40.6 °C, non-dengue
   36.0–37.6 °C). Models trained on them score 0.99+ internally and collapse to AUC
   0.53–0.61 on real patients — worse than guessing.

2. **SHAP overstated the environmental contribution by an order of magnitude.** SHAP
   attributes ~50% of the model's reasoning to climate/land cover, but an ablation shows
   environment adds only **+0.006 PR-AUC** beyond historical incidence, and *alone*
   (0.597) is **worse than persistence** (0.760). Attribution ≠ incremental value.

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
| Leave-whole-states-out | mean 0.872 (range 0.602–0.972) |
| Shuffled-label control | collapses to 0.124 vs 0.144 chance ✅ |
| Persistence baseline | 0.615 — beaten decisively |

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

Four screens: **Dashboard** (three tasks + decision-support pathway) · **Patient
assessment** · **Geographic risk map** · **Model evidence** (validation, calibration,
explainability, generalization, data audit — including the weaknesses).

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
.venv/Scripts/python.exe freeze_v2.py             # immutable hashed release
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
