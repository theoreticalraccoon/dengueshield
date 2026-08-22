# DengueShield — a dengue screening and outbreak early-warning system

**Release:** `frozen/v2_final/` (immutable, SHA-256 hashed, 69 artifacts)

---

## 1. Problem

Dengue creates two operationally distinct problems that are usually conflated.

At the **individual level**, dengue presents as an undifferentiated febrile illness.
Early in the course it is clinically indistinguishable from many other infections, and
confirmatory testing (RT-PCR/NAAT, NS1 antigen, serology) is not always available at
the point of first contact. Clinicians must decide who to test and who to watch.

At the **population level**, dengue is spatially heterogeneous and episodic. Outbreaks
concentrate in particular districts in particular weeks, driven by mosquito ecology,
climate and human density. Vector control, testing capacity and hospital preparedness
are finite and must be positioned in advance.

These require different data, different models, and — critically — different success
metrics.

## 2. Research question

> Can machine learning provide reliable dengue screening from routine haematological
> data, while forecasting geographically localized dengue outbreaks from
> epidemiological and environmental data?

## 3. The three tasks are not one task

The system contains **three** prediction tasks. Reporting a single "dengue accuracy"
across them would be meaningless.

| | A · Dengue screening | B · Complication risk | C · Outbreak forecasting |
|---|---|---|---|
| Question | Does this patient appear likely to have dengue? | Among dengue patients, who needs closer monitoring? | Which districts will see an outbreak in 14 days? |
| Unit | one febrile patient | one dengue admission | one municipality-week |
| n | 1,511 | 303 | 66,000 test rows |
| Headline | ROC-AUC **0.681**, acc **76.2%** | Sens **90.5%**, spec **63.7%**, NPV **96.2%** | PR-AUC **0.961**, acc **96.7%**, recall **89.4%** |
| Verdict | Modest | Strong | Strong |

## 4. Data

**Clinical.** Four public dengue screening datasets were collected (Mendeley
6fsrsk3mb8, xrsbyjs24t, 673swz9tb4, zdtc3n6xv2) plus a paediatric dengue cohort
(Zenodo 6476112, 303 admissions with symptoms, fever duration, CBC and organ labs).

**Epidemiological.** DATASET_MULTIMODAL_V8 (Zenodo 22029053): 4.7M Brazilian
municipality-weeks, 2010–2025, 87 variables integrating epidemiology, meteorology,
MODIS remote sensing, ENSO indices, topography, land cover, demography and sanitation.
Restricted to the 699 municipalities with population ≥50,000, where surveillance counts
are stable.

**Sri Lanka.** denguedatahub district-weekly surveillance (26 districts × 1,012 weeks,
2006–2026), joined to NASA POWER daily reanalysis at district centroids. A single
reanalysis product is used for every district — mixing sources would inject a source
effect the model could mistake for geography.

## 5. Data-quality audit — the first major finding

Every dataset is audited before modelling. **The rule:** any single feature with
AUC ≥ 0.95, or disjoint class-conditional ranges, blocks the dataset.

| Dataset | n | Max single-feature AUC | Verdict |
|---|---|---|---|
| `hematology_1523` (6fsrsk3mb8) | 1,511 | 0.624 | **PASS** |
| `vitals_1003` (xrsbyjs24t) | 989 | 0.9998 (`WBC Count`) | REJECT |
| `bd_structured` (673swz9tb4) | 1,018 | 0.9926 (`WBC`) | REJECT |
| `bd_comprehensive` (zdtc3n6xv2) | 1,000 | 1.0000 (`Body_Temperature`, `IgG`) | REJECT |
| `peds_complications` (Zenodo 6476112) | 303 | 0.753 | **PASS** |

**Three of four screening datasets failed.** In `bd_comprehensive`, body temperature is
*completely disjoint* between classes — dengue-positive patients span 38.1–40.6 °C and
negatives 36.0–37.6 °C, with zero overlap. Real patients do not separate like that. In
`vitals_1003` the label is reproduced at 98.97% accuracy by the single rule
`platelet < 150,000`.

### They do not survive contact with real patients

Trained on each rejected dataset, tested on the one real cohort (shared features: age,
sex, platelet, WBC):

| Trained on | Internal AUC | → AUC on real data | → Accuracy on real data |
|---|---|---|---|
| `bd_structured` | 0.989 | **0.526** | 0.588 |
| `bd_comprehensive` | 1.000 | **0.607** | 0.332 (sensitivity 0.035) |
| `vitals_1003` | 1.000 | **0.564** | 0.413 |

The real-data majority-class baseline is 0.685 — **every synthetic-trained model
performs worse than guessing.** They transfer perfectly to *each other* (AUC 0.95–1.00)
because they share a generative rule, and not at all to reality.

This is the likely origin of the 95–99% dengue-screening accuracies widely reported in
the literature, and it is this project's strongest methodological contribution.

**Serology is excluded by design.** NS1/IgM/IgG are the confirmatory tests. Using them
as inputs to predict laboratory-confirmed dengue is circular. They serve as ground
truth for the label, never as predictors.

## 6. Model A — individual dengue screening

**Data.** 1,511 febrile patients, 19 CBC variables, plus four clinically motivated
derived ratios (NLR, PLR, platelet/WBC, MPV/platelet).

**Protocol.** Nested 5-fold cross-validation with hyperparameter search inside each
outer fold; isotonic calibration; all preprocessing fitted within the fold.

```
ROC-AUC 0.6811   PR-AUC 0.7790   Brier 0.1804
                         accuracy   sensitivity   specificity     NPV
threshold 0.50             0.7617      0.9729        0.3025
sensitivity >= 0.95        0.7604      0.9507        0.3466      0.764
sensitivity >= 0.90        0.7498      0.9014        0.4202      0.662
best balanced accuracy     0.7571      0.9169        0.4097
trivial majority-class     0.6850           —             —
```

| Model | ROC-AUC | Accuracy |
|---|---|---|
| **LightGBM** | **0.685** | 0.748 |
| XGBoost | 0.675 | 0.756 |
| PyTorch MLP | 0.656 | 0.671 |
| Logistic Regression | 0.651 | 0.753 |

**Threshold selection.** This is a screening tool, so the operating point is chosen for
sensitivity, not accuracy. Missing a dengue case is far costlier than an unnecessary
test.

**The signal is real but weak.** A permutation test gives z = 12.7 against a
label-shuffled null, so the model has learned something genuine. There simply is not
enough information in a full blood count alone to separate dengue from other febrile
illness. Gradient boosting beat the neural network, as expected on small tabular data.

**This is a finding, not a failure.** Reaching 90% on this task requires either
serology (circular) or a synthetic dataset.

## 7. Model B — complication risk ("who needs closer monitoring?")

The screening cohort records no symptoms and no fever duration, so the information
ablation was run on the paediatric cohort, predicting **complicated dengue**. Treatment
variables (IV fluids, steroids, antibiotics) are consequences of deterioration and are
excluded as leakage.

All rows held at ~90% sensitivity — the screening operating point:

| Information added | k | ROC-AUC | PR-AUC | Specificity | NPV |
|---|---|---|---|---|---|
| A. CBC only | 5 | 0.7474 | 0.4567 | 0.333 | 0.930 |
| B. + demographics | 7 | 0.7481 (+0.001) | 0.4602 | 0.371 | 0.937 |
| C. + symptoms | 16 | **0.8186 (+0.071)** | 0.6765 | 0.438 | 0.946 |
| D. + fever day | 17 | 0.8193 (+0.001) | 0.6720 | 0.475 | 0.950 |
| E. + organ labs | 20 | **0.8741 (+0.055)** | 0.7311 | 0.637 | 0.962 |

**Symptoms and organ labs carry the information; demographics carry almost none.**
Holding sensitivity at 90%, specificity nearly doubles (0.33 → 0.64) — the model
roughly halves false alarms without missing more patients. Permutation control z = 6.2.

At the deployed operating point: **90.5% sensitivity, 63.7% specificity, NPV 0.962** —
designed to be safe to rule *out*, with a false-positive cost of one extra observation.

## 8. Model C — spatiotemporal outbreak forecasting

Predicts whether weekly incidence will reach ≥100/100k, 2–8 weeks ahead. Train ≤2021,
validate 2022–23, **test 2024–25 (locked)**. Threshold tuned on validation only.

### Locked test set, identical rows for every model (n = 56,100 at h=4)

| Model | ROC-AUC | PR-AUC | Accuracy | Recall | Precision | Brier |
|---|---|---|---|---|---|---|
| Ensemble (GBM + LSTM) | 0.9841 | **0.9243** | **0.9556** | 0.8522 | 0.8247 | 0.0390 |
| **LightGBM** | 0.9829 | 0.9227 | 0.9544 | 0.8625 | 0.8116 | **0.0325** |
| PyTorch LSTM | 0.9810 | 0.9130 | 0.9525 | 0.8376 | 0.8157 | 0.0733 |
| Persistence baseline | 0.8862 | 0.6152 | 0.9326 | 0.8226 | 0.7188 | 0.0674 |
| Climatology baseline | 0.7462 | 0.4248 | 0.8672 | 0.0168 | 0.9621 | 0.1126 |
| Trivial ("never an outbreak") | — | — | 0.8650 | 0.0000 | — | — |

Accuracy alone misleads — predicting "no outbreak" forever scores 86.5%. The headline
metrics are **PR-AUC and recall**.

**The LSTM did not beat gradient boosting.** The lag and rolling features already
encode the temporal structure it must learn from scratch, and LightGBM is far better
calibrated (Brier 0.033 vs 0.073). LightGBM is deployed; the LSTM is retained as a
documented comparison, not because it was originally planned.

### Validation battery

**Rolling-origin backtest** — retrain each year, predict the next unseen year:

| Test year | PR-AUC | Accuracy | Trivial | Recall |
|---|---|---|---|---|
| 2019 | 0.8131 | 0.9711 | 0.9468 | 0.7375 |
| 2020 | 0.7446 | 0.9746 | 0.9720 | 0.7823 |
| 2021 | 0.7265 | 0.9858 | 0.9778 | 0.5068 |
| 2022 | 0.7836 | 0.9685 | 0.9425 | 0.6177 |
| 2023 | 0.8431 | 0.9585 | 0.9135 | 0.7481 |
| 2024 | 0.9421 | 0.9428 | 0.8028 | 0.8892 |
| 2025 | 0.8712 | 0.9589 | 0.9122 | 0.7929 |
| **mean** | **0.8178** | **0.9657** | — | 0.7249 |

**Horizon sweep** — above 90% accuracy out to 8 weeks:

| Horizon | PR-AUC | Accuracy | Trivial | Recall |
|---|---|---|---|---|
| **2 weeks (14-day warning)** | **0.9612** | **0.9669** | 0.8544 | **0.8983** |
| 4 weeks | 0.9189 | 0.9487 | 0.8563 | 0.8633 |
| 8 weeks | 0.8130 | 0.9233 | 0.8712 | 0.8004 |

**Outbreak-definition sweep** — 0.9254 / 0.9189 / 0.8499 PR-AUC at ≥50, ≥100, ≥300 per
100k. Not an artefact of one threshold.

**Shuffled-label control** — PR-AUC collapses to 0.124 against a 0.144 chance rate
(ROC-AUC 0.453). A leak would still score well here. It does not.

### Calibration

| | Brier | ECE |
|---|---|---|
| Raw LightGBM | 0.0266 | 0.0190 |
| **Isotonic (fitted on validation only)** | **0.0234** | **0.0052** |

Isotonic calibration cuts expected calibration error by 3.6× with no meaningful loss of
ranking power (PR-AUC 0.961 → 0.958). The Brazil model's outputs may honestly be called
**calibrated outbreak probabilities**. The Sri Lanka model may not — its overall ECE is
0.046, but in the highest-risk band it predicts ~0.81 where outbreaks occur ~0.54 of
the time. The application labels it *predicted*, not *calibrated*, and tells the user
to trust the ranking over the number.

## 9. Explainability — and an important correction

SHAP attribution on validation data:

```
Recent dengue incidence  ████████████████████████████  50.1%
Temperature              ██████                        10.4%
Seasonality              █████                          8.3%
Rainfall                 ████                           7.2%
Population               ███                            6.2%
Humidity                 ███                            5.5%
Land cover               ██                             4.3%
Sanitation               ██                             3.0%
ENSO                     █                              2.5%
Vegetation (NDVI/EVI)    █                              2.4%
```

Read alone, this suggests environment supplies roughly half the model's reasoning. **An
ablation shows that is misleading.**

| Information source | k | PR-AUC | Accuracy | Recall |
|---|---|---|---|---|
| Historical incidence only | 15 | 0.9559 | 0.9658 | 0.8700 |
| Environmental only | 43 | **0.5966** | 0.8679 | 0.6068 |
| Historical + environmental | 58 | **0.9614** | 0.9673 | 0.8942 |
| Persistence baseline | 1 | 0.7595 | 0.9589 | 0.8714 |

- Historical incidence adds **+0.196 PR-AUC** over persistence — a large gain.
- Environment adds only **+0.006** beyond history.
- Environment *alone* (0.597) is **worse than persistence** (0.760).

**Attribution is not incremental predictive value.** Environmental variables correlate
strongly with recent incidence, so SHAP distributes credit among them even though
removing them costs almost nothing. The model's skill is overwhelmingly a *sophisticated
reading of epidemiological momentum*, not environmental prediction.

This correction only became visible because the ablation was run. It is the single most
important methodological result in the project.

### Error analysis — where does it fail?

| Stratum | Recall | Note |
|---|---|---|
| Baseline incidence Q4 (highest) | **0.920** | strong |
| Baseline incidence Q1–Q3 | **≈0.00** | near-total failure |
| Season Jan–Mar (peak) | 0.925 | strong |
| Season Jul–Sep (trough) | 0.389 | weak |
| Unprecedented outbreaks (> historical max) | 0.941 | strong |
| Within historical range | 0.872 | strong |

Two failure modes emerge:

1. **The model tracks trajectory, not emergence.** Where transmission is already
   elevated it is excellent; where incidence is low it almost never fires. It is an
   epidemic *trajectory tracker*, not an early detector of new outbreaks — precisely
   consistent with the ablation result.
2. **Momentum carries through subsiding outbreaks.** 54.5% of false positives are
   municipalities already at epidemic level, versus 1.5% of true negatives — **37×
   more likely**. False alarms are overwhelmingly outbreaks that are ending, not
   outbreaks that never existed.

Notably, recall is *higher* for unprecedented outbreaks (0.941) than for ordinary ones
(0.872), contrary to the intuition that record-breaking events would be hardest.

## 10. Generalization

### Spatial holdout — can it predict unseen places?

| Condition | PR-AUC | Accuracy |
|---|---|---|
| Temporal holdout, municipalities seen in training | 0.9623 | 0.9677 |
| **Spatiotemporal: unseen municipalities AND unseen years** | **0.9548** | 0.9648 |
| Spatial only: unseen municipalities, training period | 0.8811 | 0.9856 |
| Persistence on the same unseen municipalities | 0.7504 | 0.9567 |

**The model generalises rather than memorising place identity.** Municipalities it has
never seen score 0.955 against 0.962 for seen ones — a negligible drop. This rules out
the concern that it merely learned "municipality X always gets dengue".

**Leave-whole-states-out** is materially harder: mean PR-AUC 0.872, range 0.602–0.972
across five folds. Whole-region transfer degrades where random municipality holdout does
not — regional climate and surveillance regimes differ.

### Brazil → Sri Lanka

| Condition | PR-AUC |
|---|---|
| Brazil validation (source domain, shared features) | **0.910** |
| **Brazil → Sri Lanka, zero-shot** | **0.449** |
| Persistence baseline | 0.476 |
| Sri Lanka only | **0.708** |
| Brazil pretrained + Sri Lanka fine-tuned | 0.660 |

**Zero-shot transfer fails** — below the persistence baseline. Fine-tuning recovers most
of the gap (0.660) and yields the best recall (0.711), but does not beat training
locally. The Sri Lanka-only model is deployed.

> A high-performing dengue forecasting model in one geographic domain cannot be assumed
> to generalize to another. Local epidemiological dynamics and calibration remain
> necessary.

Sri Lankan incidence is far lower than Brazil's (median 1.5/100k/week). Only 0.17% of
Sri Lankan district-weeks reach Brazil's 100/100k threshold, so that configuration is
degenerate and is reported as skipped; a country-calibrated threshold of 9.9/100k gives
a comparable 8.7% outbreak prevalence.

## 11. Application

`streamlit run app.py` — four areas:

1. **Dashboard** — the three tasks side by side, the decision-support pathway, and the
   four unexpected findings.
2. **Patient assessment** — CBC screening and complication risk, each labelled
   *model-estimated probability*, with the operating point and honest performance shown
   alongside, and an unmissable non-diagnosis disclaimer.
3. **Geographic risk map** — 14-day district forecast, risk bands, per-district case and
   rainfall history, with the calibration caveat stated on the page.
4. **Model evidence** — validation, calibration, explainability, generalization and the
   data audit, *including the weaknesses*.

Framing throughout is **decision support for prioritisation**, never autonomous action.

## 12. Limitations

- **Screening is weak.** ROC-AUC 0.681 on real data. A CBC alone cannot reliably
  distinguish dengue from other febrile illness. The model prioritises testing; it
  cannot rule dengue out.
- **The complication cohort is small and paediatric** (n = 303, 63 events). Estimates
  carry wide uncertainty and may not transfer to adults.
- **The information ablation was run on the complication task, not screening**, because
  no audited real dataset exists with both dengue-negative controls and symptom data.
  The trend is plausibly directional for screening but was not measured there.
- **The outbreak model detects escalation, not emergence.** Recall is near zero where
  baseline incidence is low — arguably the situation where early warning would matter
  most.
- **Environmental data contributes far less than SHAP suggests** (+0.006 PR-AUC). The
  remote-sensing pipeline is expensive for the marginal value it delivers.
- **Sri Lankan probabilities are not calibrated** and are overconfident in the
  high-risk band; the ranking is more trustworthy than the number.
- **Whole-region generalization is variable** (PR-AUC 0.602–0.972 across state folds).
- **Surveillance data is not ground truth.** Reported cases reflect testing and
  reporting behaviour as well as true incidence, and the 2024 Brazilian epidemic may
  have changed both.
- **No prospective validation.** Every result is retrospective.

## 13. Conclusion

> Reliable dengue forecasting is achievable at geographic scale under strict temporal
> validation, whereas routine-CBC dengue screening remains substantially harder on
> real-world data. Geographic transfer also cannot be assumed, demonstrating the
> importance of local calibration and validation.

Three further results qualify that headline:

1. **Most public dengue screening datasets are synthetic**, and models trained on them
   collapse on real patients. Data auditing must precede modelling.
2. **The outbreak model's skill is epidemiological momentum, not environmental
   prediction.** SHAP attribution overstated the environmental contribution by an order
   of magnitude relative to its incremental value.
3. **It tracks trajectory rather than emergence** — strong where transmission is already
   under way, near-blind where it is not.

The original target was ≥90% accuracy. That target was met where it is scientifically
defensible — 14-day outbreak forecasting at 96.7% accuracy, PR-AUC 0.961, recall 89.8%,
under temporal, spatial and adversarial validation — and was shown *not* to be
defensible for routine-CBC dengue screening. Demonstrating why a metric does not apply
is a stronger result than forcing the number.
