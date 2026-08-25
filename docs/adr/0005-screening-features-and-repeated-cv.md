# 5. Screening: absolute cell counts, and an estimate that stops moving

Date: 2026-08-24
Status: Accepted

## Context

The screening model is the weakest of the four by ROC-AUC (0.681) and the hardest to
improve, because it is the only one whose ceiling is set by the assay rather than by
the modelling: a complete blood count is being asked to separate dengue from other
febrile illness, and three of the four public datasets that would have added
symptoms or serology are synthetic and rejected by `run_audit.py`.

Two things were nonetheless available.

**The features were percentages where the physiology is counts.** `clinical_ratios`
built NLR, PLR, PLT/WBC and MPV/PLT, all ratios of *percentages*. But 40%
lymphocytes means something different at a white cell count of 3,000 than at 11,000,
and lymphopenia — with thrombocytopenia, the classic dengue haematology pair — is a
statement about an absolute count. A tree splitting on the percentage alone cannot
express it without also splitting on WBC.

**The final model was chosen on one 5-fold split.** `finalize_model1.py` ran a
single `StratifiedKFold(5)` outer loop. On 1,511 patients that estimate moves by
more than any feature work gains, so which hyperparameters "won" was partly a
property of `random_state`.

## Decision

**Add the counts and the indices built on them** to `dengue.datasets.clinical_ratios`
— absolute neutrophil, lymphocyte and monocyte counts, lymphocyte-monocyte ratio,
systemic immune-inflammation index, calculated plateletcrit, RDW/platelet and
HCT/platelet. Added to the shared function, not to the loader, so inference rebuilds
them by the same code path; `predictor._screening_ratios` and the loader now both
call it through one `HEMA_COLUMNS` mapping instead of two hand-written copies of the
column names. They also become their own ablation tier, `D_+absolute_counts`, so the
report shows what they bought over the ratios alone.

**Repeat the outer CV three times** and average the out-of-fold probabilities.

## Results

Nested CV, out-of-fold, calibrated:

| | before | after |
|---|---|---|
| ROC-AUC | 0.6811 | **0.6900** |
| PR-AUC | 0.7790 | **0.7823** |
| Brier | 0.1804 | **0.1778** |
| Specificity at sensitivity ≥ 0.90 | 0.4202 | **0.4454** |
| Accuracy at 0.5 | 0.7617 | 0.7644 |

The specificity row is the one that matters clinically: at the same sensitivity, the
model now rules out 44.5% of non-dengue patients rather than 42.0%.

**The delta is two changes, not one**, and the split between them is not what the
first estimate suggested. Two within-protocol measurements of the feature block
alone disagree:

| Protocol | C (ratios, 22 feat) | D (+counts, 30 feat) | Δ ROC-AUC |
|---|---|---|---|
| Fixed hyperparameters, 5×5 repeated CV, uncalibrated | 0.6875 | 0.6907 | +0.003 |
| `nested_oof` — tuned inside every fold, 3 repeats, calibrated (`reports/ablation_hematology.csv`) | 0.6740 | 0.6923 | **+0.018** |

The first held hyperparameters fixed at values that had been tuned for the *old*
feature set, so the new columns had to earn their keep without the model being
allowed to change how it used them. The second retunes inside each fold. Since the
deployed pipeline tunes, the ablation figure is the one that describes what actually
shipped, and the counts are worth considerably more than the first pass implied.

Neither has a confidence interval — both are single estimates on 1,511 patients,
not paired rolling-origin comparisons — so the honest reading is "somewhere between
+0.003 and +0.018, probably nearer the top when hyperparameters are free to adapt".
Repeated outer CV accounts for the remainder of the headline move and is a genuine
variance reduction in its own right.

The ablation table also shows the counts doing something the ratios did not: tier C
*lowered* PR-AUC against tier B (0.7790 → 0.7700) while raising ROC-AUC, and tier D
lifts both (0.7909, 0.6923). Ratios of percentages were reshuffling information the
model already had; absolute counts add some.

The dataset gate is unaffected: maximum single-feature AUC is still 0.624 (RDW-CV),
no new feature is leaky or class-disjoint, and `run_audit.py` still returns PASS.

## Consequences

- `cbc_population_medians.json` gains eight entries. It is load-bearing — Quick entry
  mode fills every unentered field from it — so `derive_reports.py --write` has to run
  after this, and `tests/test_artifacts.py` fails until it does.
- The bundle's feature list grows from 22 to 30. Older bundles still load; they simply
  ask for fewer columns.
- **This does not make CBC-only screening good.** ROC-AUC 0.690 against a 0.685
  majority-class baseline is a modest model, and the About screen says so. The honest
  ceiling here is set by the data, not the estimator: rank-average ensembling with
  XGBoost and logistic regression makes it *worse* (docs/adr/0003), and no audited
  real dataset offers symptoms alongside dengue-negative controls.

## Alternatives rejected

**Ensembling.** Tested; it loses 0.015 ROC-AUC. LightGBM is meaningfully stronger
than the other two learners and averaging drags it toward them.

**Monotonic constraints on platelet count and WBC.** Clinically well-motivated and
not attempted — the direction is only unambiguous holding everything else fixed,
which is not what a constraint enforces. Worth a paired test if this is revisited.
