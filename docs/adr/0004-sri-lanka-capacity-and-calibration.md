# 4. Sri Lanka: less capacity, averaged seeds, and scores that are probabilities

Date: 2026-08-24
Status: Accepted

## Context

Three separate problems in the two Sri Lanka models, found while looking for
accuracy and only one of which is about accuracy.

**The continuation model's capacity was never searched.** `LGBM_PARAMS_SL` set
`num_leaves = 63` with a comment explaining that 26 districts need smaller trees
than Brazil's thousands of municipalities. The reasoning was right; the number was a
guess, and `finalize_srilanka.ROUNDS = 500` was another one. Nothing in the repo had
ever tested either.

**A single booster is a lottery ticket.** `LGBM_PARAMS` pins `seed` and leaves
`bagging_seed` and `feature_fraction_seed` at their defaults, so `subsample=0.85`
and `colsample_bytree=0.8` were never reproducible. On the continuation test years
one seed scores PR-AUC 0.701 where a three-seed average scores 0.725 — the seed was
worth more than any tuning.

**The scores were not probabilities.** Both models train with `scale_pos_weight`
(~14 for continuation, ~23 for emergence). That is correct for ranking and it
deliberately destroys the probability scale. Test ECE was 0.140 for continuation
and 0.292 for emergence — a district shown as "80%" was not an 80% district. The
app said so in prose, but `current_forecast` still cut Low/Moderate/High/Very High
bands on the raw number, and the emergence operating point was a sensitivity target
read off a distorted scale.

## Decision

**Capacity.** `num_leaves = 15`, `ROUNDS = 250`. Chosen over 8 rolling-origin folds
with a paired bootstrap on the per-fold difference
(`experiments/accuracy_v2/search_continuation.py`), never on the test years:

| Candidate | Δ CV PR-AUC | 95% CI | |
|---|---|---|---|
| `num_leaves=15`, 250 rounds | **+0.0191** | [+0.0105, +0.0273] | ship |
| `num_leaves=7`, 500 rounds | +0.0160 | [+0.0079, +0.0235] | ship |
| `num_leaves=15`, 500 rounds | +0.0108 | [+0.0056, +0.0160] | ship |
| `num_leaves=31`, 500 rounds | +0.0023 | [−0.0008, +0.0058] | not resolvable |
| 1000 rounds | −0.0076 | [−0.0094, −0.0057] | regression |

Seven of nine candidates beat production. The robust finding is the direction — 63
leaves was too many — not the exact winner, which is not distinguishable from the
next two.

**Seeds.** `dengue.validation.SeedEnsemble` holds one booster per seed and averages
them. Both finalize scripts deploy one. Development measured an averaged model, so
production deploys an averaged model.

**Calibration.** `dengue.calibration.fit_isotonic_oof` fits isotonic regression on
rolling-origin out-of-fold predictions across the development years. Out-of-fold
rather than validation-only because both scripts refit the production model on
train+validation; a calibrator fitted on validation predictions from a train-only
model would map from a distribution the shipped model never produces. Thresholds and
operating points are now chosen on the calibrated scale, because that is the scale
every consumer sees.

## Results on the locked test years

| | before | after |
|---|---|---|
| Continuation PR-AUC | 0.7086 | **0.7253** |
| Continuation ECE | 0.140 | **0.023** |
| Continuation Brier | 0.098 | **0.050** |
| Emergence PR-AUC | 0.4079 (one seed) | 0.4054 (averaged) |
| Emergence ECE | 0.292 | **0.039** |
| Emergence Brier | 0.199 | **0.052** |
| Emergence recall @ target 0.70 | 0.737 | 0.731 |

Persistence baselines are unchanged: 0.469 continuation, 0.250 emergence.

The PR-AUC rows are a genuine before/after of two different models. The ECE and Brier
rows are not: there was no calibrator before, so "before" is the *current* model's
raw score and "after" is the same model's score through the calibrator. That is the
correct paired comparison for a calibration change, and it is what
`reports/improvement_summary.json` records.

## Consequences

- **Emergence did not get more accurate, and the number went down slightly.** 0.4079
  was a single-seed draw; 0.4054 is the average of three. The honest comparison is
  average-to-average, and by that measure nothing changed. What changed is that the
  figure is now reproducible.
- **Discrimination and calibration are reported on different estimators, on
  purpose.** Isotonic cannot reverse a pair, so it cannot improve ranking — but it
  merges scores it cannot distinguish into ties, and PR-AUC charges for ties
  (continuation 0.7253 raw, 0.7092 calibrated). Booking that coarsening as lost
  model quality would be wrong: thresholding a monotone transform gives the identical
  partition, so no decision changes. `pr_auc` is therefore the raw score and
  `pr_auc_calibrated` sits beside it. The latter must never exceed the former; if it
  does, the calibrator saw test labels.
- **Saved thresholds changed scale.** Emergence went from 0.649 to 0.164 at the same
  sensitivity target. Anything quoting a raw threshold is now wrong; the operating
  point table in `config.py` has been updated.
- **Calibrated forecasts are capped at 0.001/0.999** (`calibration.CERTAINTY_CAP`).
  Isotonic's end steps are unregularised, so an all-positive top bin fits to exactly
  1.0 and the dashboard would print "100%" — a claim of certainty about the future
  from a few hundred observations.
- **Several districts now tie on the dashboard**, because isotonic says the data
  cannot separate them. That is a true statement about the evidence, but it does
  degrade the ranked table; if fine ordering matters, break ties on the raw score,
  which is monotone-consistent with the calibrated one.
- `models/srilanka_outbreak.joblib` and `models/srilanka_emergence.joblib` now hold a
  `SeedEnsemble` and a `calibrator`. `predictor.from_bundle` reads the calibrator
  when present and passes scores through unchanged when absent, so older bundles
  still load.

## Alternatives rejected

**Platt scaling instead of isotonic.** The distortion `scale_pos_weight` introduces
is monotone but not sigmoid-shaped, and there are ~10k development rows — well past
where isotonic overfits.

**Calibrate on the validation split only.** Simpler, and it would fit the mapping to
a model that is not the one shipped.

**Keep a single booster to preserve the existing bundle shape.** It would mean
publishing a number the deployed artifact only reaches on a lucky seed.
