# 3. Measured dead ends, and the harness that made them measurable

Date: 2026-08-24
Status: Accepted

## Context

An accuracy push started from the obvious levers — more features, more tuning,
ensembling. Almost none of them work, and the first thing that had to be built was
the ability to tell that.

**The evaluation could not distinguish a gain from noise.** Every model here is
selected on one validation split. For emergence that split holds 2,191
district-weeks and 201 positives, and its PR-AUC moves from 0.357 to 0.381 when
nothing changes but the LightGBM seed. `LGBM_PARAMS` sets `seed` and leaves
`bagging_seed` and `feature_fraction_seed` at their defaults, so `subsample=0.85`
and `colsample_bytree=0.8` were never actually pinned.

A ±0.024 measurement error is larger than every candidate improvement below. Run
enough variants against an estimate that noisy and one of them wins, and the win is
real-looking, publishable, and false. That is how honest work produces inflated
numbers.

## Decision

`src/dengue/validation.py` supplies the three things selection needs:
rolling-origin folds, seed-averaged predictions with all three seeds pinned, and
`compare()`, which bootstraps the **per-fold difference** between two
configurations. Folds are shared between the two arms, so fold difficulty cancels.
Nothing ships unless its confidence interval excludes zero.

`rolling_origin` stops at `VAL_END` unless told otherwise, so the locked test years
cannot be selected on by accident.

## The results this produced

Emergence, 8 rolling-origin folds (2016–2023), 3-seed averaged, against production
at CV PR-AUC 0.3918:

| Lever | CV PR-AUC | Δ |
|---|---|---|
| **spatial neighbour features** (k=4 nearest districts by centroid, incidence + outbreak-fraction + national mean, lags 1/2/4 and a 4-week roll) | 0.3759 | **−0.016** |
| neighbour features with `num_leaves=3` | 0.3406 | **−0.051** |
| `num_leaves=3` | 0.3838 | −0.008 |
| `num_leaves=3`, `reg_lambda=150` | 0.3782 | −0.014 |
| ENSO — niño3.4 + SOI, lags 0/2/3/6/9/12 months | 0.3952 | +0.003 |
| ENSO — niño3.4 only | 0.4003 | +0.009 (fold sd 0.13; not resolvable) |

Emergence again, this time changing what it trains on and what it is asked rather
than its columns (`experiments/accuracy_v2/search_emergence.py`, paired):

| Arm | CV PR-AUC | Δ (95% CI) | |
|---|---|---|---|
| A production — eligible rows, binary 4-week label | 0.3918 | — | |
| B + ineligible district-weeks as context, with an eligibility flag | 0.3990 | +0.0072 [−0.0051, +0.0237] | noise |
| C discrete-time hazard over the risk set, recombined | 0.3887 | −0.0031 [−0.0193, +0.0110] | noise |
| D both | 0.3920 | +0.0002 [−0.0253, +0.0240] | noise |

Screening, 5×5 repeated stratified CV on the 1,511-patient cohort:

| Lever | ROC-AUC | PR-AUC |
|---|---|---|
| production LightGBM | 0.6875 | 0.7829 |
| rank-average ensemble of LightGBM + XGBoost + logistic regression | 0.6724 | 0.7712 |

And one that was already in the repo and had simply not been read:
`reports/srilanka_transfer.csv` records **Brazil fine-tuning** (`init_model`, boosting
continued on Sri Lankan data) at PR-AUC 0.660 against 0.708 for the Sri Lanka-only
model. Warm-starting from three orders of magnitude more data is not merely
unhelpful here, it costs 0.048. It was planned as this project's most promising
untried lever, on the mistaken belief that only *zero-shot* transfer had been tested.
It had not been untried; it had been tried and had lost.

## Consequences

- **Spatial neighbour features are a dead end for Sri Lanka emergence.** This is
  the counter-intuitive one: dengue spreads geographically, and a quiet district
  next to a burning one ought to be at risk. With 26 districts the k-nearest set is
  a quarter of the country, so a "neighbour" is a national average with extra
  steps — and the model already has the district's own history, which correlates
  with it. Twelve columns of dilution, no information.
- **Ensembling hurts screening.** LightGBM is meaningfully better than the other two
  (0.6875 vs 0.6741 and 0.6494); averaging drags it toward them. The right move on
  a single strong learner is not to average it with weak ones.
- **Emergence hyperparameters are already at a local optimum.** The rolling-origin
  tuning recorded in `finalize_emergence.PARAMS` was done properly. Re-tuning
  `num_leaves` and `reg_lambda` around it only loses ground.
- **More rows do not help emergence, and neither does a better-shaped question.**
  Arm B roughly ten-times the positives by keeping the district-weeks the
  eligibility rule discards; most of them are trivially positive (a district in
  outbreak is still in outbreak next week), so the model learns "high incidence →
  yes", which is the one thing a quiet district cannot use. Arm C replaces the
  lumped binary label with a proper discrete-time hazard over a shrinking risk set —
  the formulation the data is actually shaped like — and lands within noise of
  production. The eligibility rule is not throwing away signal; the signal is not
  there.
- **Before proposing transfer learning, read `reports/srilanka_transfer.csv`.** It
  already answers the question, in the negative, for both zero-shot and fine-tuned.
- **ENSO is unresolved, not rejected.** The direction is positive and the mechanism
  is real — ENSO drives the monsoon at a 3–6 month lead, and no other feature in the
  panel looks further back than eight weeks. But +0.009 against a fold sd of 0.13
  is not a result. It needs the paired harness and more folds before it is anything.
- Do not re-run these without new information. Re-testing a dead end until it passes
  is the same failure the harness exists to prevent, one level up.

## What did survive

Recorded here so the file is not read as "nothing helps":

- Absolute cell counts and the indices built on them for screening (+0.003 ROC-AUC,
  +0.010 PR-AUC over 25 folds) — see `dengue.datasets.clinical_ratios`.
- Cutting the continuation model's capacity — see ADR 4.
- Calibrating the Sri Lanka probabilities, which changes no ranking metric at all
  and was never about accuracy — see `dengue.calibration`.

## Alternatives rejected

**Keep the single validation split and accept the numbers.** It is cheaper and it
is how the ±0.024 seed spread went unnoticed. The split is still what thresholds are
chosen on — that part of the protocol is unchanged — but it is not strong enough to
choose *models* on.

**Report the best-seed result.** Selecting the seed is selecting noise, and it is
indistinguishable from selecting a feature that works.
