# 6. Emergence: a two-model blend, and selecting on two axes instead of one

Date: 2026-08-25
Status: Accepted

## Context

ADR 0003 left emergence as the weakest deployed model — locked-test PR-AUC 0.405
against a persistence baseline of 0.250 — and recorded six measured dead ends. It
also left two things unresolved.

**The instrument could not see what was left.** Selection used eight rolling-origin
folds from 2016 with a paired bootstrap over folds. At a fold sd of ~0.13 that
interval cannot resolve anything below about 0.05, and every remaining candidate is
smaller than that. ENSO sat in ADR 0003 as "+0.009, unresolved" for exactly this
reason: not rejected, not accepted, unmeasurable.

**The deployed score was not the score that had been measured.** Isotonic calibration
merges distinct scores onto the same step, and the app scores through the calibrator.
The reports recorded `pr_auc` 0.405 and `pr_auc_calibrated` 0.358, and 0.358 was what
the dashboard actually ranked districts by. That gap was a defect, not a property of
the model.

## Decision

### Selection now runs on two resampling axes

`validation.pooled_compare` pools out-of-fold predictions across folds into one score
per arm and bootstraps the difference by resampling the 26 **districts**. A district's
weeks are strongly autocorrelated, so a district — not a district-week — is the
independent unit; resampling rows would treat 23,000 correlated observations as
independent and return an interval far too tight to be honest.

`compare` (over years) is kept unchanged. An arm ships only if **both** intervals
exclude zero in the same direction. Years and districts are different ways for a
result to be a fluke, so this is stricter than either test alone, not a way around the
one that was failing. Folds also start at 2010 rather than 2016 (8 → 14) and selection
uses 5 seeds rather than 3.

`validation.fast_average_precision` replaces sklearn's in the bootstrap loops — a
six-arm search spent thirty of its thirty-five minutes inside input validation. It is
identical to `average_precision_score` when no two scores are tied, which holds for
every caller, and a test pins the equivalence.

### The model is a blend of two formulations of the same question

`dengue.blend.BlendedEmergence` averages, on a probability scale:

- the **binary classifier** — the shipped formulation, "does incidence cross 9.9
  within four weeks?", trained on 1,090 positives out of 23,053 rows;
- an **incidence regression** — the same window and the same features, but never
  thresholded, so all 23,053 rows carry a target instead of one row in twenty-one.

The binary label is where most of this task's information was going: a week that
reached 9.8 scored identically to one that reached 0.2, and a week that hit 40 counted
the same as one that grazed 10.0.

This is not ADR 0003's arm C. That reformulation was still binary — "does it cross in
exactly week k?" — and landed inside the noise. This one is continuous.

Nor is it in tension with ADR 0003's rejection of ensembling for screening. There the
members were one strong learner and two weak ones and averaging dragged the strong one
down. Here they are two comparably good models of the same phenomenon that make
different errors.

### Isotonic calibration no longer destroys the ranking

`calibration.TieBrokenIsotonic` adds a vanishing, strictly increasing function of the
raw score, bounded below the smallest real isotonic step, so points sharing a step
keep the model's ordering. The deployed score now ranks exactly as well as the model
it came from, which is the most a calibrator can honestly leave intact. The one-sided
invariant (`calibrated ≤ raw`) still holds, now at equality — a stronger guard, and
the existing test still enforces it.

## Results

Fourteen rolling-origin folds, 5 seeds, both intervals. Baseline: per-fold 0.3610,
pooled 0.3402.

| Arm | per fold (14) | per district (26) | ships |
|---|---|---|---|
| climate block, all 13 columns | +0.0206 [−0.0134, +0.0628] | +0.0102 [−0.0122, +0.0356] | no |
| climate, anomalies only | +0.0072 [−0.0113, +0.0312] | −0.0066 [−0.0156, +0.0019] | no |
| climate core (8 columns) | +0.0107 [−0.0177, +0.0488] | −0.0068 [−0.0239, +0.0124] | no |
| susceptible-pool reconstruction | −0.0041 [−0.0228, +0.0138] | −0.0022 [−0.0183, +0.0132] | no |
| gravity-weighted importation | −0.0202 [−0.0442, −0.0004] | −0.0037 [−0.0189, +0.0121] | **regression** |
| national epidemic phase | −0.0100 [−0.0362, +0.0094] | +0.0038 [−0.0075, +0.0163] | no |
| **ENSO, niño3.4 lagged** | +0.0155 [−0.0130, +0.0433] | +0.0126 [−0.0128, +0.0411] | no |
| regression on log incidence | +0.0120 [−0.0213, +0.0491] | +0.0175 [−0.0096, +0.0475] | no |
| Tweedie on incidence | −0.0039 [−0.0327, +0.0232] | −0.0020 [−0.0314, +0.0352] | no |
| quantile regression, α=0.95 | −0.0377 [−0.0973, +0.0058] | −0.0092 [−0.0449, +0.0322] | no |
| **blend (classifier + regression)** | **+0.0337 [+0.0067, +0.0671]** | **+0.0237 [+0.0050, +0.0462]** | **SHIP** |
| blend + climate core | +0.0361 [+0.0005, +0.0830] | +0.0218 [+0.0032, +0.0431] | ships, not taken |
| blend + climate + ENSO | +0.0352 [−0.0046, +0.0770] | +0.0493 [+0.0223, +0.0807] | no |

**The plain blend is deployed, not the highest-scoring arm.** Blend+climate scores
marginally higher per fold and marginally lower per district, and its fold interval
bottoms out at +0.0005 — one fold from failing — for eight extra columns. Blend +
climate + ENSO has the best pooled delta and the best ROC of anything measured, and
fails the fold axis outright. Choosing either on its point estimate would be selecting
the noisiest signal available, which is the failure ADR 0003 exists to prevent.

### On the locked test years

| | before | after |
|---|---|---|
| PR-AUC | 0.4054 | **0.4162** |
| PR-AUC of the score the app serves | 0.3584 | **0.4162** |
| ROC-AUC | 0.8268 | 0.8245 |
| vs persistence | +0.155, no interval | **+0.167 [+0.086, +0.236]** |
| districts flagged per week for 90% recall | 17.31 of 26 | **12.99 of 26** |
| ECE | 0.0392 | 0.0414 |
| mean lead time | not measured | **2.4 weeks** |

The persistence comparison now carries an interval that excludes zero, so "better than
the operational status quo" is a claim rather than two numbers side by side.

**Deployment uses three seeds, not more.** Raising the seed count looked free -
averaging more draws of one estimator is variance reduction rather than selection -
but measured over the development folds the blend scores 0.3824 at three seeds,
0.3799 at five and 0.3816 at nine: a spread of 0.0025, which is the noise the
averaging was meant to remove. The cost is not noise. The blend already doubles the
bundle because it carries two members, and CI commits that bundle every Tuesday, so
nine seeds put a 9 MB file into git history weekly instead of 3 MB.

## Consequences

- **Weekly top-k allocation is a dead end, and it was the most promising idea on
  paper.** Ranking districts within each week and flagging the top k fits how a team
  with fixed capacity actually works, and needs less of the model — only the ordering
  inside one week. It is worse at every budget: 90% recall costs 10.8 districts/week
  under top-k against 7.9 under a global threshold
  (`experiments/accuracy_v2/alert_policy.py`). Emergence events concentrate in the
  transmission season, so a constant weekly budget rations alerts exactly when they
  are needed and wastes them when nothing is starting. The score is comparable across
  weeks after all. `metrics.top_k_by_week` is kept, and is not used by the pipeline.
- **A rank-normalised blend cannot be pooled.** The first version rank-averaged within
  each fold, which is fine per fold and destroys the pooled comparison: fourteen folds
  each spanning 0 to 1 make a quiet year indistinguishable from an epidemic year. It
  read +0.0272 per fold and −0.0669 pooled. Both numbers were about the normalisation,
  not the model. The fix — mapping the regression through an isotonic fitted on the
  training fold — is a scale bridge, monotone, and changes no per-fold ranking.
- **ENSO is still not resolved, but it is now the strongest unshipped lead.** Positive
  on both axes (+0.0155, +0.0126), consistent with ADR 0003's +0.009, and consistently
  short of significance. It is fetched, cached, committed and wired into the weekly
  refresh, so the next person to look at it does not start from nothing. The
  publication-lag guard in `dengue.enso` matters: the ONI for month M is published
  during M+1, and wiring it in by calendar month leaks a small, plausible, invisible
  amount of the future.
- **Susceptible-pool reconstruction does not help, which is the surprising one.**
  Multi-annual dengue cycles are driven by susceptible depletion and it is the one
  mechanism the panel had no column for. It measures flat to slightly negative on both
  axes. The likely reason is that `weeks_since_outbreak` and `hist_outbreak_rate`
  already carry most of what a 26-district panel can support, and the reconstruction
  adds assumptions (a reporting multiplier, a waning half-life) without adding
  resolution.
- **Emergence's own climate features do not survive contact with the harness either**,
  despite finding 3 measuring environment at +0.089 for this task. Trimming the block
  made it worse, not better, on the pooled axis — the opposite of what the dilution
  argument predicts — which is itself a reason not to trust any of these deltas
  individually.
- **Do not re-run the rejected arms without new information.** Two of them were run
  here precisely because ADR 0003 recorded them as unresolved rather than rejected;
  that is the distinction worth preserving.

## Alternatives rejected

**Ship blend + climate + ENSO because it has the best pooled delta and ROC.** It fails
the fold axis. Taking it would mean the two-axis rule applies only when it agrees with
the result already wanted.

**Relax the ship rule to "positive mean, no test regression".** It would have shipped
six of the twelve arms above, most of them noise, and the locked test set would have
become a selection surface.

**MOH-division data, to raise the panel from 26 units to ~350.** This is the one
structural lever with step-change potential — the Brazil continuation model reaches
PR-AUC 0.96 on thousands of municipalities for the same kind of question. It was
investigated and rejected on availability: the NDCU weekly PDFs carry MOH counts only
for ~76 *high-risk* areas, which is a selection-biased subset, and the archive begins
at 2026 week 12. One year of biased data cannot replace a 2007–2026 panel. Those same
PDFs do carry national serotype surveillance, which is absent from the model and is
the best-documented driver of Sri Lankan epidemic years; it is the strongest remaining
untried lead, gated on finding more than one year of history.
