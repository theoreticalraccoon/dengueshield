# 1. A blank emergence risk means "not assessable", never a continuation risk

Date: 2026-08-23
Status: Accepted

## Context

Continuation (task C) and emergence (task D) are the same phenomenon asked of
disjoint populations. `finalize_emergence.py` writes `NaN` into
`emergence_risk` for every district it does not ask about, with the comment
`# question not asked`. Two populations are excluded:

- districts already at or above the epidemic threshold, where continuation is the
  question instead; and
- districts that left outbreak within the last `QUIET_WEEKS` weeks, which are
  eligible for neither model.

The app read that `NaN` three different ways in one file. Two call sites
(`app.py:406`, `app.py:566`) substituted `continuation_risk`; a third
(`app.py:578`) classified the district as "Not assessable". The map therefore
coloured six districts with a band derived from a continuation probability produced
by a model fitted on a population those districts are not in, while the triage panel
directly above it said no forecast existed for them.

The substitution was not conservative in either direction. Trincomalee showed
"Moderate" from a 0.49 continuation risk; Vavuniya showed "Low" from 0.01. Both
readings look like forecasts and neither is one.

## Decision

One rule, in `src/dengue/risk.py`, used by every caller.

A district that is not in outbreak and has no emergence score is **not assessable**:
`headline_risk` is `None` and `band` is `None`. Callers must render that absence —
a grey marker on the map, an em-dash in the table, "n/a" on the patient screen —
rather than substituting the other model's number.

`assess_frame` applies `assess` row by row rather than reimplementing it in
vectorised form, so the table and the single-district read-out cannot drift apart
again. There are 26 districts; the cost is irrelevant.

## Consequences

- Triage counts are unchanged: 16 / 0 / 4 / 6 before and after.
- Six districts lose a map band they should never have had. No district gains one.
- `headline_risk` is now nullable, so callers must handle absence. That is the
  point: the type makes the missing forecast impossible to ignore.
- Continuation and emergence probabilities remain visible as separate columns in
  the forecast table. They are facts about what each model output; only their use
  as *the district's risk* is constrained.

## Alternatives rejected

**Fall back to continuation everywhere.** Consistent, and it preserves the current
rendering, but it publishes a number from a model applied outside its population.
The project's stated purpose includes reporting where the models fail; papering
over an unanswerable question contradicts that.

**Show 0 risk.** Worse. "We did not ask" and "we asked and the answer is no" are
different statements, and this one is read by people deciding where to send vector
control.
