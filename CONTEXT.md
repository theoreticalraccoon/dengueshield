# Domain language

The words the code is named after. If a term here changes meaning, the module named
after it changes with it.

## Populations

**District** — a Sri Lankan administrative district, 26 of them, the unit of the
outbreak forecast. The Brazilian equivalent is a **municipality**; the two panels
are the same shape (place × week) but are never mixed.

**District-week** — one district observed in one surveillance week. The row of the
panel, and the thing a forecast is about.

**In outbreak** — incidence at or above the epidemic threshold this week
(`SL_INC` = 9.9 per 100k/week for Sri Lanka, `BR_INC` = 100 for Brazil). Not a
prediction: a statement about the week just reported.

**Quiet** — below the threshold. **Eligible** — quiet for `QUIET_WEEKS` consecutive
weeks, and therefore a district the emergence model will score. A district that left
outbreak last week is quiet but not yet eligible.

## The questions

Four separate questions asked of four different populations. They are never
combined into one accuracy figure.

**Screening** — does this febrile patient have dengue? Asked of one patient, from
routine bloodwork.

**Complication** — does this dengue patient need closer monitoring? Asked of one
admission already known or suspected to have dengue.

**Continuation** — will this district's outbreak still be underway in 14 days?
Asked only of districts currently **in outbreak**.

**Emergence** — will a new outbreak begin here within 1–4 weeks? Asked only of
**eligible** districts. Continuation and emergence are the same phenomenon asked of
**disjoint** populations, which is why neither answers for the other.

## Verdicts

**Headline risk** — the probability answering the question actually being asked of
a district: continuation if it is in outbreak, emergence if it is eligible. It is
**absent** when neither question applies. See
[ADR 0001](docs/adr/0001-blank-emergence-risk.md).

**Assessable** — a district some model is asked about. A district that is out of
outbreak too recently for emergence is **not assessable**, and its headline risk is
absent rather than zero, low, or borrowed.

**Triage** — the four groups the forecast screen sorts districts into:
*Outbreak now*, *Outbreak likely*, *Clear*, *Not assessable*.

**Band** — headline risk cut into *Low* / *Moderate* / *High* / *Very High*. A
district with no headline risk has no band.

## Evidence

**Locked test set** — 2024–25. Never tuned against; thresholds come from the
2022–23 validation years only.

**Persistence** — the trivial baseline that predicts next period from this one.
Quoted alongside every outbreak metric, because a model that cannot beat it has
learned nothing.

**Serology** (NS1/IgM/IgG) — ground truth for the screening label. Never a
predictor.
