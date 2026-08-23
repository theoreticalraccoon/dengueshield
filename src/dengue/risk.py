"""The district risk rule: which model answers for a district, and what it means.

Continuation and emergence are the same phenomenon asked of DISJOINT populations.
Continuation asks "will this outbreak persist 14 days?" of districts already at or
above the epidemic threshold. Emergence asks "will a new outbreak begin?" of the
districts continuation is not asked about - and only once they have been quiet for
`QUIET_WEEKS` consecutive weeks. A district that left outbreak last week is in
neither population: it is too recently hot for emergence and no longer hot enough
for continuation.

That last case is why this module exists. A blank `emergence_risk` does not mean
"low risk", it means THE QUESTION WAS NOT ASKED - the forecast script writes NaN
there deliberately. Read as a number it silently becomes a continuation risk
computed for a population the district is not in, which is how the app came to
show three different answers for the same district: two call sites fell back to
`continuation_risk` and a third called it "Not assessable".

One rule, one place. `assess_frame` applies `assess` row by row rather than
vectorising it separately, so the table and the single-district read-out cannot
drift apart again. There are 26 districts; nothing here needs to be clever.

See docs/adr/0001-blank-emergence-risk.md for why "not assessable" won.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Risk bands, low to high. The edges are open at the bottom and closed at the top,
# so 0.25 lands in "Low" and 0.75 in "High".
BANDS = ("Low", "Moderate", "High", "Very High")
BAND_EDGES = (0.25, 0.50, 0.75)

OUTBREAK_NOW = "Outbreak now"
OUTBREAK_LIKELY = "Outbreak likely"
CLEAR = "Clear"
NOT_ASSESSABLE = "Not assessable"

# Display order: worst first, unknown last.
TRIAGE_ORDER = (OUTBREAK_NOW, OUTBREAK_LIKELY, CLEAR, NOT_ASSESSABLE)

TRIAGE_REASON = {
    OUTBREAK_NOW: "incidence is at or above the epidemic threshold",
    OUTBREAK_LIKELY: "quiet now, but emergence risk is at or above {threshold:.0%}",
    CLEAR: "quiet, and no emergence signal",
    NOT_ASSESSABLE: (
        "out of outbreak too recently for the emergence model, "
        "which needs two consecutive quiet weeks"
    ),
}


def band_for(p: float | None) -> str | None:
    """Band a probability, or None when there is no probability to band."""
    if p is None or pd.isna(p):
        return None
    for edge, name in zip(BAND_EDGES, BANDS[:-1], strict=True):
        if p <= edge:
            return name
    return BANDS[-1]


@dataclass(frozen=True)
class DistrictRisk:
    """What we can say about one district this week.

    `headline_risk` is the probability that answers the question actually being
    asked of this district - continuation if it is in outbreak, emergence if it is
    quiet and eligible. It is None when neither question applies, and callers must
    render that absence rather than substituting the other model's number.
    """

    district: str
    in_outbreak: bool
    triage: str
    headline_risk: float | None
    band: str | None
    continuation_risk: float | None
    emergence_risk: float | None

    @property
    def assessable(self) -> bool:
        return self.triage != NOT_ASSESSABLE

    @property
    def status(self) -> str:
        return "In outbreak" if self.in_outbreak else "Not in outbreak"

    def reason(self, emergence_threshold: float) -> str:
        return TRIAGE_REASON[self.triage].format(threshold=emergence_threshold)


def _opt(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def assess(row, emergence_threshold: float) -> DistrictRisk:
    """Apply the rule to one district-week.

    `row` is anything with attribute access over the dual-risk columns: a namedtuple
    from `itertuples`, a Series from `.iloc[0]`, or a plain object.
    """
    in_outbreak = bool(row.currently_in_outbreak)
    continuation = _opt(row.continuation_risk)
    emergence = _opt(row.emergence_risk)

    if in_outbreak:
        # Continuation is the question being asked. Emergence is blank by
        # construction here, and that blank carries no information.
        triage, headline = OUTBREAK_NOW, continuation
    elif emergence is None:
        # Quiet, but not quiet for long enough. Neither model speaks for this
        # district; falling back to continuation would answer a question about a
        # population it is not in.
        triage, headline = NOT_ASSESSABLE, None
    elif emergence >= emergence_threshold:
        triage, headline = OUTBREAK_LIKELY, emergence
    else:
        triage, headline = CLEAR, emergence

    return DistrictRisk(
        district=str(row.district),
        in_outbreak=in_outbreak,
        triage=triage,
        headline_risk=headline,
        band=band_for(headline),
        continuation_risk=continuation,
        emergence_risk=emergence,
    )


def assess_frame(dual: pd.DataFrame, emergence_threshold: float) -> pd.DataFrame:
    """Add triage, headline_risk, band, assessable and status to a dual-risk table.

    Returns a copy; the input is not modified.
    """
    out = dual.copy()
    verdicts = [assess(r, emergence_threshold) for r in out.itertuples()]

    def col(values, dtype):
        # Explicit dtype: a column that is entirely "not assessable" would
        # otherwise land as object and break downstream rounding and formatting.
        return pd.Series(values, index=out.index, dtype=dtype)

    out["triage"] = col([v.triage for v in verdicts], "string")
    out["headline_risk"] = col([v.headline_risk for v in verdicts], "float64")
    # object, not "string": a missing band must read back as None here exactly as
    # `assess` returns it, rather than as pd.NA. The two must not disagree.
    out["band"] = col([v.band for v in verdicts], "object")
    out["assessable"] = col([v.assessable for v in verdicts], "bool")
    out["status"] = col([v.status for v in verdicts], "string")
    return out


def triage_counts(assessed: pd.DataFrame) -> dict[str, int]:
    """District count per triage group, in display order, including empty groups."""
    return {g: int((assessed.triage == g).sum()) for g in TRIAGE_ORDER}
