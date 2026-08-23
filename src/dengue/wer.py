"""Parse Sri Lanka Weekly Epidemiological Reports (WER) for district dengue counts.

denguedatahub is the reliable structured source, but it lags the primary publisher
by weeks to months. The Epidemiology Unit publishes the same numbers weekly as PDFs,
so this module reads them directly to close the gap.

The WER district table is typeset SIDEWAYS: every glyph is individually placed in a
vertical run, so the reading order pdfplumber produces is inverted along y.
Reconstructing each run bottom-to-top recovers rows of the form

    Colombo 1138 10495 0 16 0 2 ...
    <district> <dengue this week> <dengue cumulative> <next disease> ...

Nothing here is trusted blindly: validate_against() re-parses weeks that
denguedatahub already covers and refuses the source if the numbers disagree.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
WER_DIR = RAW / "wer"
INDEX_URL = "https://www.epid.gov.lk/weekly-epidemiological-report/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}

# the 26 reporting districts, spelled as the panel expects them
DISTRICTS = [
    "Colombo",
    "Gampaha",
    "Kalutara",
    "Kandy",
    "Matale",
    "NuwaraEliya",
    "Galle",
    "Hambanthota",
    "Matara",
    "Jaffna",
    "Kilinochchi",
    "Mannar",
    "Vavuniya",
    "Mullaitivu",
    "Batticaloa",
    "Ampara",
    "Trincomalee",
    "Kurunegala",
    "Puttalam",
    "Anuradhapura",
    "Polonnaruwa",
    "Badulla",
    "Monaragala",
    "Ratnapura",
    "Kegalle",
    "Kalmune",
]

# how the PDF spells them -> panel spelling
ALIASES = {
    "nuwara eliya": "NuwaraEliya",
    "nuwaraeliya": "NuwaraEliya",
    "hambantota": "Hambanthota",
    "hambanthota": "Hambanthota",
    "monaragala": "Monaragala",
    "moneragala": "Monaragala",
    "kalmunai": "Kalmune",
    "kalmune": "Kalmune",
    "mullaitivu": "Mullaitivu",
    "mullativu": "Mullaitivu",
    "kilinochchi": "Kilinochchi",
    "killinochchi": "Kilinochchi",
}
for _d in DISTRICTS:
    ALIASES.setdefault(_d.lower(), _d)

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )
}


def list_reports(limit: int = 40) -> list[dict]:
    """Available WER PDFs, newest first, as {vol, no, url}."""
    r = requests.get(INDEX_URL, headers=UA, timeout=120, verify=False)
    r.raise_for_status()
    out, seen = [], set()
    for url in re.findall(r'href="([^"]+\.pdf)"', r.text, re.I):
        m = re.search(r"Vol[_\s]*(\d+)[_\s]*no[_\s]*(\d+)", url, re.I)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"vol": key[0], "no": key[1], "url": url})
    out.sort(key=lambda d: (-d["vol"], -d["no"]))
    return out[:limit]


def fetch(rep: dict) -> Path:
    """Download one report, cached by volume and number."""
    WER_DIR.mkdir(parents=True, exist_ok=True)
    p = WER_DIR / f"vol{rep['vol']}_no{rep['no']}.pdf"
    if p.exists() and p.stat().st_size > 50_000:
        return p
    r = requests.get(rep["url"], headers=UA, timeout=300, verify=False)
    r.raise_for_status()
    p.write_bytes(r.content)
    return p


def _name_of(run: str) -> str | None:
    """District this run starts with, if any."""
    low = run.lower().strip()
    for alias, canon in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if low.startswith(alias):
            return canon
    return None


def _runs(page) -> list[tuple[float, str]]:
    """Vertical text runs, each reconstructed bottom-to-top."""
    cols: dict[int, list] = defaultdict(list)
    for c in page.chars:
        cols[round(c["x0"] / 3.0)].append(c)
    runs = []
    for cs in cols.values():
        cs.sort(key=lambda c: -c["top"])
        s = "".join(c["text"] for c in cs).strip()
        if s:
            runs.append((min(c["x0"] for c in cs), s))
    runs.sort()
    return runs


def district_rows(page) -> list[tuple[str, list[int]]]:
    """(district, [numbers]) pairs from one table page.

    Most districts arrive as a single run - name then its whole numeric row. A few
    (Badulla, Kilinochchi) split into a name run and a separate numeric run, and the
    name run can also pick up stray digits bleeding in from the page header. So the
    numbers are ALWAYS taken from the numeric run, never from the name run.
    """
    runs = _runs(page)
    out: list[tuple[str, list[int]]] = []
    used: set[int] = set()

    for i, (x, s) in enumerate(runs):
        name = _name_of(s)
        if not name:
            continue
        tail = s[len(name) :] if s.lower().startswith(name.lower()) else s
        nums = [int(n) for n in re.findall(r"\d+", tail)]
        if len(nums) >= 8:  # a complete row lives in this run
            out.append((name, nums))
            used.add(i)
            continue
        # otherwise borrow the adjacent numeric run - the name run is unreliable here
        best, best_dx = None, 6.0
        for j, (xj, sj) in enumerate(runs):
            if j == i or j in used or _name_of(sj):
                continue
            njs = [int(n) for n in re.findall(r"\d+", sj)]
            if len(njs) >= 8 and abs(xj - x) < best_dx:
                best, best_dx = (j, njs), abs(xj - x)
        if best:
            j, njs = best
            used.add(j)
            out.append((name, njs))
    return out


CAPTION = "Distribution of Notified Diseases"


def _month(tok: str) -> int | None:
    """Month number from a full or abbreviated name."""
    t = tok.lower().strip(".")
    if t in MONTHS:
        return MONTHS[t]
    for name, num in MONTHS.items():
        if name.startswith(t) and len(t) >= 3:
            return num
    return None


def _week_dates(text: str) -> tuple[date, date] | None:
    """Data week from the Table 1 caption ONLY.

    Anchoring on the caption matters: the cover of each issue carries a *different*
    date range one week later, and matching that instead silently shifts every
    reading by seven days.

    Two caption forms occur:
        22nd - 28th June 2026            (one month)
        27th Apr - 03rd May 2026         (spanning a month boundary)
    """
    line = next((ln for ln in text.splitlines() if CAPTION in ln), None)
    if line is None:
        return None

    # Month tokens need >= 3 letters, otherwise "22nd - 28th June" matches this
    # pattern with "d" as the first month and the single-month form never runs.
    two = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,})\.?\s*[-–—]\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,})\.?\s+(\d{4})",
        line,
    )
    if two:
        m1, m2 = _month(two.group(2)), _month(two.group(4))
        if m1 and m2:
            d1, d2, yr = int(two.group(1)), int(two.group(3)), int(two.group(5))
            y1 = yr - 1 if m1 == 12 and m2 == 1 else yr
            try:
                return date(y1, m1, d1), date(yr, m2, d2)
            except ValueError:
                return None
        # fall through to the single-month form rather than giving up

    one = re.search(r"(\d{1,2})\w{0,2}\s*[-–—]\s*(\d{1,2})\w{0,2}\s+([A-Za-z]+)\.?\s+(\d{4})", line)
    if not one:
        return None
    d1, d2, mi, y = (int(one.group(1)), int(one.group(2)), _month(one.group(3)), int(one.group(4)))
    if not mi:
        return None
    try:
        end = date(y, mi, d2)
    except ValueError:
        return None
    if d1 > d2:  # week spans a month boundary
        pm, py = (mi - 1, y) if mi > 1 else (12, y - 1)
        try:
            start = date(py, pm, d1)
        except ValueError:
            return None
    else:
        start = date(y, mi, d1)
    return start, end


def parse(pdf_path: Path) -> pd.DataFrame:
    """District dengue counts for the week this report covers."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "Distribution of Notified Diseases" not in text:
                continue
            wk = _week_dates(text)
            rows = [
                {"district": name, "cases": nums[0], "cumulative": nums[1]}
                for name, nums in district_rows(page)
            ]
            if len(rows) >= 20 and wk:
                df = pd.DataFrame(rows).drop_duplicates("district", keep="first")
                df["week_start"] = pd.Timestamp(wk[0])
                df["week_end"] = pd.Timestamp(wk[1])
                df["source"] = pdf_path.name
                return df
    return pd.DataFrame()


def validate_against(parsed: pd.DataFrame, known: pd.DataFrame, tol: float = 0.02) -> dict:
    """Compare parsed weeks against the denguedatahub weeks that overlap.

    Surveillance figures get revised, so exact equality is not required - but a
    parser reading the wrong column would be wildly off, and that is what this
    is here to catch.
    """
    k = known.copy()
    k["week_start"] = pd.to_datetime(k["start.date"], format="mixed", errors="coerce")
    k["district"] = k["district"].map(lambda d: ALIASES.get(str(d).lower(), d))
    m = parsed.merge(
        k[["district", "week_start", "cases"]],
        on=["district", "week_start"],
        suffixes=("_pdf", "_hub"),
    )
    if m.empty:
        return {"overlapping_rows": 0, "verdict": "NO OVERLAP"}
    m["abs_diff"] = (m.cases_pdf - m.cases_hub).abs()
    m["rel_diff"] = m.abs_diff / m.cases_hub.clip(lower=1)
    return {
        "overlapping_rows": len(m),
        "exact_match_rate": float((m.abs_diff == 0).mean()),
        "within_tolerance_rate": float((m.rel_diff <= tol).mean()),
        "max_abs_diff": int(m.abs_diff.max()),
        "verdict": "PASS" if float((m.rel_diff <= tol).mean()) >= 0.95 else "FAIL",
        "worst": m.nlargest(5, "abs_diff")[
            ["district", "week_start", "cases_pdf", "cases_hub"]
        ].to_dict("records"),
    }
