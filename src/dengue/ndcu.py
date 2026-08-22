"""Parse the National Dengue Control Unit Weekly Dengue Update for district counts.

This is the freshest of the three Sri Lankan sources:

    denguedatahub   structured, reliable, lags months
    WER PDFs        Epidemiology Unit, lags ~9 weeks
    NDCU weekly     this module, lags ~1 week

Table 1 of each issue gives, per district, six figures:

    <district> <2025 wk n-1> <2025 wk n> <2026 wk n-1> <2026 wk n> <2025 cum> <2026 cum>

so the current week is the FOURTH number. The table is interleaved with narrative
text in the extracted layer, but district rows are unambiguous once anchored on a
known district name.

The report states its own weekly total, which gives a self-check no other source
offers: `parse` returns that stated total alongside the rows, and `validate` refuses
the issue when the parsed district figures do not sum to it.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
NDCU_DIR = RAW / "ndcu"
INDEX_URL = "https://www.dengue.health.gov.lk/weekly-report/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}

# NDCU spelling -> the spelling the panel uses
ALIASES = {
    "colombo": "Colombo", "gampaha": "Gampaha", "kalutara": "Kalutara",
    "kandy": "Kandy", "matale": "Matale",
    "nuwara eliya": "NuwaraEliya", "nuwaraeliya": "NuwaraEliya",
    "galle": "Galle", "hambantota": "Hambanthota", "hambanthota": "Hambanthota",
    "matara": "Matara", "jaffna": "Jaffna", "kilinochchi": "Kilinochchi",
    "mannar": "Mannar", "vavuniya": "Vavuniya", "mullaitivu": "Mullaitivu",
    "batticaloa": "Batticaloa", "ampara": "Ampara", "trincomalee": "Trincomalee",
    "kalmunai": "Kalmune", "kalmune": "Kalmune",
    "kurunegala": "Kurunegala", "puttalam": "Puttalam",
    "anuradhapura": "Anuradhapura", "polonnaruwa": "Polonnaruwa",
    "badulla": "Badulla", "monaragala": "Monaragala", "moneragala": "Monaragala",
    "ratnapura": "Ratnapura", "kegalle": "Kegalle",
}

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}

# "Colombo 115 117 742* 623 8974 21987" - Nil stands in for zero, * is a footnote.
#
# Anchored on the known district names rather than on line boundaries: in the
# extracted layer the table is interleaved with the narrative, so a district row can
# be appended to a sentence ("...Ratnapura Kandy 106 85 332 273 3045 6636") or carry
# a stray chart-axis character in front of it ("n Mannar 3 1 1 1 116 92"). Matching
# the name itself sidesteps both.
_NAMES = sorted(ALIASES, key=len, reverse=True)
_NUM = r"(?:\d[\d,]*\*?|Nil)"
ROW = re.compile(
    r"(?<![A-Za-z])(" + "|".join(n.replace(" ", r"\s+") for n in _NAMES) + r")\*?\s+"
    r"(" + _NUM + r"(?:[ \t]+" + _NUM + r"){5})(?![ \t]*[\d,])",
    re.I)


def list_reports(limit: int = 20) -> list[dict]:
    """Weekly Dengue Update PDFs, newest first, as {year, week, url}."""
    r = requests.get(INDEX_URL, headers=UA, timeout=120, verify=False)
    r.raise_for_status()
    out, seen = [], set()
    for url in re.findall(r'href="([^"]+\.pdf)"', r.text, re.I):
        m = re.search(r"(20\d\d)[-_\s]*Week[-_\s]*(\d{1,2})", url, re.I)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"year": key[0], "week": key[1], "url": url})
    out.sort(key=lambda d: (-d["year"], -d["week"]))
    return out[:limit]


def fetch(rep: dict) -> Path:
    """Download one issue, cached by year and week."""
    NDCU_DIR.mkdir(parents=True, exist_ok=True)
    p = NDCU_DIR / f"{rep['year']}_week{rep['week']:02d}.pdf"
    if p.exists() and p.stat().st_size > 50_000:
        return p
    r = requests.get(rep["url"], headers=UA, timeout=300, verify=False)
    r.raise_for_status()
    p.write_bytes(r.content)
    return p


def _num(tok: str) -> int:
    t = tok.strip().rstrip("*").replace(",", "")
    return 0 if t.lower() == "nil" else int(t)


def _week_dates(text: str) -> tuple[date, date] | None:
    """From the cover line, e.g. 'Week 33 (10th - 16th August 2026)'."""
    m = re.search(r"Week\s+\d{1,2}\s*\(\s*(\d{1,2})\w{0,2}\s*(?:([A-Za-z]{3,})\.?\s*)?"
                  r"[-–—]\s*(\d{1,2})\w{0,2}\s*([A-Za-z]{3,})\.?\s*(20\d\d)", text)
    if not m:
        return None
    d1, m1_raw, d2, m2_raw, yr = m.groups()
    m2 = MONTHS.get(m2_raw.lower()) or next(
        (v for k, v in MONTHS.items() if k.startswith(m2_raw.lower()[:3])), None)
    if not m2:
        return None
    yr = int(yr)
    if m1_raw:
        m1 = MONTHS.get(m1_raw.lower()) or next(
            (v for k, v in MONTHS.items() if k.startswith(m1_raw.lower()[:3])), None)
    else:
        m1 = m2
    if not m1:
        return None
    y1 = yr - 1 if m1 == 12 and m2 == 1 else yr
    try:
        return date(y1, m1, int(d1)), date(yr, m2, int(d2))
    except ValueError:
        return None


def parse(pdf_path: Path) -> pd.DataFrame:
    """District dengue counts for the week this issue reports.

    The returned frame carries `stated_total`, the figure the report itself gives
    for the week, so callers can check the parse against the document.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""

    wk = _week_dates(text)
    if not wk:
        return pd.DataFrame()

    total_m = re.search(r"([\d,]+)\s+suspected dengue cases", text, re.I)
    stated_total = _num(total_m.group(1)) if total_m else None

    rows = []
    for name_raw, nums_raw in ROW.findall(text):
        canon = ALIASES.get(re.sub(r"\s+", " ", name_raw).strip().lower())
        if not canon:
            continue
        nums = [_num(t) for t in nums_raw.split()]
        if len(nums) < 6:
            continue
        rows.append({"district": canon, "cases": nums[3],          # 2026 current week
                     "prev_week": nums[2], "cumulative": nums[5]})

    if len(rows) < 20:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("district", keep="first")
    df["week_start"] = pd.Timestamp(wk[0])
    df["week_end"] = pd.Timestamp(wk[1])
    df["stated_total"] = stated_total
    df["source"] = pdf_path.name
    return df


def validate(parsed: pd.DataFrame, tol: float = 0.02) -> dict:
    """Check each issue against the total the report states for itself."""
    if parsed.empty:
        return {"verdict": "EMPTY", "issues": 0}
    rows = []
    for (wk, total), g in parsed.groupby(["week_start", "stated_total"], dropna=False):
        summed = int(g.cases.sum())
        if pd.isna(total):
            rows.append({"week": wk, "stated": None, "summed": summed, "ok": None})
            continue
        rel = abs(summed - total) / max(int(total), 1)
        rows.append({"week": wk, "stated": int(total), "summed": summed,
                     "districts": len(g), "rel_diff": rel, "ok": bool(rel <= tol)})
    checked = [r for r in rows if r["ok"] is not None]
    passed = [r for r in checked if r["ok"]]
    return {"issues": len(rows), "checked": len(checked), "passed": len(passed),
            "pass_rate": (len(passed) / len(checked)) if checked else None,
            "verdict": "PASS" if checked and len(passed) == len(checked) else
                       ("PARTIAL" if passed else "FAIL"),
            "detail": rows}
