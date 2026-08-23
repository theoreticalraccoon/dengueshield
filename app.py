"""DengueShield - dengue screening + outbreak early warning.

  1. Patient assessment - CBC screening, complication risk, optional district context
  2. Outbreak forecast  - district forecasts: continuation AND emergence
  3. About the models   - what it does, how well, where it fails, what it was built on

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODELS, REPORTS, PROC = ROOT / "models", ROOT / "reports", ROOT / "data" / "processed"

st.set_page_config(page_title="DengueShield", page_icon=":material/coronavirus:",
                   layout="wide")

# ---------------------------------------------------------------- design tokens
INK, MUTED, FAINT = "#16211C", "#5C6B63", "#8A978F"
PAPER, SURFACE, RULE = "#F7F8F5", "#FFFFFF", "#DCE3DA"
ACCENT = "#0F5F63"
# Risk semantics, saturated enough to carry a map but chosen against measured
# contrast rather than by eye: each is the most vivid option in its hue family that
# still clears WCAG AA against either white or ink, so text printed on a band is
# always readable. `readable_on` picks which.
BAND_COLOR = {"Low": "#1A8754", "Moderate": "#C68A0E",
              "High": "#D2691E", "Very High": "#C42A1C"}


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def readable_on(background: str) -> str:
    """White or ink, whichever has more contrast against this background."""
    lb = _luminance(background)
    white = (1.05) / (lb + 0.05)
    ink = (lb + 0.05) / (_luminance("#16211C") + 0.05)
    return "#FFFFFF" if white >= ink else "#16211C"

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="st-"], .stMarkdown {{
  font-family: 'Public Sans', -apple-system, 'Segoe UI', sans-serif;
}}

/* Streamlit draws its chrome (sidebar collapse, expander chevrons, dataframe
   controls) as Material Symbols LIGATURES. The broad rule above would otherwise
   override that font and print the ligature names as literal text -
   "keyboard_double_arrow_left", "arrow_drop_down". Put the icon font back. */
[data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"],
span[class*="material-symbols"], span[class*="material-icons"],
.material-icons, .material-icons-outlined, .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
  font-weight: normal !important; font-style: normal !important;
  letter-spacing: normal !important; text-transform: none !important;
  white-space: nowrap; word-wrap: normal; direction: ltr;
  -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
}}
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 2.6rem; max-width: 1180px; }}

h1, h2, h3 {{
  font-family: Newsreader, Georgia, serif !important;
  font-weight: 500 !important; letter-spacing: -.012em; color: {INK};
}}
h1 {{ font-size: 2.5rem !important; line-height: 1.08 !important; margin-bottom: .1rem !important; }}
h2 {{ font-size: 1.5rem !important; margin-top: .4rem !important; }}
h3 {{ font-size: 1.06rem !important; font-weight: 600 !important;
     font-family: 'Public Sans', sans-serif !important; letter-spacing: 0; }}

/* quiet the sidebar into a nav rail */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] .stRadio label {{ font-size: .95rem; }}

/* figures read as data, not decoration */
[data-testid="stMetricValue"] {{
  font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
  font-size: 1.55rem; font-weight: 500; color: {INK};
}}
[data-testid="stMetricLabel"] {{
  font-size: .7rem; letter-spacing: .09em; text-transform: uppercase; color: {FAINT};
}}
[data-testid="stMetricDelta"] {{ font-size: .78rem; }}

/* replace Streamlit's stack of coloured callouts with hairline notes */
[data-testid="stAlert"] {{
  background: {SURFACE}; border: none; border-left: 2px solid {RULE};
  border-radius: 0; padding: .7rem 1rem; color: {MUTED}; font-size: .9rem;
}}
[data-testid="stAlert"] p {{ color: {MUTED}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{
  padding: 0 0 .55rem 0; font-size: .92rem; color: {MUTED};
}}
.stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}

.stDataFrame {{ border: 1px solid {RULE}; }}
code {{ font-family: 'IBM Plex Mono', monospace; font-size: .84em; }}
hr {{ border-color: {RULE}; }}

/* --- custom components --- */
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; font-weight: 500;
  letter-spacing: .14em; text-transform: uppercase; color: {FAINT}; margin-bottom: .2rem;
}}
.lede {{ color: {MUTED}; font-size: 1rem; max-width: 62ch; margin-bottom: .2rem; }}

.stamp {{
  display: flex; gap: 1.8rem; align-items: baseline; flex-wrap: wrap;
  border-left: 2px solid {ACCENT}; background: {SURFACE};
  padding: .7rem 1.1rem; margin: .1rem 0 1.1rem;
}}
.stamp .k {{ font-size: .66rem; letter-spacing: .12em; text-transform: uppercase;
             color: {FAINT}; display: block; }}
.stamp .v {{ font-family: 'IBM Plex Mono', monospace; font-size: .95rem; color: {INK};
             font-variant-numeric: tabular-nums; }}

.verdict {{
  font-family: Newsreader, Georgia, serif; font-size: 1.5rem; line-height: 1.25;
  margin: .3rem 0 .1rem;
}}
.verdict .dot {{ display: inline-block; width: .55rem; height: .55rem;
                 border-radius: 50%; margin-right: .5rem; vertical-align: middle; }}

.pill {{
  display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
  font-weight: 500; letter-spacing: .08em; text-transform: uppercase;
  padding: .16rem .5rem; border: 1px solid currentColor; border-radius: 2px;
}}
.finding {{ border-top: 1px solid {RULE}; padding: .95rem 0; }}
.finding b {{ color: {INK}; }}
.finding p {{ color: {MUTED}; font-size: .93rem; margin: .25rem 0 0; max-width: 60ch; }}
.note {{ color: {MUTED}; font-size: .88rem; max-width: 66ch; }}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

DISCLAIMER = ("Not a medical diagnosis. Clinical and laboratory evaluation is required. "
              "This does not provide medical clearance and must not be used to rule out "
              "dengue.")

PLOT_LAYOUT = {
    "font": {"family": "Public Sans, sans-serif", "size": 12, "color": MUTED},
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": {"gridcolor": RULE, "zerolinecolor": RULE},
    "yaxis": {"gridcolor": RULE, "zerolinecolor": RULE},
    "margin": {"t": 30, "b": 20, "l": 10, "r": 10},
}


@st.cache_resource
def load_models():
    out = {}
    for key, fn in [("screening", "model1_screening.joblib"),
                    ("peds", "peds_complications.joblib"),
                    ("sl_continuation", "srilanka_outbreak.joblib"),
                    ("sl_emergence", "srilanka_emergence.joblib")]:
        p = MODELS / fn
        out[key] = joblib.load(p) if p.exists() else None
    return out


@st.cache_data
def load_reports():
    def _csv(n):
        p = REPORTS / n
        return pd.read_csv(p) if p.exists() else None

    def _json(n):
        p = REPORTS / n
        return json.loads(p.read_text()) if p.exists() else None

    hist = PROC / "srilanka_history.parquet"
    em = ROOT / "experiments" / "emergence_v1" / "emergence_results.csv"
    return {
        "dual": _csv("srilanka_dual_risk.csv"),
        "history": pd.read_parquet(hist) if hist.exists() else None,
        "medians": _json("cbc_population_medians.json") or {},
        "audit": _csv("dataset_audit.csv"),
        "m1_ops": _json("model1_operating_points.json"),
        "robust": _json("model2_robustness.json"),
        "info_abl": _csv("model2_information_ablation.csv"),
        "spatial": _csv("model2_spatial_holdout.csv"),
        "calib": _json("model2_calibration_errors.json"),
        "rel_raw": _csv("model2_reliability_raw.csv"),
        "rel_cal": _csv("model2_reliability_calibrated.csv"),
        "sl_calib": _json("srilanka_calibration.json"),
        "transfer": _csv("srilanka_transfer.csv"),
        "ablation_peds": _csv("ablation_peds.csv"),
        "m2_head": _csv("model2_headtohead.csv"),
        "emergence": pd.read_csv(em) if em.exists() else None,
        "delay": _csv("reporting_delay_stress.csv"),
        "freshness": _json("data_freshness.json"),
        "situation": _csv("srilanka_latest_situation.csv"),
        "sit_meta": _json("situation_freshness.json"),
    }


M, R = load_models(), load_reports()


def header(title: str, eyebrow: str = "", lede: str = "") -> None:
    if eyebrow:
        st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def finding(title: str, body: str) -> None:
    st.markdown(f'<div class="finding"><b>{title}</b><p>{body}</p></div>',
                unsafe_allow_html=True)


def build_cbc_row(vals: dict) -> dict:
    """Fill any unsupplied CBC field from the cohort median, then derive ratios."""
    row = dict(R["medians"])
    row.update(vals)
    neu, lym = row["Neutrophils(%)"], row["Lymphocytes(%)"]
    plt_c, wbc = row["Total Platelet Count(/cumm)"], row["Total WBC count(/cumm)"]
    row["NLR"] = neu / max(lym, 1)
    row["PLR"] = plt_c / max(lym, 1)
    row["PLT_WBC_ratio"] = plt_c / max(wbc, 1)
    row["MPV_PLT_ratio"] = row["MPV(fl)"] / max(plt_c / 1000, 1e-6)
    return row


def screen_patient(vals: dict):
    b = M["screening"]
    p = float(b["model"].predict_proba(pd.DataFrame([build_cbc_row(vals)])[b["features"]])[0, 1])
    return p, b.get("threshold_sens90", 0.5)


def probability_bar(p: float, thr: float, label: str):
    """A quiet horizontal probability read-out with the decision threshold marked.

    Deliberately not a speedometer gauge - the threshold is the only reference
    point that matters here, and a bar states it without the dashboard theatrics.
    """
    colour = BAND_COLOR["Very High"] if p >= thr else BAND_COLOR["Low"]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, line_width=0, fillcolor="#EDF0EA")
    fig.add_shape(type="rect", x0=0, x1=p, y0=0, y1=1, line_width=0, fillcolor=colour)
    fig.add_shape(type="line", x0=thr, x1=thr, y0=-0.25, y1=1.25,
                  line={"color": INK, "width": 2})
    fig.add_annotation(x=thr, y=1.75, text=f"threshold {thr:.0%}", showarrow=False,
                       font={"size": 11, "color": MUTED, "family": "IBM Plex Mono"})
    fig.add_annotation(x=0, y=-1.1, text=label, showarrow=False, xanchor="left",
                       font={"size": 11, "color": FAINT})
    fig.add_annotation(x=1, y=-1.1, text="100%", showarrow=False, xanchor="right",
                       font={"size": 11, "color": FAINT})
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[-1.6, 2.2], visible=False)
    fig.update_layout(height=110, margin={"t": 26, "b": 6, "l": 0, "r": 0},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def verdict(text: str, high: bool) -> None:
    colour = BAND_COLOR["Very High"] if high else BAND_COLOR["Low"]
    st.markdown(f'<p class="verdict"><span class="dot" style="background:{colour}"></span>'
                f'{text}</p>', unsafe_allow_html=True)


st.sidebar.markdown('<p class="eyebrow" style="margin-bottom:.6rem">DengueShield</p>',
                    unsafe_allow_html=True)
screen = st.sidebar.radio(
    "Screen", ["Patient assessment", "Outbreak forecast", "About the models"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="note" style="font-size:.82rem">Decision support for prioritisation. '
    'Not a diagnostic device. RT-PCR/NAAT and antigen testing remain the clinical '
    'standard.</p>', unsafe_allow_html=True)


# ========================================================= 1. PATIENT ASSESSMENT
if screen == "Patient assessment":
    header("Patient assessment", "Individual risk",
           "Two independent estimates from routine bloodwork.")
    tab1, tab2 = st.tabs(["Dengue screening", "Complication risk"])

    with tab1:
        b = M["screening"]
        if b is None:
            st.error("Screening model not found. Run finalize_model1.py.")
        else:
            mode = st.radio("Entry mode", ["Quick", "Full blood count"], horizontal=True)
            vals = {}
            if mode == "Quick":
                c1, c2, c3 = st.columns(3)
                vals["Age"] = c1.number_input("Age", 1, 100, 32)
                vals["Gender"] = 1 if c1.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
                vals["Total Platelet Count(/cumm)"] = c2.number_input(
                    "Platelet count (/cumm)", 5_000, 600_000, 145_000, 1_000)
                vals["Total WBC count(/cumm)"] = c2.number_input(
                    "Total WBC (/cumm)", 500, 30_000, 4_800, 100)
                vals["Hemoglobin(g/dl)"] = c3.number_input(
                    "Haemoglobin (g/dl)", 5.0, 22.0, 14.2, 0.1)
                vals["HCT(%)"] = c3.number_input("Haematocrit (%)", 20.0, 60.0, 43.0, 0.1)
                st.markdown('<p class="note">Unentered values default to the study '
                            'cohort median.</p>', unsafe_allow_html=True)
            else:
                c1, c2, c3, c4 = st.columns(4)
                vals["Age"] = c1.number_input("Age", 1, 100, 32)
                vals["Gender"] = 1 if c1.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
                vals["Hemoglobin(g/dl)"] = c1.number_input("Haemoglobin (g/dl)", 5.0, 22.0, 14.2, 0.1)
                vals["RBC"] = c1.number_input("RBC (10⁶/µl)", 2.0, 8.0, 4.8, 0.1)
                vals["HCT(%)"] = c1.number_input("Haematocrit (%)", 20.0, 60.0, 43.0, 0.1)
                vals["Total Platelet Count(/cumm)"] = c2.number_input(
                    "Platelet count (/cumm)", 5_000, 600_000, 145_000, 1_000)
                vals["Total WBC count(/cumm)"] = c2.number_input(
                    "Total WBC (/cumm)", 500, 30_000, 4_800, 100)
                vals["Neutrophils(%)"] = c2.number_input("Neutrophils (%)", 0, 100, 55)
                vals["Lymphocytes(%)"] = c2.number_input("Lymphocytes (%)", 0, 100, 35)
                vals["Monocytes(%)"] = c2.number_input("Monocytes (%)", 0, 50, 6)
                vals["Eosinophils(%)"] = c3.number_input("Eosinophils (%)", 0, 30, 2)
                vals["MCV(fl)"] = c3.number_input("MCV (fl)", 50.0, 130.0, 88.0, 0.1)
                vals["MCH(pg)"] = c3.number_input("MCH (pg)", 15.0, 45.0, 29.0, 0.1)
                vals["MCHC(g/dl)"] = c3.number_input("MCHC (g/dl)", 25.0, 40.0, 32.5, 0.1)
                vals["RDW-CV(%)"] = c3.number_input("RDW-CV (%)", 8.0, 25.0, 13.0, 0.1)
                vals["MPV(fl)"] = c4.number_input("MPV (fl)", 5.0, 20.0, 10.5, 0.1)
                vals["PDW(%)"] = c4.number_input("PDW (%)", 5.0, 25.0, 15.0, 0.1)
                vals["PCT(%)"] = c4.number_input("PCT (%)", 0.0, 1.0, 0.15, 0.01)

            dual = R["dual"]
            district = None
            if dual is not None:
                district = st.selectbox("District (optional, adds local outbreak context)",
                                        ["Not specified", *dual.district.tolist()])

            if st.button("Assess", type="primary", key="screen_btn"):
                p, thr = screen_patient(vals)
                flag = p >= thr
                st.markdown("---")
                a, bcol = st.columns([1, 2])
                with a:
                    st.metric("Screening probability", f"{p:.0%}")
                    verdict("Prioritise dengue testing" if flag
                            else "Lower testing priority", flag)
                bcol.plotly_chart(
                    probability_bar(p, thr, "model-estimated probability"),
                    width="stretch")

                if dual is not None and district and district != "Not specified":
                    row = dual[dual.district == district].iloc[0]
                    geo = (row.continuation_risk if row.currently_in_outbreak
                           else (row.emergence_risk if pd.notna(row.emergence_risk)
                                 else row.continuation_risk))
                    k = st.columns(4)
                    k[0].metric("District", district)
                    k[1].metric("Cases this week", f"{int(row.casos):,}")
                    k[2].metric("In outbreak", "Yes" if row.currently_in_outbreak else "No")
                    k[3].metric("14-day outbreak risk", f"{geo:.0%}")
                    msg = {
                        (True, True): f"Elevated patient probability in a high-risk "
                                      f"district. Confirmatory testing; vector-control "
                                      f"inspection and community advisory for {district}.",
                        (True, False): "Elevated patient probability, lower geographic "
                                       "risk. Confirmatory testing for this patient.",
                        (False, True): f"Lower patient probability, but {district} is at "
                                       f"elevated risk. Maintain district surveillance.",
                        (False, False): "Neither signal elevated. Routine care and "
                                        "routine surveillance.",
                    }[(flag, geo >= 0.5)]
                    st.markdown(f'<p class="note">{msg}</p>', unsafe_allow_html=True)

                ops = R["m1_ops"]
                tail = (f" ROC-AUC {ops['at_0.5']['roc_auc']:.3f}, accuracy "
                        f"{ops['at_0.5']['accuracy']:.3f} against a 0.685 majority-class "
                        f"baseline." if ops else "")
                st.markdown(f'<p class="note" style="margin-top:1rem">{DISCLAIMER}'
                            f'{tail} Use this to prioritise testing, never to rule dengue '
                            f'out.</p>', unsafe_allow_html=True)

    with tab2:
        b = M["peds"]
        if b is None:
            st.error("Complication model not found. Run finalize_peds.py.")
        else:
            st.markdown('<p class="note">For a patient already known or suspected to have '
                        'dengue. Estimated relative to the training population of 303 '
                        'paediatric admissions.</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### Course")
                p_age = st.selectbox("Age band", [1, 2], format_func=lambda v: f"Band {v}")
                p_sex = st.selectbox("Sex", ["Male", "Female"], key="psex")
                fever = st.number_input("Duration of fever (days)", 0, 21, 4)
                bp = st.selectbox("BP at admission", ["Normal", "Low"])
                organ = st.selectbox("Organomegaly", ["No", "Yes"])
            with c2:
                st.markdown("### Symptoms")
                sx = {s: st.checkbox(s.capitalize()) for s in
                      ["headache", "myalgia", "abdominal pain", "rash",
                       "vomiting", "breathlessness", "bleeding"]}
            with c3:
                st.markdown("### Laboratory")
                p_hb = st.number_input("Hb (g/dl)", 5.0, 22.0, 13.2, 0.1, key="phb")
                pcv = st.number_input("PCV (%)", 20.0, 60.0, 40.0, 0.1)
                lo_wbc = st.number_input("Lowest WBC", 300, 30_000, 3_500, 100)
                hi_wbc = st.number_input("Highest WBC", 300, 40_000, 5_100, 100)
                p_plt = st.number_input("Platelet count", 5_000, 600_000, 90_000, 1_000, key="pplt")
                alt = st.number_input("ALT (U/L)", 1.0, 2000.0, 26.0, 1.0)
                creat = st.number_input("Creatinine (mg/dl)", 0.1, 10.0, 0.5, 0.1)
                ferr = st.number_input("Ferritin (ng/ml)", 1.0, 30000.0, 763.0, 10.0)

            if st.button("Assess complication risk", type="primary", key="peds_btn"):
                row = {"HB": p_hb, "PCV": pcv, "LOWEST WBC": lo_wbc, "HIGHEST WBC": hi_wbc,
                       "PLATELET COUNT": p_plt, "AGE": p_age,
                       "GENDER": 1 if p_sex == "Male" else 0,
                       "HEADACHE": int(sx["headache"]), "MYALGIA": int(sx["myalgia"]),
                       "ABDOMINAL PAIN": int(sx["abdominal pain"]), "RASH": int(sx["rash"]),
                       "VOMITING": int(sx["vomiting"]),
                       "BREATHLESSNESS": int(sx["breathlessness"]),
                       "BLEEDING": int(sx["bleeding"]),
                       "ORGANOMEGALY": 1 if organ == "Yes" else 0,
                       "BP AT ADMISSION": 1 if bp == "Low" else 0,
                       "DURATION OF FEVER": fever, "ALT": alt,
                       "CREATININE": creat, "FERRITIN": ferr}
                p = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
                thr = b["threshold_sens90"]
                mm = b["metrics"]
                st.markdown("---")
                a, bcol = st.columns([1, 2])
                with a:
                    st.metric("Complication risk", f"{p:.0%}")
                    verdict("Monitor closely" if p >= thr else "Routine care", p >= thr)
                bcol.plotly_chart(probability_bar(p, thr, "estimated risk"),
                                  width="stretch")
                st.markdown(
                    f'<p class="note">Operating at {mm["sensitivity"]:.0%} sensitivity and '
                    f'{mm["specificity"]:.0%} specificity, NPV {mm["npv"]:.2f}. Tuned for '
                    f'sensitivity: missing a deteriorating patient costs more than one '
                    f'extra observation. {DISCLAIMER}</p>', unsafe_allow_html=True)


# ========================================================== 2. OUTBREAK FORECAST
elif screen == "Outbreak forecast":
    dual = R["dual"]
    if dual is None:
        st.error("No forecasts. Run experiments/emergence_v1/finalize_emergence_srilanka.py.")
    else:
        header("Sri Lanka outbreak forecast", "14-day early warning",
               "Two models per district: whether an existing outbreak continues, "
               "and whether a new one begins.")

        fr = R["freshness"] or {}
        wk = pd.to_datetime(dual.week_start).max()
        updated = str(fr.get("refreshed_at", ""))[:10]
        st.markdown(
            '<div class="stamp">'
            f'<div><span class="k">Data last updated</span>'
            f'<span class="v">{pd.Timestamp(updated).strftime("%d %b %Y") if updated else "unknown"}</span></div>'
            f'<div><span class="k">Forecast surveillance week</span>'
            f'<span class="v">{wk.strftime("%d %b %Y")}</span></div>'
            + (f'<div><span class="k">Latest reported week</span>'
               f'<span class="v">{pd.Timestamp(sm["latest_week_start"]).strftime("%d %b %Y")}</span></div>'
               if (sm := R["sit_meta"]) else "")
            + '</div>', unsafe_allow_html=True)

        d = dual.copy()
        d["status"] = np.where(d.currently_in_outbreak, "In outbreak", "Not in outbreak")
        d["headline_risk"] = np.where(d.currently_in_outbreak, d.continuation_risk,
                                      d.emergence_risk.fillna(d.continuation_risk))
        d["band"] = pd.cut(d.headline_risk, [-0.01, 0.25, 0.5, 0.75, 1.01],
                           labels=["Low", "Moderate", "High", "Very High"])

        # ---- triage: the answer the page exists to give ----
        emg_thr = (M["sl_emergence"] or {}).get("threshold", 0.5)
        d["triage"] = np.select(
            [d.currently_in_outbreak,
             d.emergence_risk.notna() & (d.emergence_risk >= emg_thr),
             d.emergence_risk.notna()],
            ["Outbreak now", "Outbreak likely", "Clear"],
            default="Not assessable")
        TRIAGE = {
            "Outbreak now": (BAND_COLOR["Very High"],
                             "incidence is at or above the epidemic threshold"),
            "Outbreak likely": (BAND_COLOR["High"],
                                f"quiet now, but emergence risk is at or above {emg_thr:.0%}"),
            "Clear": (BAND_COLOR["Low"], "quiet, and no emergence signal"),
            "Not assessable": (FAINT,
                               "out of outbreak too recently for the emergence model, "
                               "which needs two consecutive quiet weeks"),
        }
        order = ["Outbreak now", "Outbreak likely", "Clear", "Not assessable"]
        counts = {g: int((d.triage == g).sum()) for g in order}

        # proportional bar - the whole country in one line
        bar = go.Figure()
        for g in order:
            if not counts[g]:
                continue
            bar.add_trace(go.Bar(
                x=[counts[g]], y=[""], orientation="h", name=f"{g} ({counts[g]})",
                marker_color=TRIAGE[g][0], hovertemplate=f"{g}: {counts[g]} districts<extra></extra>",
                text=[str(counts[g])], textposition="inside",
                insidetextfont={"color": readable_on(TRIAGE[g][0]),
                                "family": "IBM Plex Mono", "size": 13}))
        bar.update_layout(barmode="stack", height=96, showlegend=True,
                          legend={"orientation": "h", "y": -0.85, "x": 0, "title": ""},
                          margin={"t": 6, "b": 0, "l": 0, "r": 0},
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family": "Public Sans, sans-serif", "color": MUTED},
                          xaxis={"visible": False}, yaxis={"visible": False})
        st.plotly_chart(bar, width="stretch")

        cols = st.columns(4)
        for col, g in zip(cols, order, strict=True):
            colour, why = TRIAGE[g]
            rows = d[d.triage == g].sort_values("headline_risk", ascending=False)
            with col:
                st.markdown(
                    f'<div style="border-top:3px solid {colour};padding-top:.6rem">'
                    f'<span class="eyebrow" style="color:{colour}">{g}</span>'
                    f'<div style="font-family:\'IBM Plex Mono\';font-size:1.9rem;'
                    f'color:{INK};line-height:1.1">{counts[g]}</div>'
                    f'<p class="note" style="font-size:.8rem;margin-top:.2rem">{why}</p>'
                    "</div>", unsafe_allow_html=True)
                if rows.empty:
                    st.markdown('<p class="note" style="font-size:.82rem">'
                                "<em>No districts.</em></p>", unsafe_allow_html=True)
                else:
                    listing = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'gap:.6rem;font-size:.84rem;padding:.16rem 0;'
                        f'border-bottom:1px solid {RULE}">'
                        f'<span>{r.district}</span>'
                        f'<span style="font-family:\'IBM Plex Mono\';color:{MUTED}">'
                        f'{r.casos:,.0f}</span></div>'
                        for r in rows.itertuples())
                    st.markdown(listing, unsafe_allow_html=True)
                    st.markdown('<p class="note" style="font-size:.74rem;'
                                'margin-top:.35rem">cases this week</p>',
                                unsafe_allow_html=True)

        if counts["Outbreak likely"] == 0:
            st.markdown(
                '<p class="note" style="margin-top:.8rem">No district is currently '
                "flagged for a <em>new</em> outbreak. With most of the country already "
                "above the threshold there is little quiet ground left to flare, and the "
                "districts that recently subsided are not yet eligible for the emergence "
                "model.</p>", unsafe_allow_html=True)

        st.markdown("---")

        left, right = st.columns([1, 1])
        with left:
            fig = px.scatter_map(
                d, lat="lat", lon="lon", size="casos", color="band",
                color_discrete_map=BAND_COLOR, hover_name="district",
                hover_data={"casos": True, "p_inc100k": ":.1f", "status": True,
                            "continuation_risk": ":.2f", "emergence_risk": ":.2f",
                            "lat": False, "lon": False, "band": False},
                size_max=42, zoom=6.2, height=540,
                category_orders={"band": ["Very High", "High", "Moderate", "Low"]})
            fig.update_layout(map_style="carto-positron",
                              margin={"l": 0, "r": 0, "t": 0, "b": 0},
                              legend={"orientation": "h", "y": -0.04, "title": ""},
                              font={"family": "Public Sans, sans-serif"})
            st.plotly_chart(fig, width="stretch")
        with right:
            # Status is deliberately omitted: the triage panel above already groups
            # every district by it, and a blank Emergence cell says the same thing.
            # Dropping it leaves room to spell both probability columns out in full
            # so nothing needs scrolling to read.
            show = d[["district", "casos", "continuation_risk", "emergence_risk"]].copy()
            show.columns = ["District", "Cases", "Continuation", "Emergence"]
            st.dataframe(
                show.style.format({"Cases": "{:,.0f}", "Continuation": "{:.2f}",
                                   "Emergence": "{:.2f}"}, na_rep="—"),
                hide_index=True, width="stretch", height=500,
                column_config={
                    "District": st.column_config.TextColumn(width="small"),
                    "Cases": st.column_config.NumberColumn(width="small"),
                    "Continuation": st.column_config.NumberColumn(
                        width="small",
                        help="Probability an existing outbreak persists 14 days"),
                    "Emergence": st.column_config.NumberColumn(
                        width="small",
                        help="Probability a new outbreak begins within 1-4 weeks"),
                })
            st.markdown('<p class="note">Emergence is blank where a district is already '
                        'in outbreak. Status is in the panel above.</p>',
                        unsafe_allow_html=True)

        st.markdown("---")
        sel = st.selectbox("District detail", d.district.tolist())
        row = d[d.district == sel].iloc[0]
        c = st.columns(5)
        c[0].metric("Cases", f"{int(row.casos):,}")
        c[1].metric("Per 100k", f"{row.p_inc100k:.1f}")
        c[2].metric("Status", row.status)
        c[3].metric("Continuation", f"{row.continuation_risk:.0%}")
        c[4].metric("Emergence",
                    "n/a" if pd.isna(row.emergence_risk) else f"{row.emergence_risk:.0%}")

        hist = R["history"]
        if hist is not None:
            h = hist[hist.district == sel].sort_values("week_start").tail(160)
            f = go.Figure()
            f.add_trace(go.Bar(x=h.week_start, y=h.precip_total_semana, name="Rainfall (mm)",
                               marker_color="#C9D6CE", yaxis="y2"))
            f.add_trace(go.Scatter(x=h.week_start, y=h.casos, name="Cases",
                                   line={"width": 2, "color": ACCENT}))
            f.update_layout(**PLOT_LAYOUT, height=300,
                            yaxis2={"overlaying": "y", "side": "right",
                                    "showgrid": False, "title": "Rainfall (mm)"},
                            legend={"orientation": "h", "y": 1.14, "title": ""})
            f.update_yaxes(title="Weekly cases")
            st.plotly_chart(f, width="stretch")

        st.markdown('<p class="note">Predicted probabilities, not calibrated ones. The '
                    'ranking is more reliable than the number, and the model detects '
                    'escalation far better than emergence.</p>', unsafe_allow_html=True)


# ============================================================== 3. ABOUT
else:
    header("About the models", "Evidence",
           "Including the weaknesses. Everything below is out-of-sample.")
    t1, t2, t3, t4 = st.tabs(["What it does", "How well it works",
                              "Where it fails", "Data & methods"])

    with t1:
        st.markdown('<p class="note">Four separate questions, different populations, '
                    'very different difficulty. They are never combined into one '
                    '"dengue accuracy" figure.</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Question": "Does this patient appear to have dengue?",
             "Asked of": "one febrile patient", "Score": "ROC-AUC 0.681", "Verdict": "Modest"},
            {"Question": "Does this dengue patient need closer monitoring?",
             "Asked of": "one dengue admission", "Score": "ROC-AUC 0.874", "Verdict": "Strong"},
            {"Question": "Will this district's outbreak continue 14 days?",
             "Asked of": "any district", "Score": "PR-AUC 0.961", "Verdict": "Strong"},
            {"Question": "Will a new outbreak begin here?",
             "Asked of": "quiet districts only", "Score": "PR-AUC 0.583", "Verdict": "Harder"},
        ]), hide_index=True, width="stretch")
        st.markdown("### How a forecast would be used")
        st.code("""high risk flagged (14-day forecast)
        |
public-health authority reviews
        |
   +----+----+----------------+
   v         v                v
vector    community      testing capacity
control   advisory       prioritised
   +----+----+----------------+
        v
hospital preparedness raised""", language=None)
        st.markdown('<p class="note">Decision support for prioritisation. A flag is an '
                    'input to human judgement, not a trigger for automatic action.</p>',
                    unsafe_allow_html=True)

    with t2:
        st.markdown('<p class="note">The 2024–25 test set was locked before any later '
                    'development and never tuned against.</p>', unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("14-day forecast", "0.961", "PR-AUC")
        c[1].metric("vs persistence", "0.615", "+0.35")
        c[2].metric("Unseen districts", "0.955", "−0.007")
        c[3].metric("Calibration (ECE)", "0.0052", "−0.014", delta_color="inverse")

        rb = R["robust"]
        if rb and "rolling_origin" in rb:
            ro = pd.DataFrame(rb["rolling_origin"])
            f = go.Figure()
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.accuracy, name="Model",
                                   mode="lines+markers",
                                   line={"width": 2.5, "color": ACCENT}))
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.trivial_acc, name="Guess no outbreak",
                                   mode="lines+markers",
                                   line={"dash": "dot", "width": 1.5, "color": FAINT}))
            f.update_layout(**PLOT_LAYOUT, height=300,
                            title={"text": "Retrained each year, tested on the next",
                                   "y": 0.97, "yanchor": "top"},
                            legend={"orientation": "h", "y": -0.18, "x": 0,
                                    "title": ""})
            f.update_yaxes(range=[0.7, 1.0], title="Accuracy")
            st.plotly_chart(f, width="stretch")

        finding("Attribution is not incremental value",
                "SHAP credits about half the model's reasoning to climate and land cover, "
                "but removing all of it costs +0.006 PR-AUC, and environment alone (0.597) "
                "scores worse than persistence (0.760). The skill is epidemiological "
                "momentum. Ask the same features about outbreak <i>emergence</i> and they "
                "add +0.089 — a 15× difference.")
        ia = R["info_abl"]
        if ia is not None:
            f = px.bar(ia.sort_values("pr_auc"), x="pr_auc", y="condition",
                       orientation="h", text="pr_auc", height=210,
                       labels={"pr_auc": "PR-AUC", "condition": ""})
            f.update_traces(texttemplate="%{text:.3f}", marker_color=ACCENT,
                            textfont={"family": "IBM Plex Mono"})
            f.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(f, width="stretch")

        with st.expander("Full validation battery"):
            st.dataframe(pd.DataFrame(
                [("Temporal holdout (locked 2024–25)", "PR-AUC 0.961 · 96.7% acc"),
                 ("Rolling-origin backtest, 7 years", "mean 0.818 · every year >90%"),
                 ("Spatial holdout, unseen municipalities", "0.955 vs 0.962 seen"),
                 ("Leave-whole-states-out", "mean 0.872 (spread is prevalence)"),
                 ("Horizon sweep (2 / 4 / 8 weeks)", "0.961 / 0.919 / 0.813"),
                 ("Shuffled-label control", "collapses to 0.124 (chance 0.144)"),
                 ("Temporal-leakage audit", "no lag matched a future value"),
                 ("Reporting-delay stress (2 weeks)", "0.961 → 0.916")],
                columns=["Test", "Result"]), hide_index=True, width="stretch")
            if R["m2_head"] is not None:
                st.dataframe(R["m2_head"][["model", "pr_auc", "accuracy", "recall",
                                           "precision", "brier"]].round(4),
                             hide_index=True, width="stretch")

    with t3:
        finding("It tracks trajectory, not emergence",
                "Recall is 0.92 where transmission is already elevated and near zero "
                "where it is low — arguably the exact situation where early warning "
                "would matter most.")
        finding("Momentum carries through subsiding outbreaks",
                "54.5% of false alarms are districts already at epidemic level, against "
                "1.5% of true negatives — 37× more likely.")
        finding("Screening from a blood count is weak",
                "ROC-AUC 0.681. It prioritises testing; it cannot rule dengue out.")
        finding("It does not transfer between countries",
                "The same model scoring 0.910 on Brazil reaches 0.449 on Sri Lanka, below "
                "the persistence baseline of 0.476. Local training reaches 0.708.")
        st.markdown('<p class="note" style="margin-top:1rem">Also true: the complication '
                    'cohort is small (303 paediatric admissions, 63 events); Sri Lankan '
                    'probabilities are not calibrated; headline figures assume complete '
                    'reporting at week close, and a realistic two-week delay costs 0.045 '
                    'PR-AUC; nothing has been tested prospectively.</p>',
                    unsafe_allow_html=True)

        with st.expander("Error breakdown and transfer experiment"):
            cal = R["calib"]
            if cal and "strata" in cal and "baseline_inc_q" in cal["strata"]:
                st.markdown("Recall by starting incidence")
                st.dataframe(pd.DataFrame(cal["strata"]["baseline_inc_q"]).round(3),
                             hide_index=True, width="stretch")
            tf = R["transfer"]
            if tf is not None:
                t = tf[tf.threshold_name == "srilanka_calibrated"] if "threshold_name" in tf else tf

                def _v(cond):
                    s = t[t.condition == cond]
                    return float(s.pr_auc.iloc[0]) if len(s) else np.nan

                bars = pd.DataFrame({
                    "condition": ["Brazil (home)", "Brazil → Sri Lanka", "Persistence",
                                  "Sri Lanka only", "Fine-tuned"],
                    "pr_auc": [0.910, _v("2_brazil_zeroshot"), _v("1_persistence"),
                               _v("3_srilanka_only"), _v("4_brazil_finetuned")]})
                f = px.bar(bars, x="condition", y="pr_auc", text="pr_auc", height=300,
                           labels={"pr_auc": "PR-AUC", "condition": ""})
                f.update_traces(texttemplate="%{text:.3f}", marker_color=ACCENT,
                                textfont={"family": "IBM Plex Mono"})
                f.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(f, width="stretch")

    with t4:
        finding("Three of four public dengue screening datasets are synthetic",
                "In one, body temperature is completely disjoint between classes — "
                "dengue 38.1–40.6 °C, non-dengue 36.0–37.6 °C, "
                "zero overlap. In another the label is reproduced at 98.97% accuracy by the "
                "single rule platelet &lt; 150,000. Models trained on them score 0.99+ "
                "internally and collapse to AUC 0.53–0.61 on real patients.")
        st.markdown('<p class="note">The rule now applied before modelling: any single '
                    'feature reaching AUC ≥ 0.95, or showing disjoint class ranges, '
                    'blocks the dataset. Serology is ground truth for the label, never a '
                    'predictor — feeding the confirmatory test in as an input is '
                    'circular.</p>', unsafe_allow_html=True)
        au = R["audit"]
        if au is not None:
            cols = [c for c in ["dataset", "n", "prevalence", "max_single_feature_auc",
                                "verdict"] if c in au.columns]
            st.dataframe(au[cols], hide_index=True, width="stretch")

        st.markdown("### Sources")
        st.dataframe(pd.DataFrame([
            {"Layer": "Screening", "Source": "Mendeley 6fsrsk3mb8",
             "Detail": "1,511 febrile patients, 19 CBC variables"},
            {"Layer": "Complications", "Source": "Zenodo 6476112",
             "Detail": "303 paediatric dengue admissions"},
            {"Layer": "Outbreak, Brazil", "Source": "Zenodo 22029053",
             "Detail": "4.7M municipality-weeks, 2010–2025"},
            {"Layer": "Outbreak, Sri Lanka", "Source": "denguedatahub + Epidemiology Unit",
             "Detail": "26 districts × weekly, 2006–present, NASA POWER weather"},
        ]), hide_index=True, width="stretch")

        with st.expander("Method notes"):
            st.markdown(
                "- Splits are strictly temporal: train ≤2021, validate 2022–23, "
                "test 2024–25. Thresholds tuned on validation only.\n"
                "- Screening uses nested cross-validation with tuning inside each outer "
                "fold and isotonic calibration.\n"
                "- Lags are date-exact. The published Brazil panel carried 8,107 duplicate "
                "municipality-week rows whose positional lags were misaligned; features are "
                "rebuilt by joining on the calendar date. Impact: −0.0006 PR-AUC.\n"
                "- Sri Lankan surveillance is refreshed weekly from the Epidemiology Unit's "
                "reports, validated at 100% agreement against denguedatahub before use.\n"
                "- Frozen release frozen/v2_final: 72 artifacts, SHA-256 verified.")
