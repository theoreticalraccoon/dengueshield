"""DengueShield - dengue screening + autonomous outbreak early warning.

Screens
  1. Dashboard        - the three tasks, and how a forecast would be used
  2. Guided demo      - one realistic end-to-end scenario in three steps
  3. Patient assessment - quick or full CBC entry -> screening + complication risk
  4. Outbreak forecast  - autonomous district forecasts: continuation AND emergence
  5. Model evidence     - validation, calibration, explainability, generalization

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
MODELS, REPORTS, PROC = ROOT / "models", ROOT / "reports", ROOT / "data" / "processed"

st.set_page_config(page_title="DengueShield", page_icon="🦟", layout="wide")

BAND_COLOR = {"Low": "#2E9E5B", "Moderate": "#E8B93B", "High": "#E8743B", "Very High": "#C7392F"}
DISCLAIMER = (
    "**Not a medical diagnosis.** Clinical and laboratory evaluation is required. "
    "This system does not provide medical clearance and must not be used to rule out dengue."
)

# CBC fields the user can type in "quick" mode; the rest auto-fill from cohort medians
QUICK_FIELDS = ["Age", "Gender", "Total Platelet Count(/cumm)", "Total WBC count(/cumm)",
                "Hemoglobin(g/dl)", "HCT(%)"]


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
        "risk": _csv("srilanka_current_risk.csv"),
        "history": pd.read_parquet(hist) if hist.exists() else None,
        "medians": _json("cbc_population_medians.json") or {},
        "shap": _csv("model2_shap_drivers.csv"),
        "transfer": _csv("srilanka_transfer.csv"),
        "ablation_peds": _csv("ablation_peds.csv"),
        "audit": _csv("dataset_audit.csv"),
        "m2_head": _csv("model2_headtohead.csv"),
        "m1_ops": _json("model1_operating_points.json"),
        "robust": _json("model2_robustness.json"),
        "spatial": _csv("model2_spatial_holdout.csv"),
        "states": _csv("model2_state_folds_with_events.csv"),
        "info_abl": _csv("model2_information_ablation.csv"),
        "calib": _json("model2_calibration_errors.json"),
        "rel_raw": _csv("model2_reliability_raw.csv"),
        "rel_cal": _csv("model2_reliability_calibrated.csv"),
        "sl_calib": _json("srilanka_calibration.json"),
        "worst_fn": _csv("model2_worst_false_negatives.csv"),
        "leakage": _json("leakage_audit.json"),
        "delay": _csv("reporting_delay_stress.csv"),
        "emergence": pd.read_csv(em) if em.exists() else None,
    }


M, R = load_models(), load_reports()


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
    row = build_cbc_row(vals)
    p = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
    return p, b.get("threshold_sens90", 0.5)


def gauge(p, thr, title, flag_colour="#C7392F", ok_colour="#2E9E5B"):
    flag = p >= thr
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        title={"text": title, "font": {"size": 13}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": flag_colour if flag else ok_colour},
               "steps": [{"range": [0, thr * 100], "color": "#EAF3EC"},
                         {"range": [thr * 100, 100], "color": "#FBEAE8"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
    fig.update_layout(height=250, margin={"t": 50, "b": 10})
    return fig


st.sidebar.title("🦟 DengueShield")
screen = st.sidebar.radio(
    "Screen", ["Dashboard", "Guided demo", "Patient assessment",
               "Outbreak forecast", "Model evidence"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Decision support for prioritisation — not an autonomous decision-maker, and not "
    "a diagnostic device. RT-PCR/NAAT and antigen/serological testing remain the "
    "clinical standard."
)


# ================================================================= 1. DASHBOARD
if screen == "Dashboard":
    st.title("DengueShield")
    st.markdown(
        "**Research question.** Can machine learning support both individual risk "
        "assessment and localized outbreak forecasting — and does high performance "
        "reflect genuine generalization or merely epidemiological persistence?"
    )
    st.info(
        "The system performs **four distinct prediction tasks**. They have different "
        "populations, targets and difficulty, and should never be collapsed into one "
        "'dengue accuracy' number."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### A · Dengue screening")
        st.caption("Does this patient appear likely to have dengue?")
        st.metric("ROC-AUC", "0.681")
        st.metric("Accuracy", "76.2%", delta="+7.7pp vs baseline")
        st.warning("**Modest.** A CBC alone is not a reliable dengue diagnostic.")
    with c2:
        st.markdown("### B · Complication risk")
        st.caption("Who needs closer monitoring?")
        st.metric("ROC-AUC", "0.874")
        st.metric("Sens / NPV", "90.5% / 96.2%")
        st.success("**Strong.** Safe to rule out, not to rule in.")
    with c3:
        st.markdown("### C · Outbreak continuation")
        st.caption("Will an outbreak persist for 14 days?")
        st.metric("PR-AUC", "0.961")
        st.metric("Accuracy / Recall", "96.7% / 89.4%")
        st.success("**Strong.** Generalises to unseen municipalities (0.955).")
    with c4:
        st.markdown("### D · Outbreak emergence")
        st.caption("Is a NEW outbreak about to begin?")
        st.metric("PR-AUC", "0.583")
        st.metric("vs best baseline", "+0.201", delta="14.3× lift")
        st.warning("**Harder — and the honest early-warning task.**")

    st.markdown("---")
    st.subheader("How a forecast would be used")
    st.code("""
        HIGH OUTBREAK RISK FLAGGED (14-day forecast)
                        |
                        v
        Public-health authority reviews the flag
                        |
        +---------------+---------------+
        v               v               v
  Mosquito-control   Community      Testing capacity
  inspection         advisory       prioritised
        |               |               |
        +---------------+---------------+
                        v
             Hospital preparedness raised
""", language=None)
    st.warning(
        "**Framing.** Decision support for *prioritisation*. The prototype does not and "
        "should not trigger public-health action automatically — a flag is an input to "
        "human judgement, not a substitute for it."
    )

    st.markdown("---")
    st.subheader("What we found that we did not expect")
    a, b = st.columns(2)
    a.error(
        "**Three of four public dengue screening datasets are synthetic.** Models trained "
        "on them score 0.99+ internally and collapse to AUC 0.53–0.61 on real patients."
    )
    a.warning(
        "**Environment is near-useless for continuation but matters for emergence.** "
        "+0.006 PR-AUC for outbreak continuation versus **+0.089** for emergence — a "
        "15× difference. This is the project's central scientific finding."
    )
    b.warning(
        "**The continuation model tracks trajectory, not emergence.** Recall 0.92 where "
        "incidence is already high; near zero where it is low. Hence a second model."
    )
    b.error(
        "**Zero-shot geographic transfer fails.** Brazil-trained scores 0.449 on Sri "
        "Lanka — worse than persistence (0.476). Local calibration is necessary."
    )


# =============================================================== 2. GUIDED DEMO
elif screen == "Guided demo":
    st.title("Guided demo — one patient, one district, one decision")
    st.caption("A realistic end-to-end scenario in three steps.")

    st.markdown("### Step 1 · A patient arrives at a clinic")
    c1, c2 = st.columns([2, 3])
    with c1:
        age = st.number_input("Age", 1, 100, 17, key="d_age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="d_sex")
        fever = st.number_input("Days of fever", 0, 21, 4, key="d_fev")
        plt_c = st.number_input("Platelet count (/cumm)", 5_000, 600_000, 95_000, 1_000, key="d_plt")
        wbc = st.number_input("Total WBC (/cumm)", 500, 30_000, 3_100, 100, key="d_wbc")
        hb = st.number_input("Haemoglobin (g/dl)", 5.0, 22.0, 13.7, 0.1, key="d_hb")
        hct = st.number_input("Haematocrit (%)", 20.0, 60.0, 44.0, 0.1, key="d_hct")
    with c2:
        if M["screening"] is None:
            st.error("Screening model missing.")
        else:
            p, thr = screen_patient({
                "Age": age, "Gender": 1 if sex == "Male" else 0,
                "Total Platelet Count(/cumm)": plt_c, "Total WBC count(/cumm)": wbc,
                "Hemoglobin(g/dl)": hb, "HCT(%)": hct})
            st.plotly_chart(gauge(p, thr, "Model-estimated screening probability"),
                            width="stretch")
            st.markdown(
                f"**Screening probability {p:.0%}** — "
                f"{'🔴 prioritise dengue testing' if p >= thr else '🟢 lower testing priority'}"
            )
            b = M["peds"]
            if b is not None:
                row = dict.fromkeys(b["features"], 1.0)
                row.update({"HB": hb, "PCV": hct, "LOWEST WBC": wbc, "HIGHEST WBC": wbc * 1.4,
                            "PLATELET COUNT": plt_c, "AGE": 1 if age < 18 else 2,
                            "GENDER": 1 if sex == "Male" else 0,
                            "DURATION OF FEVER": fever, "ALT": 26.0, "CREATININE": 0.5,
                            "FERRITIN": 763.0, "BP AT ADMISSION": 0, "ORGANOMEGALY": 0,
                            "HEADACHE": 1, "MYALGIA": 1, "ABDOMINAL PAIN": 0,
                            "RASH": 1, "VOMITING": 0, "BREATHLESSNESS": 0, "BLEEDING": 0})
                pc = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
                st.metric("Estimated risk of complicated dengue", f"{pc:.0%}")
    st.error(DISCLAIMER)

    st.markdown("---")
    st.markdown("### Step 2 · Where is this patient?")
    dual = R["dual"]
    if dual is None:
        st.warning("Run `experiments/emergence_v1/finalize_emergence_srilanka.py` first.")
    else:
        d = st.selectbox("District", dual.district.tolist(),
                         index=int(np.argmax(dual.district == "Colombo"))
                         if (dual.district == "Colombo").any() else 0)
        row = dual[dual.district == d].iloc[0]
        k = st.columns(4)
        k[0].metric("Current cases (week)", int(row.casos))
        k[1].metric("Incidence /100k", f"{row.p_inc100k:.1f}")
        k[2].metric("Currently in outbreak", "Yes" if row.currently_in_outbreak else "No")
        k[3].metric("14-day continuation risk", f"{row.continuation_risk:.0%}")
        if pd.notna(row.emergence_risk):
            st.info(f"**New-outbreak emergence risk (1–4 weeks): {row.emergence_risk:.0%}** — "
                    f"this district is not currently in outbreak, so the emergence model "
                    f"is the relevant signal.")
        else:
            st.info("This district is **already in outbreak**, so emergence is not asked — "
                    "the continuation risk is the relevant signal.")

        st.markdown("---")
        st.markdown("### Step 3 · Decision support")
        high_ind = 'p' in dir() and p >= thr
        high_geo = (row.continuation_risk >= 0.5) or (
            pd.notna(row.emergence_risk) and row.emergence_risk >= 0.5)
        if high_ind and high_geo:
            st.error(
                f"**Elevated individual screening probability in a high-risk district.**\n\n"
                f"Suggested prioritisation: confirmatory dengue testing for this patient; "
                f"vector-control inspection and community advisory for {d}; review testing "
                f"and bed capacity."
            )
        elif high_ind:
            st.warning(
                f"**Elevated individual probability, lower geographic risk.**\n\n"
                f"Suggested prioritisation: confirmatory testing for this patient. No "
                f"district-level escalation indicated for {d} at this time."
            )
        elif high_geo:
            st.warning(
                f"**Lower individual probability, but {d} is at elevated risk.**\n\n"
                f"Suggested prioritisation: routine care for this patient; maintain "
                f"district-level surveillance and vector control in {d}."
            )
        else:
            st.success(
                f"**Neither signal elevated.** Routine care for this patient; routine "
                f"surveillance in {d}."
            )
        st.caption(
            "These are prioritisation suggestions derived from two independent models. "
            "They are not clinical instructions and not public-health directives."
        )


# ========================================================= 3. PATIENT ASSESSMENT
elif screen == "Patient assessment":
    st.title("Patient assessment")
    st.caption(DISCLAIMER)
    tab1, tab2 = st.tabs(["Dengue screening (CBC)", "Complication risk"])

    with tab1:
        b = M["screening"]
        if b is None:
            st.error("Screening model not found — run `finalize_model1.py` first.")
        else:
            mode = st.radio("Entry mode", ["Quick (6 values)", "Full blood count"],
                            horizontal=True)
            med = R["medians"]
            vals = {}
            if mode.startswith("Quick"):
                st.caption("Unsupplied values are filled from the study cohort median. "
                           "Switch to full entry for a complete blood count.")
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

            if st.button("Assess", type="primary", key="screen_btn"):
                p, thr = screen_patient(vals)
                m1, m2 = st.columns([1, 2])
                m1.markdown("**Model-estimated screening probability**")
                m1.metric("prob", f"{p:.0%}", label_visibility="collapsed")
                m1.markdown(f"### {'🔴 Prioritise dengue testing' if p >= thr else '🟢 Lower testing priority'}")
                m2.plotly_chart(gauge(p, thr, "Model-estimated screening probability"),
                                width="stretch")
                st.error(DISCLAIMER)
                ops = R["m1_ops"]
                if ops:
                    st.info(
                        f"**How much to trust this.** On real patients (nested "
                        f"cross-validation): ROC-AUC **{ops['at_0.5']['roc_auc']:.3f}**, "
                        f"accuracy **{ops['at_0.5']['accuracy']:.3f}** against a "
                        f"majority-class baseline of 0.685. Use this to **prioritise "
                        f"testing**, never to rule dengue out."
                    )

    with tab2:
        st.caption("For a patient already known or strongly suspected to have dengue: "
                   "estimated risk of complicated dengue, relative to the model's "
                   "training population (303 paediatric admissions).")
        b = M["peds"]
        if b is None:
            st.error("Complication model not found — run `finalize_peds.py` first.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Demographics & course**")
                p_age = st.selectbox("Age band", [1, 2], format_func=lambda v: f"Band {v}")
                p_sex = st.selectbox("Sex", ["Male", "Female"], key="psex")
                fever = st.number_input("Duration of fever (days)", 0, 21, 4)
                bp = st.selectbox("BP at admission", ["Normal", "Low"])
                organ = st.selectbox("Organomegaly", ["No", "Yes"])
            with c2:
                st.markdown("**Symptoms**")
                sx = {s: st.checkbox(s.title()) for s in
                      ["headache", "myalgia", "abdominal pain", "rash",
                       "vomiting", "breathlessness", "bleeding"]}
            with c3:
                st.markdown("**Laboratory**")
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
                a, c = st.columns([1, 2])
                a.markdown("**Estimated risk of complicated dengue**")
                a.metric("risk", f"{p:.0%}", label_visibility="collapsed")
                a.markdown(f"### {'🔴 Monitor closely' if p >= thr else '🟢 Routine care'}")
                mm = b["metrics"]
                c.info(f"Operating at **{mm['sensitivity']:.0%} sensitivity** / "
                       f"**{mm['specificity']:.0%} specificity** "
                       f"(ROC-AUC {mm['roc_auc']:.3f}, NPV {mm['npv']:.3f}). Tuned for "
                       "sensitivity: missing a deteriorating patient is far costlier "
                       "than one extra observation.")
                st.error(DISCLAIMER)


# ========================================================== 4. OUTBREAK FORECAST
elif screen == "Outbreak forecast":
    st.title("Sri Lanka outbreak forecast")
    dual = R["dual"]
    if dual is None:
        st.error("No dual forecasts — run "
                 "`experiments/emergence_v1/finalize_emergence_srilanka.py`.")
    else:
        wk = pd.to_datetime(dual.week_start).max().date()
        st.caption(f"Generated automatically from surveillance week beginning **{wk}**. "
                   f"Two independent models run for every district.")

        a, b = st.columns(2)
        a.info("**Continuation risk** — will an outbreak persist over the next 14 days? "
               "Asked of every district. (Brazil-validated architecture, PR-AUC 0.961; "
               "Sri Lanka-trained, 0.708.)")
        b.info("**Emergence risk** — is a NEW outbreak about to begin in the next 1–4 "
               "weeks? Asked only of districts *not* currently in outbreak — exactly "
               "where the continuation model is blind. (PR-AUC 0.357 vs 0.208 baseline.)")

        sc = R["sl_calib"]
        if sc:
            st.warning(
                f"**Predicted probabilities, not calibrated ones.** Overall ECE "
                f"{sc['ece']:.3f}, but in the highest band the continuation model predicts "
                f"~0.81 where outbreaks occur ~0.54 of the time. Treat the **ranking** as "
                f"more reliable than the number."
            )

        d = dual.copy()
        d["status"] = np.where(d.currently_in_outbreak, "In outbreak", "Not in outbreak")
        d["headline_risk"] = np.where(d.currently_in_outbreak, d.continuation_risk,
                                      d.emergence_risk.fillna(d.continuation_risk))
        d["band"] = pd.cut(d.headline_risk, [-0.01, 0.25, 0.5, 0.75, 1.01],
                           labels=["Low", "Moderate", "High", "Very High"])
        k = st.columns(5)
        k[0].metric("Districts in outbreak", int(d.currently_in_outbreak.sum()))
        for i, band in enumerate(["Very High", "High", "Moderate", "Low"]):
            k[i + 1].metric(band, int((d.band == band).sum()))

        left, right = st.columns([3, 2])
        with left:
            fig = px.scatter_map(
                d, lat="lat", lon="lon", size="casos", color="band",
                color_discrete_map=BAND_COLOR, hover_name="district",
                hover_data={"casos": True, "p_inc100k": ":.1f", "status": True,
                            "continuation_risk": ":.2f", "emergence_risk": ":.2f",
                            "lat": False, "lon": False, "band": False},
                size_max=45, zoom=6.2, height=560,
                category_orders={"band": ["Very High", "High", "Moderate", "Low"]})
            fig.update_layout(map_style="carto-positron", margin={"l": 0, "r": 0, "t": 0, "b": 0})
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("#### District risk table")
            show = d[["district", "status", "casos", "p_inc100k",
                      "continuation_risk", "emergence_risk"]].copy()
            show.columns = ["District", "Status", "Cases", "Per 100k",
                            "Continuation", "Emergence"]
            st.dataframe(
                show.style.format({"Per 100k": "{:.1f}", "Continuation": "{:.2f}",
                                   "Emergence": "{:.2f}"}, na_rep="—"),
                hide_index=True, width="stretch", height=520)
            st.caption("Emergence shows “—” where a district is already in outbreak: "
                       "the question does not apply.")

        st.markdown("---")
        sel = st.selectbox("Inspect a district", d.district.tolist())
        row = d[d.district == sel].iloc[0]
        c = st.columns(5)
        c[0].metric("Current cases", int(row.casos))
        c[1].metric("Incidence /100k", f"{row.p_inc100k:.1f}")
        c[2].metric("Status", row.status)
        c[3].metric("Continuation risk", f"{row.continuation_risk:.0%}")
        c[4].metric("Emergence risk",
                    "n/a" if pd.isna(row.emergence_risk) else f"{row.emergence_risk:.0%}")

        hist = R["history"]
        if hist is not None:
            h = hist[hist.district == sel].sort_values("week_start").tail(160)
            f = go.Figure()
            f.add_trace(go.Scatter(x=h.week_start, y=h.casos, name="Cases", line={"width": 2}))
            f.add_trace(go.Bar(x=h.week_start, y=h.precip_total_semana, name="Rainfall (mm)",
                               yaxis="y2", opacity=0.3))
            f.update_layout(height=340, yaxis={"title": "Weekly cases"},
                            yaxis2={"title": "Rainfall (mm)", "overlaying": "y", "side": "right"},
                            legend={"orientation": "h", "y": 1.12},
                            margin={"t": 30, "b": 10}, title=f"{sel} — last 3 years")
            st.plotly_chart(f, width="stretch")


# ============================================================== 5. MODEL EVIDENCE
else:
    st.title("Model evidence")
    st.caption("Including the weaknesses. Everything here is out-of-sample.")
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Validation", "Emergence", "Calibration", "Explainability",
         "Generalization", "Data integrity"])

    with t1:
        st.subheader("Temporal holdout — Brazil, locked test set 2024–25")
        hh = R["m2_head"]
        if hh is not None:
            st.dataframe(hh[["model", "roc_auc", "pr_auc", "accuracy", "recall",
                             "precision", "brier"]].round(4), hide_index=True, width="stretch")
            st.caption("Trivial 'never an outbreak' accuracy is 0.865 — accuracy alone is "
                       "misleading at this prevalence, so PR-AUC and recall lead.")
        rb = R["robust"]
        if rb and "rolling_origin" in rb:
            ro = pd.DataFrame(rb["rolling_origin"])
            f = go.Figure()
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.accuracy, name="Accuracy",
                                   mode="lines+markers"))
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.trivial_acc, name="Trivial baseline",
                                   mode="lines+markers", line={"dash": "dash"}))
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.pr_auc, name="PR-AUC",
                                   mode="lines+markers"))
            f.update_layout(height=330, title="Rolling-origin backtest",
                            yaxis={"range": [0, 1]}, margin={"t": 40})
            st.plotly_chart(f, width="stretch")

        st.subheader("Temporal-leakage audit")
        lk = R["leakage"]
        if lk:
            we = lk.get("worked_example", {})
            st.code(
                f"municipality      : {we.get('municipality')} ({we.get('state')})\n"
                f"feature week t    : {we.get('feature_week_start')}\n"
                f"FEATURE CUTOFF    : {we.get('feature_cutoff')}   (all inputs on/before)\n"
                f"forecast issued   : {we.get('forecast_issued')}  (once week t closes)\n"
                f"TARGET WINDOW     : {we.get('target_week_start')} .. {we.get('target_week_end')}\n"
                f"lead time         : {we.get('lead_time_days')} days", language=None)
            pi = lk.get("panel_integrity", {})
            # the audit reports the panel both as published and after repair
            pub = pi.get("as_published", pi if "rows" in pi else {})
            rep = pi.get("after_repair", {})
            if pub:
                st.dataframe(
                    pd.DataFrame([{"panel": "as published", **{
                        k: v for k, v in pub.items()
                        if k in ("rows", "clean_7day_steps",
                                 "duplicate_municipality_week_rows",
                                 "missing_week_gaps_14day")}},
                                  *([{"panel": "after repair", **{
                                      k: v for k, v in rep.items()
                                      if k in ("rows", "clean_7day_steps",
                                               "duplicate_municipality_week_rows",
                                               "missing_week_gaps_14day")}}] if rep else [])]),
                    hide_index=True, width="stretch")
            cc = lk.get("contamination_checks", {})
            if cc:
                st.success(
                    "**No future contamination.** Every lag column matches the value exactly "
                    f"N calendar weeks earlier (match rate "
                    f"{min(v for k, v in cc.items() if k.startswith('casos_lag')):.3f}–1.000), "
                    "and no lag column matched a future value."
                )
            if rep:
                st.info(
                    f"**Audit finding, fixed.** The published panel carried "
                    f"{pub.get('duplicate_municipality_week_rows', 0):,} duplicate "
                    f"(municipality, week) rows, so its positional lag columns did not land "
                    f"on the intended calendar offset. Features are now rebuilt date-exactly "
                    f"(`recompute_dynamics`). Impact on the headline: PR-AUC 0.9614 → 0.9608, "
                    f"recall 0.8984 → 0.9004 — the misalignment degraded rather than inflated "
                    f"performance, so the frozen v2 conclusions stand."
                )
        dl = R["delay"]
        if dl is not None:
            st.markdown("**Reporting-delay stress test** — archive counts are backfilled; "
                        "real-time data is incomplete.")
            st.dataframe(dl.round(4), hide_index=True, width="stretch")
            st.warning(
                f"A 2-week reporting delay costs "
                f"{dl.pr_auc.iloc[0] - dl.pr_auc.iloc[-1]:+.4f} PR-AUC "
                f"({dl.pr_auc.iloc[0]:.3f} → {dl.pr_auc.iloc[-1]:.3f}). Performance is "
                "robust to realistic delay, but the headline number assumes complete "
                "reporting at week close."
            )

        st.subheader("Error analysis — where does it fail?")
        cal = R["calib"]
        if cal and "strata" in cal:
            for key, title in [("season", "By season"),
                               ("baseline_inc_q", "By baseline incidence"),
                               ("vs_history", "Unprecedented vs within historical range")]:
                if key in cal["strata"]:
                    st.markdown(f"**{title}**")
                    st.dataframe(pd.DataFrame(cal["strata"][key]).round(4),
                                 hide_index=True, width="stretch")
            mom = cal.get("momentum", {})
            if mom:
                ratio = mom["fp_at_epidemic_level"] / max(mom["tn_at_epidemic_level"], 1e-9)
                st.error(
                    f"**Momentum failure mode.** {mom['fp_at_epidemic_level']:.0%} of false "
                    f"positives are municipalities already at epidemic level versus "
                    f"{mom['tn_at_epidemic_level']:.1%} of true negatives — ~{ratio:.0f}× "
                    "more likely. Momentum carries through subsiding outbreaks."
                )
        if R["worst_fn"] is not None:
            with st.expander("10 largest missed outbreaks"):
                st.dataframe(R["worst_fn"].head(10).round(2), hide_index=True, width="stretch")

    with t2:
        st.subheader("Emerging-outbreak detection (exploratory extension)")
        st.markdown(
            "The continuation model is blind where it matters most for early warning. "
            "This experiment asks a different question of a different population: among "
            "districts **not currently in outbreak**, will one *begin*?"
        )
        em = R["emergence"]
        if em is None:
            st.info("Run `experiments/emergence_v1/run_emergence.py`.")
        else:
            cfg = st.selectbox("Configuration", sorted(em.config.unique()))
            e = em[em.config == cfg]
            st.dataframe(e[["condition", "n_features", "pr_auc", "recall", "precision",
                            "prevalence", "lift_over_prevalence"]].round(4),
                         hide_index=True, width="stretch")
            f = px.bar(e, x="condition", y="pr_auc", text="pr_auc", height=400,
                       labels={"pr_auc": "PR-AUC", "condition": ""},
                       color="condition", color_discrete_sequence=px.colors.qualitative.Set2)
            f.update_traces(texttemplate="%{text:.3f}")
            f.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(f, width="stretch")
            st.success(
                "**The project's central scientific finding.** Environmental variables add "
                "only **+0.006** PR-AUC for outbreak *continuation*, but **+0.066 to "
                "+0.089** for *emergence* — a 12–15× difference. Climate and ecology are "
                "weak predictors of an outbreak persisting, and materially more useful for "
                "predicting one beginning."
            )
            st.warning(
                "Emergence is a much harder task: PR-AUC ~0.58 versus ~0.96 for "
                "continuation. It still beats every trivial baseline (persistence 0.382, "
                "growth rate 0.146, moving average 0.125) by a wide margin, at 14.3× lift "
                "over prevalence — but it is an exploratory result, not a deployed claim."
            )

    with t3:
        st.subheader("Reliability — does 80% mean 80%?")
        cal = R["calib"]
        if cal:
            a, b_, c = st.columns(3)
            a.metric("Brier (raw)", f"{cal['raw']['brier']:.4f}")
            b_.metric("ECE (raw)", f"{cal['raw']['ece']:.4f}")
            c.metric("ECE (isotonic)", f"{cal['calibrated']['ece']:.4f}",
                     delta=f"{cal['calibrated']['ece'] - cal['raw']['ece']:+.4f}")
            st.success(
                f"Isotonic calibration (fitted on validation only) cuts expected "
                f"calibration error from {cal['raw']['ece']:.4f} to "
                f"{cal['calibrated']['ece']:.4f}. The Brazil model's outputs may honestly "
                f"be called **calibrated probabilities**."
            )
        rr, rc = R["rel_raw"], R["rel_cal"]
        if rr is not None:
            f = go.Figure()
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="perfect",
                                   line={"dash": "dash", "color": "black"}))
            f.add_trace(go.Scatter(x=rr.predicted, y=rr.observed, name="raw",
                                   mode="lines+markers"))
            if rc is not None:
                f.add_trace(go.Scatter(x=rc.predicted, y=rc.observed, name="isotonic",
                                       mode="lines+markers"))
            f.update_layout(height=440, xaxis_title="Predicted outbreak probability",
                            yaxis_title="Observed outbreak frequency",
                            title="Reliability diagram — 14-day forecast, test 2024–25")
            st.plotly_chart(f, width="stretch")
        sc = R["sl_calib"]
        if sc:
            st.warning(f"**The Sri Lanka model is not equally calibrated** "
                       f"(ECE {sc['ece']:.3f}) and is overconfident in its highest-risk "
                       "band. The forecast page labels its output *predicted*, not "
                       "*calibrated*.")

    with t4:
        st.subheader("Two different questions about a feature")
        a, b = st.columns(2)
        a.markdown("#### Feature contribution to a prediction")
        a.caption("SHAP — how the model *distributes credit* among correlated inputs.")
        sd = R["shap"]
        if sd is not None:
            f = px.bar(sd.sort_values("pct_of_total"), x="pct_of_total", y="driver",
                       orientation="h", text="pct_of_total",
                       labels={"pct_of_total": "% of total |SHAP|", "driver": ""})
            f.update_traces(texttemplate="%{text:.1f}%", marker_color="#3B6EA8")
            f.update_layout(height=420, margin={"t": 10})
            a.plotly_chart(f, width="stretch")
        b.markdown("#### Incremental value from ablation")
        b.caption("What performance actually *costs* if the group is removed.")
        ia = R["info_abl"]
        if ia is not None:
            f = px.bar(ia, x="pr_auc", y="condition", orientation="h", text="pr_auc",
                       labels={"pr_auc": "PR-AUC", "condition": ""})
            f.update_traces(texttemplate="%{text:.3f}", marker_color="#C7392F")
            f.update_layout(height=420, margin={"t": 10})
            b.plotly_chart(f, width="stretch")
        st.error(
            "**These are not the same thing.** SHAP attributes ~50% of the continuation "
            "model's reasoning to environment, but removing environment costs only "
            "**+0.006 PR-AUC**, and environment alone (0.597) is *worse than persistence* "
            "(0.760). Attribution ≠ incremental predictive value when features correlate."
        )

    with t5:
        st.subheader("Spatial holdout — can it predict unseen places?")
        sp = R["spatial"]
        if sp is not None:
            main = sp[~sp.condition.str.startswith("state_fold")]
            st.dataframe(main[["condition", "pr_auc", "accuracy", "recall",
                               "precision", "n_test"]].round(4),
                         hide_index=True, width="stretch")
            st.success("Municipalities the model has **never seen** score PR-AUC 0.955 "
                       "versus 0.962 for seen ones — it generalises rather than memorising "
                       "place identity.")
        stf = R["states"]
        if stf is not None:
            st.markdown("**Leave-whole-states-out**, with positive-event counts")
            st.dataframe(stf[["condition", "n_positive", "test_prevalence", "pr_auc",
                              "lift_over_prevalence", "recall"]].round(3),
                         hide_index=True, width="stretch")
            st.info(
                "PR-AUC's own baseline *is* prevalence, so raw PR-AUC is not comparable "
                "across folds. In **lift** terms the lowest-PR-AUC fold is the strongest "
                "(25.7× vs 4.2×). The apparent regional spread is largely a prevalence "
                "artefact — though recall does genuinely degrade in low-prevalence regions "
                "(0.67 vs 0.91), consistent with the emergence limitation."
            )

        st.subheader("Brazil → Sri Lanka transfer")
        tf = R["transfer"]
        if tf is not None:
            t = tf[tf.threshold_name == "srilanka_calibrated"] if "threshold_name" in tf else tf

            def _v(cond):
                s = t[t.condition == cond]
                return float(s.pr_auc.iloc[0]) if len(s) else np.nan

            bars = pd.DataFrame({
                "condition": ["Brazil validation<br>(source domain)",
                              "Brazil → Sri Lanka<br>(zero-shot)",
                              "Persistence<br>baseline", "Sri Lanka only",
                              "Brazil + fine-tuned"],
                "pr_auc": [0.910, _v("2_brazil_zeroshot"), _v("1_persistence"),
                           _v("3_srilanka_only"), _v("4_brazil_finetuned")]})
            f = px.bar(bars, x="condition", y="pr_auc", text="pr_auc", height=420,
                       labels={"pr_auc": "PR-AUC", "condition": ""},
                       color="pr_auc", color_continuous_scale="RdYlGn")
            f.update_traces(texttemplate="%{text:.3f}")
            f.update_layout(coloraxis_showscale=False)
            st.plotly_chart(f, width="stretch")
            st.error(
                "**Zero-shot transfer fails.** The same model scoring 0.910 on Brazilian "
                "validation reaches only 0.449 on Sri Lanka — below the persistence "
                "baseline (0.476). Local epidemiological dynamics and calibration remain "
                "necessary."
            )

    with t6:
        st.subheader("Dataset integrity audit")
        au = R["audit"]
        if au is not None:
            st.dataframe(au, hide_index=True, width="stretch")
            st.error(
                "**Three of four public dengue screening datasets were rejected.** In one, "
                "body temperature is entirely disjoint between classes (dengue 38.1–40.6 °C, "
                "non-dengue 36.0–37.6 °C). Models trained on them score 0.99+ internally "
                "and collapse to AUC 0.53–0.61 on real patients."
            )
        st.markdown(
            "**The audit rule.** Any single feature with AUC ≥ 0.95, or disjoint "
            "class-conditional ranges, blocks the dataset before modelling begins. "
            "Serology (NS1/IgM/IgG) is used only as ground truth for the label, never as "
            "a predictor."
        )
        ab = R["ablation_peds"]
        if ab is not None:
            st.subheader("Feature-group ablation — complication risk")
            st.dataframe(ab[ab.model == "lgbm"][["tier", "n_features", "roc_auc", "pr_auc",
                                                 "sensitivity", "specificity"]].round(4),
                         hide_index=True, width="stretch")
            st.caption(
                "At a fixed ~90% sensitivity, specificity nearly doubles (0.33 → 0.64) as "
                "symptoms and organ labs are added; demographics add almost nothing. "
                "**This ablation was performed on the complication cohort, not the primary "
                "dengue-vs-non-dengue screening cohort** — these effects must not be read "
                "as measured improvements for dengue diagnosis itself."
            )
