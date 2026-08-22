"""DengueShield - dengue screening + outbreak early warning.

Two working tools, and one place for everything behind them:

  1. Patient assessment - CBC screening, complication risk, optional district context
  2. Outbreak forecast  - autonomous district forecasts: continuation AND emergence
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

st.set_page_config(page_title="DengueShield", page_icon="🦟", layout="wide")

BAND_COLOR = {"Low": "#2E9E5B", "Moderate": "#E8B93B", "High": "#E8743B", "Very High": "#C7392F"}
DISCLAIMER = (
    "**Not a medical diagnosis.** Clinical and laboratory evaluation is required. "
    "This does not provide medical clearance and must not be used to rule out dengue."
)


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
    p = float(b["model"].predict_proba(pd.DataFrame([build_cbc_row(vals)])[b["features"]])[0, 1])
    return p, b.get("threshold_sens90", 0.5)


def gauge(p, thr, title):
    flag = p >= thr
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        title={"text": title, "font": {"size": 13}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#C7392F" if flag else "#2E9E5B"},
               "steps": [{"range": [0, thr * 100], "color": "#EAF3EC"},
                         {"range": [thr * 100, 100], "color": "#FBEAE8"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
    fig.update_layout(height=250, margin={"t": 50, "b": 10})
    return fig


st.sidebar.title("🦟 DengueShield")
screen = st.sidebar.radio(
    "Screen", ["Patient assessment", "Outbreak forecast", "About the models"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Decision support for prioritisation — not an autonomous decision-maker, and not "
    "a diagnostic device. RT-PCR/NAAT and antigen/serological testing remain the "
    "clinical standard."
)


# ========================================================= 1. PATIENT ASSESSMENT
if screen == "Patient assessment":
    st.title("Patient assessment")
    st.caption(DISCLAIMER)

    tab1, tab2 = st.tabs(["Dengue screening", "Complication risk"])

    with tab1:
        b = M["screening"]
        if b is None:
            st.error("Screening model not found — run `finalize_model1.py`.")
        else:
            mode = st.radio("Entry mode", ["Quick (6 values)", "Full blood count"],
                            horizontal=True)
            vals = {}
            if mode.startswith("Quick"):
                st.caption("Anything you leave out is filled from the study cohort median.")
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

            dual = R["dual"]
            district = None
            if dual is not None:
                district = st.selectbox(
                    "Patient's district (optional — adds local outbreak context)",
                    ["— not specified —"] + dual.district.tolist())

            if st.button("Assess", type="primary", key="screen_btn"):
                p, thr = screen_patient(vals)
                flag = p >= thr
                m1, m2 = st.columns([1, 2])
                m1.markdown("**Model-estimated screening probability**")
                m1.metric("prob", f"{p:.0%}", label_visibility="collapsed")
                m1.markdown(
                    f"### {'🔴 Prioritise dengue testing' if flag else '🟢 Lower testing priority'}")
                m2.plotly_chart(gauge(p, thr, "Model-estimated screening probability"),
                                width="stretch")

                # local outbreak context, if a district was given
                if dual is not None and district and not district.startswith("—"):
                    row = dual[dual.district == district].iloc[0]
                    geo = (row.continuation_risk if row.currently_in_outbreak
                           else (row.emergence_risk if pd.notna(row.emergence_risk)
                                 else row.continuation_risk))
                    st.markdown("---")
                    k = st.columns(4)
                    k[0].metric("District", district)
                    k[1].metric("Cases this week", int(row.casos))
                    k[2].metric("Currently in outbreak",
                                "Yes" if row.currently_in_outbreak else "No")
                    k[3].metric("14-day outbreak risk", f"{geo:.0%}")
                    if flag and geo >= 0.5:
                        st.error(
                            f"**Elevated patient probability in a high-risk district.** "
                            f"Suggested prioritisation: confirmatory testing for this "
                            f"patient; vector-control inspection and community advisory "
                            f"for {district}; review testing and bed capacity.")
                    elif flag:
                        st.warning(
                            f"**Elevated patient probability, lower geographic risk.** "
                            f"Confirmatory testing for this patient. No district-level "
                            f"escalation indicated for {district}.")
                    elif geo >= 0.5:
                        st.warning(
                            f"**Lower patient probability, but {district} is at elevated "
                            f"risk.** Routine care here; maintain district surveillance "
                            f"and vector control.")
                    else:
                        st.success(
                            f"**Neither signal elevated.** Routine care, routine "
                            f"surveillance in {district}.")
                    st.caption("Prioritisation suggestions from two independent models — "
                               "not clinical instructions or public-health directives.")

                st.error(DISCLAIMER)
                ops = R["m1_ops"]
                if ops:
                    st.caption(
                        f"Reliability: ROC-AUC {ops['at_0.5']['roc_auc']:.3f}, accuracy "
                        f"{ops['at_0.5']['accuracy']:.3f} against a 0.685 majority-class "
                        f"baseline, on real patients under nested cross-validation. Use "
                        f"this to prioritise testing, never to rule dengue out.")

    with tab2:
        st.caption("For a patient already known or suspected to have dengue: estimated "
                   "risk of complications, relative to the model's training population "
                   "(303 paediatric admissions).")
        b = M["peds"]
        if b is None:
            st.error("Complication model not found — run `finalize_peds.py`.")
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
                       f"**{mm['specificity']:.0%} specificity** (NPV {mm['npv']:.2f}). "
                       "Deliberately tuned for sensitivity: missing a deteriorating "
                       "patient costs far more than one extra observation.")
                st.error(DISCLAIMER)


# ========================================================== 2. OUTBREAK FORECAST
elif screen == "Outbreak forecast":
    st.title("Sri Lanka outbreak forecast")
    dual = R["dual"]
    if dual is None:
        st.error("No forecasts — run "
                 "`experiments/emergence_v1/finalize_emergence_srilanka.py`.")
    else:
        wk = pd.to_datetime(dual.week_start).max().date()
        st.caption("Two models run per district: whether an existing outbreak will "
                   "**continue**, and whether a new one will **emerge**.")

        # Surveillance is published with a lag, so say plainly how old this is.
        fr = R["freshness"]
        age = fr.get("age_days") if fr else None
        if age is None:
            age = (pd.Timestamp.utcnow().tz_localize(None).normalize()
                   - pd.Timestamp(wk)).days
        stamp = (f"**Surveillance week beginning {wk}** — {age} days old"
                 + (f", refreshed {str(fr['refreshed_at'])[:10]}" if fr and fr.get("refreshed_at") else ""))
        if age <= 21:
            st.success(f"{stamp}. This is the most recent week Sri Lanka has published.")
        elif age <= 70:
            st.info(f"{stamp}. Sri Lanka's Epidemiology Unit publishes with a lag of "
                    "roughly six to eight weeks, so this is at or near the newest "
                    "data that exists.")
        else:
            st.warning(f"{stamp}. Older than the usual publication lag — run "
                       "`python refresh_data.py` to pull anything newer.")

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
            show = d[["district", "status", "casos", "continuation_risk",
                      "emergence_risk"]].copy()
            show.columns = ["District", "Status", "Cases", "Continuation", "Emergence"]
            st.dataframe(
                show.style.format({"Continuation": "{:.2f}", "Emergence": "{:.2f}"},
                                  na_rep="—"),
                hide_index=True, width="stretch", height=520)
            st.caption("“—” means the district is already in outbreak, so the emergence "
                       "question does not apply.")

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
            f.update_layout(height=330, yaxis={"title": "Weekly cases"},
                            yaxis2={"title": "Rainfall (mm)", "overlaying": "y", "side": "right"},
                            legend={"orientation": "h", "y": 1.12},
                            margin={"t": 30, "b": 10}, title=f"{sel} — last 3 years")
            st.plotly_chart(f, width="stretch")

        sc = R["sl_calib"]
        if sc:
            st.caption(
                f"**Read the ranking, not the number.** These are predicted probabilities, "
                f"not calibrated ones (ECE {sc['ece']:.3f}); in the highest band the model "
                f"says ~0.81 where outbreaks occur ~0.54 of the time. It is also far better "
                f"at spotting escalation than emergence — see *About the models*.")


# ========================================================== 3. ABOUT THE MODELS
else:
    st.title("About the models")
    t1, t2, t3, t4 = st.tabs(["What it does", "How well it works",
                              "Where it fails", "Data & methods"])

    # ---------------- what it does ----------------
    with t1:
        st.markdown(
            "DengueShield answers **four separate questions**. They involve different "
            "populations and differ enormously in difficulty, so they are never combined "
            "into one “dengue accuracy” figure."
        )
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

        st.markdown("#### How a forecast would be used")
        st.code("""HIGH RISK FLAGGED (14-day forecast)
        |
        v
Public-health authority reviews the flag
        |
   +----+----+----------------+
   v         v                v
Vector    Community      Testing capacity
control   advisory       prioritised
   +----+----+----------------+
        v
Hospital preparedness raised""", language=None)
        st.info(
            "**This is decision support for prioritisation.** A flag is an input to human "
            "judgement — the prototype does not and should not trigger public-health "
            "action on its own."
        )

    # ---------------- how well ----------------
    with t2:
        st.markdown(
            "The 2024–25 test set was **locked before** any later development and never "
            "tuned against. Every figure below is out-of-sample."
        )
        c = st.columns(4)
        c[0].metric("14-day forecast", "PR-AUC 0.961")
        c[1].metric("vs. persistence", "0.615", delta="+0.35")
        c[2].metric("Unseen districts", "0.955", delta="−0.007")
        c[3].metric("Calibration (ECE)", "0.0052", delta="−0.014", delta_color="inverse")

        rb = R["robust"]
        if rb and "rolling_origin" in rb:
            ro = pd.DataFrame(rb["rolling_origin"])
            f = go.Figure()
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.accuracy, name="Model accuracy",
                                   mode="lines+markers", line={"width": 3}))
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.trivial_acc,
                                   name="Guessing “no outbreak”", mode="lines+markers",
                                   line={"dash": "dash"}))
            f.update_layout(height=300, yaxis={"range": [0.7, 1.0], "title": "Accuracy"},
                            margin={"t": 36}, title="Retrained each year, tested on the next",
                            legend={"orientation": "h", "y": 1.15})
            st.plotly_chart(f, width="stretch")
            st.caption("Every one of seven backtest years clears 90% accuracy while beating "
                       "the trivial baseline.")

        st.markdown(
            "**The result that mattered most.** SHAP attributed about half the model's "
            "reasoning to climate and land cover — but removing all of it costs just "
            "**+0.006 PR-AUC**, and environment *alone* scores worse than assuming next "
            "month looks like this one. The skill is epidemiological momentum, not "
            "environmental prediction. Ask the same features about outbreak *emergence*, "
            "though, and they add **+0.089** — a 15× difference."
        )
        ia = R["info_abl"]
        if ia is not None:
            f = px.bar(ia.sort_values("pr_auc"), x="pr_auc", y="condition",
                       orientation="h", text="pr_auc", height=230,
                       labels={"pr_auc": "PR-AUC", "condition": ""})
            f.update_traces(texttemplate="%{text:.3f}", marker_color="#3B6EA8")
            f.update_layout(margin={"t": 10, "b": 10})
            st.plotly_chart(f, width="stretch")

        with st.expander("Full validation battery and model comparison"):
            rows = [("Temporal holdout (locked 2024–25)", "PR-AUC 0.961 · 96.7% acc"),
                    ("Rolling-origin backtest, 7 years", "mean 0.818 · every year >90%"),
                    ("Spatial holdout — unseen municipalities", "0.955 vs 0.962 seen"),
                    ("Leave-whole-states-out", "mean 0.872 (spread is prevalence)"),
                    ("Horizon sweep (2 / 4 / 8 weeks)", "0.961 / 0.919 / 0.813"),
                    ("Shuffled-label control", "collapses to 0.124 (chance 0.144)"),
                    ("Temporal-leakage audit", "no lag matched a future value"),
                    ("Reporting-delay stress (2 weeks)", "0.961 → 0.916")]
            st.dataframe(pd.DataFrame(rows, columns=["Test", "Result"]),
                         hide_index=True, width="stretch")
            st.caption("The shuffled-label control is the decisive one: a leak would still "
                       "score well there. It does not.")
            if R["m2_head"] is not None:
                st.dataframe(R["m2_head"][["model", "pr_auc", "accuracy", "recall",
                                           "precision", "brier"]].round(4),
                             hide_index=True, width="stretch")
                st.caption("The LSTM did not beat gradient boosting, and is far worse "
                           "calibrated. LightGBM is deployed.")

    # ---------------- where it fails ----------------
    with t3:
        st.markdown(
            "A forecasting system that hides its blind spots is worse than one that "
            "has none."
        )
        a, b = st.columns(2)
        a.error(
            "**It tracks trajectory, not emergence.** Recall is 0.92 where transmission "
            "is already elevated and near **zero** where it is low — arguably the exact "
            "situation where early warning would matter most."
        )
        a.warning(
            "**Momentum carries through subsiding outbreaks.** 54.5% of false alarms are "
            "districts already at epidemic level, versus 1.5% of true negatives — **37× "
            "more likely**."
        )
        b.warning(
            "**Screening from a blood count is weak.** ROC-AUC 0.681. It prioritises "
            "testing; it cannot rule dengue out."
        )
        b.error(
            "**It does not transfer between countries.** The same model scoring 0.910 on "
            "Brazil reaches **0.449** on Sri Lanka — below the persistence baseline (0.476). "
            "Local training reaches 0.708."
        )

        st.markdown("**Also true:** the complication cohort is small (303 paediatric "
                    "admissions, 63 events); Sri Lankan probabilities are not calibrated; "
                    "headline figures assume complete reporting at week close, and a "
                    "realistic two-week delay costs 0.045 PR-AUC; nothing has been tested "
                    "prospectively against a live surveillance feed.")

        with st.expander("Error breakdown and the transfer experiment"):
            cal = R["calib"]
            if cal and "strata" in cal and "baseline_inc_q" in cal["strata"]:
                st.markdown("**Recall by starting incidence** — the blind spot, quantified")
                st.dataframe(pd.DataFrame(cal["strata"]["baseline_inc_q"]).round(3),
                             hide_index=True, width="stretch")
            tf = R["transfer"]
            if tf is not None:
                t = tf[tf.threshold_name == "srilanka_calibrated"] if "threshold_name" in tf else tf

                def _v(cond):
                    s = t[t.condition == cond]
                    return float(s.pr_auc.iloc[0]) if len(s) else np.nan

                bars = pd.DataFrame({
                    "condition": ["Brazil (home)", "Brazil → Sri Lanka",
                                  "Persistence", "Sri Lanka only", "Fine-tuned"],
                    "pr_auc": [0.910, _v("2_brazil_zeroshot"), _v("1_persistence"),
                               _v("3_srilanka_only"), _v("4_brazil_finetuned")]})
                f = px.bar(bars, x="condition", y="pr_auc", text="pr_auc", height=320,
                           labels={"pr_auc": "PR-AUC", "condition": ""},
                           color="pr_auc", color_continuous_scale="RdYlGn")
                f.update_traces(texttemplate="%{text:.3f}")
                f.update_layout(coloraxis_showscale=False, margin={"t": 20})
                st.plotly_chart(f, width="stretch")

    # ---------------- data & methods ----------------
    with t4:
        st.error(
            "**Three of four public dengue screening datasets turned out to be synthetic.** "
            "In one, body temperature is *completely disjoint* between classes — dengue "
            "38.1–40.6 °C, non-dengue 36.0–37.6 °C, zero overlap. In another, the label is "
            "reproduced at 98.97% accuracy by the single rule `platelet < 150,000`. Models "
            "trained on them score 0.99+ internally and collapse to AUC 0.53–0.61 on real "
            "patients — worse than guessing."
        )
        st.markdown(
            "**The rule now applied to every dataset before modelling:** any single feature "
            "reaching AUC ≥ 0.95, or showing disjoint class ranges, blocks the dataset. "
            "Serology (NS1/IgM/IgG) is used only as ground truth for the label, never as a "
            "predictor — feeding the confirmatory test in as an input is circular."
        )
        au = R["audit"]
        if au is not None:
            cols = [c for c in ["dataset", "n", "prevalence", "max_single_feature_auc",
                                "verdict"] if c in au.columns]
            st.dataframe(au[cols], hide_index=True, width="stretch")

        st.markdown("#### What was used")
        st.dataframe(pd.DataFrame([
            {"Layer": "Screening", "Source": "Mendeley 6fsrsk3mb8",
             "Detail": "1,511 febrile patients, 19 CBC variables"},
            {"Layer": "Complications", "Source": "Zenodo 6476112",
             "Detail": "303 paediatric dengue admissions"},
            {"Layer": "Outbreak — Brazil", "Source": "Zenodo 22029053",
             "Detail": "4.7 M municipality-weeks, 2010–2025"},
            {"Layer": "Outbreak — Sri Lanka", "Source": "denguedatahub + NASA POWER",
             "Detail": "26 districts × 1,012 weeks, 2006–2026"},
        ]), hide_index=True, width="stretch")

        with st.expander("Method notes and reproducibility"):
            st.markdown(
                "- **Splits are strictly temporal.** Train ≤2021, validate 2022–23, test "
                "2024–25. Thresholds are tuned on validation only.\n"
                "- **Screening uses nested cross-validation** with tuning inside each outer "
                "fold and isotonic calibration, so the reported score is unbiased.\n"
                "- **Lags are date-exact.** The published Brazil panel carried 8,107 "
                "duplicate (municipality, week) rows whose positional lag columns were "
                "misaligned; features are now rebuilt by joining on the actual calendar "
                "date. Impact on the headline: −0.0006 PR-AUC.\n"
                "- **Operating points target sensitivity, not accuracy** — missing a case "
                "costs more than a false alarm in both clinical models.\n"
                "- **Frozen release** `frozen/v2_final/`: 72 artifacts, SHA-256 verified, "
                "with models, feature lists, thresholds, seeds and package versions."
            )
            if R["delay"] is not None:
                st.dataframe(R["delay"].round(4), hide_index=True, width="stretch")
                st.caption("Reporting-delay stress test: archive counts are backfilled, so "
                           "features are shifted back 1–2 extra weeks to simulate what "
                           "would genuinely have been available at forecast time.")
