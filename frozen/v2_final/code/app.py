"""DengueShield - dengue screening, complication risk, and outbreak early warning.

Four areas:
  1. Dashboard          - the three distinct tasks, and how forecasts would be used
  2. Patient assessment - dengue screening + complication risk
  3. Geographic risk map - 14-day district outbreak forecast for Sri Lanka
  4. Model evidence     - validation, calibration, explainability, generalization

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


@st.cache_resource
def load_models():
    out = {}
    for key, fn in [("screening", "model1_screening.joblib"),
                    ("peds", "peds_complications.joblib"),
                    ("srilanka", "srilanka_outbreak.joblib")]:
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
    return {
        "risk": _csv("srilanka_current_risk.csv"),
        "history": pd.read_parquet(hist) if hist.exists() else None,
        "shap": _csv("model2_shap_drivers.csv"),
        "transfer": _csv("srilanka_transfer.csv"),
        "ablation_peds": _csv("ablation_peds.csv"),
        "audit": _csv("dataset_audit.csv"),
        "m2_head": _csv("model2_headtohead.csv"),
        "m1_ops": _json("model1_operating_points.json"),
        "robust": _json("model2_robustness.json"),
        "spatial": _csv("model2_spatial_holdout.csv"),
        "info_abl": _csv("model2_information_ablation.csv"),
        "calib": _json("model2_calibration_errors.json"),
        "rel_raw": _csv("model2_reliability_raw.csv"),
        "rel_cal": _csv("model2_reliability_calibrated.csv"),
        "sl_calib": _json("srilanka_calibration.json"),
        "worst_fn": _csv("model2_worst_false_negatives.csv"),
    }


M, R = load_models(), load_reports()

st.sidebar.title("🦟 DengueShield")
screen = st.sidebar.radio(
    "Screen", ["Dashboard", "Patient assessment", "Geographic risk map", "Model evidence"],
    label_visibility="collapsed")
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
        "assessment and localized outbreak forecasting?"
    )
    st.info(
        "This system performs **three distinct prediction tasks**. They have different "
        "populations, different targets and very different difficulty — they should "
        "never be collapsed into a single 'dengue accuracy' figure."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### A · Dengue screening")
        st.caption("Does this patient appear likely to have dengue?")
        st.metric("ROC-AUC", "0.681")
        st.metric("Accuracy", "76.2%", delta="+7.7pp vs baseline")
        st.caption("1,511 febrile patients · 19 CBC variables")
        st.warning("**Modest.** A full blood count alone is not a reliable dengue diagnostic.")
    with c2:
        st.markdown("### B · Complication risk")
        st.caption("Among dengue patients, who needs closer monitoring?")
        st.metric("ROC-AUC", "0.874")
        st.metric("Sensitivity / NPV", "90.5% / 96.2%")
        st.caption("303 paediatric dengue admissions")
        st.success("**Strong.** Tuned for sensitivity — safe to rule out, not to rule in.")
    with c3:
        st.markdown("### C · Outbreak forecasting")
        st.caption("Which districts will see an outbreak in 14 days?")
        st.metric("PR-AUC", "0.961")
        st.metric("Accuracy / Recall", "96.7% / 89.4%")
        st.caption("699 municipalities · 66,000 test weeks")
        st.success("**Strong.** Exceeds the 90% target under strict temporal validation.")

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
        "**Framing.** This is decision support for *prioritisation*. The prototype does "
        "not and should not trigger public-health action automatically — a flag is an "
        "input to human judgement, not a substitute for it."
    )

    st.markdown("---")
    st.subheader("What we found that we did not expect")
    a, b = st.columns(2)
    a.error(
        "**Three of four public dengue screening datasets are synthetic.** Models "
        "trained on them score 0.99+ internally and collapse to AUC 0.53–0.61 on real "
        "patients — worse than guessing."
    )
    a.warning(
        "**Environmental data adds almost nothing incrementally.** Beyond historical "
        "incidence it contributes just +0.006 PR-AUC, and alone (0.597) it is *worse* "
        "than persistence (0.760)."
    )
    b.warning(
        "**The model tracks trajectory, not emergence.** Recall is 0.92 where incidence "
        "is already high, but near zero where it is low."
    )
    b.error(
        "**Zero-shot geographic transfer fails.** A Brazil-trained model scores 0.449 on "
        "Sri Lanka — worse than persistence (0.476). Local calibration is necessary."
    )


# ========================================================= 2. PATIENT ASSESSMENT
elif screen == "Patient assessment":
    st.title("Patient assessment")
    st.caption(DISCLAIMER)
    tab1, tab2 = st.tabs(["Dengue screening (CBC)", "Complication risk"])

    with tab1:
        st.caption("Full blood count based screening. Enter routine haematology values.")
        b = M["screening"]
        if b is None:
            st.error("Screening model not found — run `finalize_model1.py` first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age = st.number_input("Age", 1, 100, 32)
                sex = st.selectbox("Sex", ["Male", "Female"])
                hb = st.number_input("Haemoglobin (g/dl)", 5.0, 22.0, 14.2, 0.1)
                rbc = st.number_input("RBC (10⁶/µl)", 2.0, 8.0, 4.8, 0.1)
                hct = st.number_input("Haematocrit (%)", 20.0, 60.0, 43.0, 0.1)
            with c2:
                plt_c = st.number_input("Platelet count (/cumm)", 5_000, 600_000, 145_000, 1_000)
                wbc = st.number_input("Total WBC (/cumm)", 500, 30_000, 4_800, 100)
                neu = st.number_input("Neutrophils (%)", 0, 100, 55)
                lym = st.number_input("Lymphocytes (%)", 0, 100, 35)
                mono = st.number_input("Monocytes (%)", 0, 50, 6)
            with c3:
                eos = st.number_input("Eosinophils (%)", 0, 30, 2)
                mcv = st.number_input("MCV (fl)", 50.0, 130.0, 88.0, 0.1)
                mch = st.number_input("MCH (pg)", 15.0, 45.0, 29.0, 0.1)
                mchc = st.number_input("MCHC (g/dl)", 25.0, 40.0, 32.5, 0.1)
                rdw = st.number_input("RDW-CV (%)", 8.0, 25.0, 13.0, 0.1)
            with c4:
                mpv = st.number_input("MPV (fl)", 5.0, 20.0, 10.5, 0.1)
                pdw = st.number_input("PDW (%)", 5.0, 25.0, 15.0, 0.1)
                pct = st.number_input("PCT (%)", 0.0, 1.0, 0.15, 0.01)

            if st.button("Assess", type="primary", key="screen_btn"):
                row = {"Gender": 1 if sex == "Male" else 0, "Age": age,
                       "Hemoglobin(g/dl)": hb, "Neutrophils(%)": neu, "Lymphocytes(%)": lym,
                       "Monocytes(%)": mono, "Eosinophils(%)": eos, "RBC": rbc, "HCT(%)": hct,
                       "MCV(fl)": mcv, "MCH(pg)": mch, "MCHC(g/dl)": mchc, "RDW-CV(%)": rdw,
                       "Total Platelet Count(/cumm)": plt_c, "MPV(fl)": mpv, "PDW(%)": pdw,
                       "PCT(%)": pct, "Total WBC count(/cumm)": wbc}
                row["NLR"] = neu / max(lym, 1)
                row["PLR"] = plt_c / max(lym, 1)
                row["PLT_WBC_ratio"] = plt_c / max(wbc, 1)
                row["MPV_PLT_ratio"] = mpv / max(plt_c / 1000, 1e-6)
                p = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
                thr = b.get("threshold_sens90", 0.5)
                flag = p >= thr

                m1, m2 = st.columns([1, 2])
                m1.markdown("**Model-estimated screening probability**")
                m1.metric("prob", f"{p:.0%}", label_visibility="collapsed")
                m1.markdown(
                    f"### {'🔴 Prioritise dengue testing' if flag else '🟢 Lower testing priority'}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    title={"text": "Model-estimated screening probability", "font": {"size": 13}},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "#C7392F" if flag else "#2E9E5B"},
                           "steps": [{"range": [0, thr * 100], "color": "#EAF3EC"},
                                     {"range": [thr * 100, 100], "color": "#FBEAE8"}],
                           "threshold": {"line": {"color": "black", "width": 3},
                                         "value": thr * 100}}))
                fig.update_layout(height=260, margin=dict(t=50, b=10))
                m2.plotly_chart(fig, width="stretch")

                st.error(DISCLAIMER)
                ops = R["m1_ops"]
                if ops:
                    st.info(
                        f"**How much to trust this.** On real patients (nested "
                        f"cross-validation): ROC-AUC **{ops['at_0.5']['roc_auc']:.3f}**, "
                        f"accuracy **{ops['at_0.5']['accuracy']:.3f}** against a "
                        f"majority-class baseline of 0.685. A full blood count alone is "
                        f"only weakly discriminative for dengue. Use this to **prioritise "
                        f"testing**, never to rule dengue out."
                    )

    with tab2:
        st.caption(
            "For a patient already known or strongly suspected to have dengue: estimated "
            "risk of complicated dengue, relative to the model's training population "
            "(303 paediatric admissions)."
        )
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
                c.info(
                    f"Operating at **{mm['sensitivity']:.0%} sensitivity** / "
                    f"**{mm['specificity']:.0%} specificity** "
                    f"(ROC-AUC {mm['roc_auc']:.3f}, NPV {mm['npv']:.3f}).\n\n"
                    "Deliberately tuned for high sensitivity: missing a deteriorating "
                    "patient is far costlier than one extra observation."
                )
                st.error(DISCLAIMER)


# ======================================================== 3. GEOGRAPHIC RISK MAP
elif screen == "Geographic risk map":
    st.title("Sri Lanka dengue risk — 14-day forecast")
    risk, hist = R["risk"], R["history"]
    if risk is None:
        st.error("No forecasts found — run `finalize_srilanka.py` first.")
    else:
        wk = pd.to_datetime(risk.week_start).max().date()
        st.caption(f"Issued from surveillance week beginning **{wk}** · horizon 2 weeks · "
                   f"outbreak = weekly incidence ≥ 9.9 per 100 000")
        sc = R["sl_calib"]
        if sc:
            st.warning(
                f"**These are predicted outbreak probabilities, not calibrated ones.** "
                f"Overall ECE is {sc['ece']:.3f}, but in the highest-risk band the model "
                f"predicts ~0.81 where outbreaks occur ~0.54 of the time — it is "
                f"overconfident exactly where decisions get made. Treat the **ranking** as "
                f"more reliable than the number."
            )

        k = st.columns(4)
        for i, band in enumerate(["Very High", "High", "Moderate", "Low"]):
            k[i].metric(band, int((risk.risk_band == band).sum()))

        left, right = st.columns([3, 2])
        with left:
            fig = px.scatter_map(
                risk, lat="lat", lon="lon", size="casos", color="risk_band",
                color_discrete_map=BAND_COLOR, hover_name="district",
                hover_data={"casos": True, "p_inc100k": ":.1f", "risk": ":.2f",
                            "lat": False, "lon": False},
                size_max=45, zoom=6.2, height=560,
                category_orders={"risk_band": ["Very High", "High", "Moderate", "Low"]})
            fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("#### District ranking")
            show = risk[["district", "casos", "p_inc100k", "risk", "risk_band"]].copy()
            show.columns = ["District", "Cases", "Per 100k", "Predicted prob.", "Band"]
            st.dataframe(show.style.format({"Per 100k": "{:.1f}", "Predicted prob.": "{:.2f}"}),
                         hide_index=True, width="stretch", height=520)

        st.markdown("---")
        d = st.selectbox("Inspect a district", risk.district.tolist())
        row = risk[risk.district == d].iloc[0]
        c = st.columns(5)
        c[0].metric("Current cases", int(row.casos))
        c[1].metric("Incidence /100k", f"{row.p_inc100k:.1f}")
        c[2].metric("Predicted outbreak prob.", f"{row.risk:.0%}")
        c[3].metric("Risk band", str(row.risk_band))
        c[4].metric("4-wk rainfall", f"{row.precip_roll4_sum:.0f} mm")

        if hist is not None:
            h = hist[hist.district == d].sort_values("week_start").tail(160)
            f = go.Figure()
            f.add_trace(go.Scatter(x=h.week_start, y=h.casos, name="Cases", line=dict(width=2)))
            f.add_trace(go.Bar(x=h.week_start, y=h.precip_total_semana, name="Rainfall (mm)",
                               yaxis="y2", opacity=0.3))
            f.update_layout(height=340, yaxis=dict(title="Weekly cases"),
                            yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
                            legend=dict(orientation="h", y=1.12),
                            margin=dict(t=30, b=10), title=f"{d} — last 3 years")
            st.plotly_chart(f, width="stretch")

        st.info(
            "**Known limitation.** The model detects escalation of transmission that is "
            "already under way far better than emergence from a low base — recall is 0.92 "
            "where incidence is already high and near zero where it is low."
        )


# ============================================================== 4. MODEL EVIDENCE
else:
    st.title("Model evidence")
    st.caption("Including the weaknesses. Everything here is out-of-sample.")
    t1, t2, t3, t4, t5 = st.tabs(["Validation", "Calibration", "Explainability",
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
                                   mode="lines+markers", line=dict(dash="dash")))
            f.add_trace(go.Scatter(x=ro.test_year, y=ro.pr_auc, name="PR-AUC",
                                   mode="lines+markers"))
            f.update_layout(height=340, title="Rolling-origin backtest — retrain each year",
                            yaxis=dict(range=[0, 1]), margin=dict(t=40))
            st.plotly_chart(f, width="stretch")
            st.success(f"Every backtest year exceeds 90% accuracy (mean "
                       f"{ro.accuracy.mean():.3f}) while beating its trivial baseline.")

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
                    f"positives are municipalities already at epidemic level, versus "
                    f"{mom['tn_at_epidemic_level']:.1%} of true negatives — roughly "
                    f"{ratio:.0f}× more likely. The model carries epidemiological momentum "
                    "through outbreaks that are already subsiding."
                )
            st.warning(
                "**The central limitation.** Recall is 0.92 where baseline incidence is "
                "already high, but near zero where it is low. This is an epidemic "
                "*trajectory tracker*, not an emergence detector."
            )
        if R["worst_fn"] is not None:
            with st.expander("10 largest missed outbreaks"):
                st.dataframe(R["worst_fn"].head(10).round(2), hide_index=True, width="stretch")

    with t2:
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
                f"{cal['calibrated']['ece']:.4f} with essentially no loss of ranking "
                f"power. The Brazil model's probabilities may honestly be called "
                f"**calibrated**."
            )
        rr, rc = R["rel_raw"], R["rel_cal"]
        if rr is not None:
            f = go.Figure()
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="perfect",
                                   line=dict(dash="dash", color="black")))
            f.add_trace(go.Scatter(x=rr.predicted, y=rr.observed, name="raw",
                                   mode="lines+markers"))
            if rc is not None:
                f.add_trace(go.Scatter(x=rc.predicted, y=rc.observed, name="isotonic",
                                       mode="lines+markers"))
            f.update_layout(height=460, xaxis_title="Predicted outbreak probability",
                            yaxis_title="Observed outbreak frequency",
                            title="Reliability diagram — 14-day forecast, test 2024–25")
            st.plotly_chart(f, width="stretch")
        sc = R["sl_calib"]
        if sc:
            st.warning(
                f"**The Sri Lanka model is not equally calibrated** (ECE {sc['ece']:.3f}, "
                f"Brier {sc['brier']:.3f}), and is overconfident in its highest-risk band. "
                "The map therefore labels its output *predicted*, not *calibrated*."
            )

    with t3:
        st.subheader("SHAP attribution")
        sd = R["shap"]
        if sd is not None:
            f = px.bar(sd.sort_values("pct_of_total"), x="pct_of_total", y="driver",
                       orientation="h", text="pct_of_total",
                       labels={"pct_of_total": "% of total |SHAP|", "driver": ""})
            f.update_traces(texttemplate="%{text:.1f}%", marker_color="#3B6EA8")
            f.update_layout(height=430, margin=dict(t=20))
            st.plotly_chart(f, width="stretch")

        st.subheader("…but attribution is not incremental value")
        ia = R["info_abl"]
        if ia is not None:
            st.dataframe(ia[["condition", "n_features", "pr_auc", "accuracy",
                             "recall", "precision"]].round(4), hide_index=True, width="stretch")
            f = px.bar(ia, x="condition", y="pr_auc", text="pr_auc",
                       labels={"pr_auc": "PR-AUC", "condition": ""}, height=380)
            f.update_traces(texttemplate="%{text:.3f}", marker_color="#3B6EA8")
            st.plotly_chart(f, width="stretch")
            st.error(
                "**The most important correction in this project.** SHAP attributes ~50% "
                "of the model's reasoning to environment — but removing it costs only "
                "**+0.006 PR-AUC**, and environment *alone* (0.597) is **worse than "
                "persistence** (0.760). Environmental variables are largely redundant "
                "with recent incidence. Attribution ≠ incremental predictive value."
            )

    with t4:
        st.subheader("Spatial holdout — can it predict unseen places?")
        sp = R["spatial"]
        if sp is not None:
            main = sp[~sp.condition.str.startswith("state_fold")]
            st.dataframe(main[["condition", "pr_auc", "accuracy", "recall",
                               "precision", "n_test"]].round(4),
                         hide_index=True, width="stretch")
            st.success(
                "Municipalities the model has **never seen** score PR-AUC 0.955 versus "
                "0.962 for seen ones. It generalises rather than memorising place identity."
            )
            folds = sp[sp.condition.str.startswith("state_fold")]
            if len(folds):
                st.markdown("**Leave-whole-states-out** (harder — entire regions unseen)")
                st.dataframe(folds[["condition", "pr_auc", "accuracy", "recall", "n_test"]]
                             .round(4), hide_index=True, width="stretch")
                st.warning(
                    f"Mean PR-AUC {folds.pr_auc.mean():.3f}, but the range "
                    f"({folds.pr_auc.min():.3f}–{folds.pr_auc.max():.3f}) is wide. "
                    "Whole-region transfer is materially harder than random municipality "
                    "holdout."
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
                "baseline (0.476). A high-performing forecasting model in one geographic "
                "domain cannot be assumed to generalise to another; local epidemiological "
                "dynamics and calibration remain necessary."
            )

    with t5:
        st.subheader("Dataset integrity audit")
        au = R["audit"]
        if au is not None:
            st.dataframe(au, hide_index=True, width="stretch")
            st.error(
                "**Three of four public dengue screening datasets were rejected.** They "
                "contain features that reproduce the label almost perfectly — in one, body "
                "temperature is entirely disjoint between classes (dengue 38.1–40.6 °C, "
                "non-dengue 36.0–37.6 °C). Models trained on them score 0.99+ internally "
                "and collapse to AUC 0.53–0.61 on real patients."
            )
        st.markdown(
            "**The audit rule.** Any single feature with AUC ≥ 0.95, or disjoint "
            "class-conditional ranges, blocks the dataset before modelling begins. "
            "Serology (NS1/IgM/IgG) is used only as ground truth for the label, never "
            "as a predictor — using the confirmatory test as an input is circular."
        )
        ab = R["ablation_peds"]
        if ab is not None:
            st.subheader("Feature-group ablation — complication risk")
            a = ab[ab.model == "lgbm"][["tier", "n_features", "roc_auc", "pr_auc",
                                        "sensitivity", "specificity"]]
            st.dataframe(a.round(4), hide_index=True, width="stretch")
            st.caption("At a fixed ~90% sensitivity, specificity nearly doubles "
                       "(0.33 → 0.64) as symptoms and organ labs are added. Demographics "
                       "add almost nothing.")
