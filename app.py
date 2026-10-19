# -*- coding: utf-8 -*-
"""
app.py — Smart Recruitment Assistant · Streamlit dashboard

Powers the graduation-project deliverable "Recruitment dashboard".
Loads the trained pipeline from recruitment_champion.pkl (created by save_artifacts.py /
the notebook Section 18) and uses aug_train.csv only for look-up statistics and trends.

Run:
    streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

PRIMARY, ACCENT, SECOND, GREEN = "#1E2761", "#F96167", "#CADCFC", "#2E8B57"
CAT_ORDER_EDU = ["Primary School", "High School", "Graduate", "Masters", "Phd", "UNKNOWN"]
CAT_ORDER_SIZE = ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "UNKNOWN"]

st.set_page_config(page_title="Smart Recruitment Assistant", page_icon="🎯", layout="wide")

# ------------------------------------------------------------------ data / model
@st.cache_resource
def load_bundle():
    try:
        return joblib.load("recruitment_champion.pkl"), None
    except FileNotFoundError:
        return None, ("`recruitment_champion.pkl` not found — generate it first with "
                      "`python save_artifacts.py` (needs `aug_train.csv` in this folder).")

@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_csv("aug_train.csv")
    df["experience"] = df["experience"].replace({"ERROR": np.nan, ">20": "21", "<1": "0"})
    df["experience"] = pd.to_numeric(df["experience"])
    df["last_new_job"] = df["last_new_job"].replace({"ERROR": np.nan, ">4": "5", "never": "0"})
    df["last_new_job"] = pd.to_numeric(df["last_new_job"])
    df["company_size"] = df["company_size"].replace("10/49", "10-49")
    for col in ["gender", "education_level", "enrolled_university",
                "major_discipline", "company_size", "company_type"]:
        df[col] = df[col].fillna("UNKNOWN")
    return df

bundle, bundle_err = load_bundle()
if bundle_err:
    st.error(bundle_err)
    st.stop()

model = bundle["model"]
MODEL_NAME = bundle["model_name"]
THRESHOLD = float(bundle["threshold"])
FEATURE_NAMES = bundle["feature_names"]
CITY_FREQ = bundle["city_freq"]
NEEDS_SCALING = bool(bundle.get("needs_scaling", MODEL_NAME == "Logistic Regression"))
TEST_SCORES = bundle["test_scores"].copy()
RAW = load_raw()

def exp_median():
    return float(bundle["imputation"]["experience_median"]) if "imputation" in bundle else float(RAW["experience"].median())

def lnj_median():
    return float(bundle["imputation"]["last_new_job_median"]) if "imputation" in bundle else float(RAW["last_new_job"].median())

# ------------------------------------------------------------------ preprocessing (mirrors the notebook exactly)
ORDINAL_EDU = {"UNKNOWN": 0, "Primary School": 1, "High School": 2, "Graduate": 3, "Masters": 4, "Phd": 5}
ORDINAL_SIZE = {"UNKNOWN": 0, "<10": 1, "10-49": 2, "50-99": 3, "100-500": 4,
                "500-999": 5, "1000-4999": 6, "5000-9999": 7, "10000+": 8}
ONEHOT_COLS = ["gender", "enrolled_university", "major_discipline", "company_type"]
RAW_FEATURE_COLS = ["city", "city_development_index", "gender", "relevent_experience",
                    "enrolled_university", "education_level", "major_discipline",
                    "experience", "company_size", "company_type", "last_new_job", "training_hours"]

def _bucket(x):
    return 0 if x <= 3 else (1 if x <= 9 else 2)

def engineer(frame, exp_med=None, lnj_med=None):
    out = frame.copy()
    out["experience"] = out["experience"].fillna(exp_med if exp_med is not None else exp_median())
    out["last_new_job"] = out["last_new_job"].fillna(lnj_med if lnj_med is not None else lnj_median())
    out["experience_level"] = out["experience"].apply(lambda x: _bucket(float(x)))
    out["is_recent_mover"] = ((out["last_new_job"] <= 1) & (out["experience"] > 1)).astype(int)
    out["training_per_experience"] = out["training_hours"] / (out["experience"] + 1)
    out["experience_x_cdi"] = out["experience"] * out["city_development_index"]
    return out

def encode(encoded_in):
    out = engineer(encoded_in)
    out["city_frequency"] = out["city"].map(CITY_FREQ).fillna(0.0)
    out = out.drop(columns=["city"])
    out["relevent_experience"] = out["relevent_experience"].map(
        {"Has relevent experience": 1, "No relevent experience": 0})
    out["education_level"] = out["education_level"].map(ORDINAL_EDU).fillna(0)
    out["company_size"] = out["company_size"].map(ORDINAL_SIZE).fillna(0)
    out = pd.get_dummies(out, columns=ONEHOT_COLS, dtype=int)

    # ordinal maps may yield NaN for values outside the training mapping — the notebook maps unseen to 0
    out[["education_level", "company_size"]] = out[["education_level", "company_size"]].fillna(0)
    return out.reindex(columns=FEATURE_NAMES, fill_value=0)

def score(encoded_in):
    enc = encode(encoded_in)
    matrix = enc.to_numpy()
    if NEEDS_SCALING:
        matrix = bundle["scaler"].transform(matrix)
    proba = model.predict_proba(matrix)[:, 1]
    return proba, (proba >= THRESHOLD).astype(int)

# ------------------------------------------------------------------ UI shared
DISCLAIMER = ("⚠️ This model predicts **job-change intent**, not qualification. "
              "Use it to **order outreach** — never to reject a candidate. "
              "Predicted probability is a ranking score, not a calibrated chance.")

def kpi_cards(cols, items: dict):
    for col, (label, (value, help_)) in zip(cols, items.items()):
        with col:
            st.metric(label=label, value=value, help=help_)

# ================================================================== TABS
tab_overview, tab_perf, tab_fi, tab_rank, tab_screen = st.tabs(
    ["📌 Overview", "📈 Model Performance", "🧭 What Drives the Prediction",
     "🏆 Candidate Ranking", "🔍 Screen a Candidate"])

# ------------------------------------------------------------------ Tab 1: Overview
with tab_overview:
    st.title("🎯 Smart Recruitment Assistant")
    st.caption("AI-based candidate screening & ranking · HR Analytics: Job Change of Data Scientists")

    cand = RAW
    n_rows = len(cand)
    base_rate = float((cand["target"] == 1).mean())

    kpi_cards(st.columns(4), {
        "Candidates": (f"{n_rows:,}", "Candidate profiles in the HR Analytics dataset"),
        "Job-seekers": (f"{base_rate:.1%}", "Share labelled as actively looking for a job change"),
        "ROC AUC": (f"{roc_auc_score(TEST_SCORES['actually_looking'], TEST_SCORES['predicted_probability']):.3f}",
                    "Champion-model discrimination on the untouched test set"),
        "Recruiter lift": (f"{(TEST_SCORES.head(max(1, int(len(TEST_SCORES) * 0.1)))['actually_looking'].mean() / base_rate):.1f}×",
                           "Coverage of real job-seekers inside the ranked top 10% vs the base rate"),
    })

    st.divider()
    st.subheader("Candidate profile trends")
    left, right = st.columns([1, 2])
    with left:
        edu_f = st.multiselect("Education level", CAT_ORDER_EDU,
                               default=[c for c in CAT_ORDER_EDU if c in RAW["education_level"].unique()])
        size_f = st.multiselect("Company size", CAT_ORDER_SIZE,
                                default=[c for c in CAT_ORDER_SIZE if c in RAW["company_size"].unique()])
        gender_f = st.selectbox("Gender", ["All", "Male", "Female", "Other"], index=0)
        n_f = st.selectbox("Experience", ["All bands", "≤ 3 yrs", "4–9 yrs", "10+ yrs"], index=0)
    with right:
        fdf = RAW[(RAW["education_level"].isin(edu_f)) & (RAW["company_size"].isin(size_f))]
        if gender_f != "All":
            fdf = fdf[fdf["gender"] == gender_f]
        if n_f == "≤ 3 yrs":
            fdf = fdf[fdf["experience"] <= 3]
        elif n_f == "4–9 yrs":
            fdf = fdf[(fdf["experience"] >= 4) & (fdf["experience"] <= 9)]
        elif n_f == "10+ yrs":
            fdf = fdf[fdf["experience"] >= 10]

        if len(fdf) == 0:
            st.warning("The filter combination selects no candidates — loosen one filter.")
        else:
            rate = float((fdf["target"] == 1).mean())
            st.metric("Selected candidates", f"{len(fdf):,}",
                      help=f"Job-seeker share inside the selection: {rate:.1%}")
            trend = []
            for group_col, label, order in [
                ("education_level", "By education level", CAT_ORDER_EDU),
                ("company_size", "By company size", CAT_ORDER_SIZE),
            ]:
                mean_by = fdf.groupby(group_col)["target"].mean()
                rows = [(c, float(mean_by[c])) for c in order if c in mean_by.index]
                t = pd.DataFrame(rows, columns=["Category", "pct"])
                t["Breakdown"] = label
                trend.append(t)
            trend_df = pd.concat(trend, ignore_index=True)
            fig = go.Figure()
            for label, chunk in trend_df.groupby("Breakdown", sort=False):
                fig.add_bar(x=chunk["Category"], y=chunk["pct"], name=label,
                            marker_color=[PRIMARY] * len(chunk))
            fig.update_layout(bargap=0.25, legend_title_text=None,
                              yaxis_tickformat=".0%", yaxis_title="% looking for a change",
                              xaxis_title=None, height=330, margin=dict(l=10, r=10, t=28, b=0))
            fig.update_yaxes(range=[0, max(0.55, trend_df["pct"].max() * 1.15)])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.info(DISCLAIMER, icon="ℹ️")

# ------------------------------------------------------------------ Tab 2: Model Performance
with tab_perf:
    st.subheader("Model performance — three models compared")
    try:
        comp = pd.read_csv("model_comparison.csv")
    except FileNotFoundError:
        comp = pd.DataFrame([bundle["test_metrics"]])

    show = comp.copy()
    show.insert(1, "Champion", ["✅" if m == MODEL_NAME else "" for m in show["Model"]])
    best = show.numeric_rows = 0
    styled = show.style.format({c: "{:.3f}" for c in show.columns if show[c].dtype.kind == "f"})
    st.dataframe(styled, width="stretch", hide_index=True)

    st.info("Accuracy flatters the models: predicting “not looking” for everyone already scores "
            "75.1%. Precision, recall, F1 and the confusion matrix carry the real signal.", icon="💡")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        y_true = TEST_SCORES["actually_looking"].astype(int).values
        y_pred = (TEST_SCORES["predicted_probability"] >= THRESHOLD).astype(int).values
        cm = confusion_matrix(y_true, y_pred)
        labels = np.array([["TN\n" + f"{cm[0,0]:,}", "FP\n" + f"{cm[0,1]:,}"],
                           ["FN\n" + f"{cm[1,0]:,}", "TP\n" + f"{cm[1,1]:,}"]])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no", "Predicted: looking"],
                                   y=["Actual: no", "Actual: looking"],
                                   text=labels, texttemplate="%{text}", colorscale=[[0, "#EAF0FA"], [1, PRIMARY]],
                                   showscale=False, xgap=3, ygap=3))
        fig.update_layout(title=f"{MODEL_NAME} — confusion matrix (test set, threshold {THRESHOLD:.3f})",
                          margin=dict(t=48, l=4, r=4, b=4), height=380)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        fpr, tpr, _ = roc_curve(y_true, TEST_SCORES["predicted_probability"])
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=MODEL_NAME,
                        line=dict(color=PRIMARY, width=3),
                        fill="tozeroy", fillcolor="rgba(30,39,97,0.10)")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC 0.500)",
                        line=dict(color="grey", dash="dash", width=1))
        fig.update_layout(title=f"ROC — AUC {roc_auc_score(y_true, TEST_SCORES['predicted_probability']):.3f}",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          margin=dict(t=48, l=4, r=4, b=4), height=380)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption("Thresholds are tuned on the validation PR curve (F1-maximising point); "
               "the test set is scored exactly once.")

# ------------------------------------------------------------------ Tab 3: What drives the prediction
with tab_fi:
    st.subheader("What drives the prediction — most important hiring factors")
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
        source = f"{MODEL_NAME} feature importance (impurity-based)"
    else:
        importance = pd.Series(np.abs(model.coef_[0]), index=FEATURE_NAMES)
        source = f"{MODEL_NAME} |coefficient| (standardised)"
    top = importance.sort_values(ascending=False).head(10)[::-1]

    label_map = {
        "experience_level": "Experience band (engineered)", "city_development_index": "City development index",
        "company_size": "Company size", "major_discipline_UNKNOWN": "Major: not stated", "company_type_UNKNOWN": "Company type: not stated",
        "city_frequency": "City (train frequency)", "education_level": "Education level", "relevent_experience": "Relevant experience",
        "company_type_Pvt Ltd": "Company: Pvt Ltd", "last_new_job": "Yrs since last job change",
        "training_hours": "Training hours", "is_recent_mover": "Recent job mover (engineered)",
    }
    top.index = [label_map.get(k, k) for k in top.index]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                           marker_color=[ACCENT if "not stated" in i else PRIMARY for i in top.index]))
    fig.update_layout(title=source, xaxis_title="Importance", height=430,
                      margin=dict(l=4, r=4, t=40, b=4))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        st.success("**Context beats credentials** — where a candidate works and how experienced "
                   "they are out-predicts academic background for job-change intent.")
    with c2:
        st.success("**Missing info is a signal** — highlighted in red: “not stated” flags (major, "
                   "company type) rank high, so profiles that skip fields behave differently.")

# ------------------------------------------------------------------ Tab 4: Candidate Ranking
with tab_rank:
    st.subheader("Candidate ranking — how HR uses the model")
    ranked = TEST_SCORES.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    base_rate = float(ranked["actually_looking"].mean())
    k_pct = st.slider("Work the top K% of the ranked list", 5, 50, 10, 1,
                      help="The recruiter contacts this slice first, instead of working applicants in arrival order")
    head_n = max(1, int(len(ranked) * k_pct / 100))
    hit = float(ranked.head(head_n)["actually_looking"].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Base rate", f"{base_rate:.1%}", "Random applicant = job seeker")
    c2.metric(f"Hit rate @ top {k_pct}%", f"{hit:.1%}", f"{len(ranked.head(head_n)):,} candidates checked")
    c3.metric("Recruiter lift", f"{hit / base_rate:.1f}×", "vs. contacting applicants in random order")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Top 10 candidates to call first**")
        tbl = ranked.head(10).copy()
        tbl["predicted_probability"] = tbl["predicted_probability"].round(3)
        tbl["actually_looking"] = tbl["actually_looking"].map({1.0: "Yes ✅", 0.0: "No ❌"})
        st.dataframe(tbl, width="stretch", hide_index=True)
    with right:
        si = ranked.head(300).copy()
        si["in_slice"] = ["Yes" if i < head_n else "No" for i in range(len(si))]
        si["color"] = np.where(si["in_slice"] == "Yes", ACCENT, SECOND)
        fig = go.Figure()
        fig.add_scatter(x=list(range(1, 301)), y=si["predicted_probability"], mode="lines",
                        name="Ranked candidates", line=dict(color=PRIMARY, width=2))
        fig.add_shape(type="line", x0=head_n, x1=head_n, y0=si["predicted_probability"].min(), y1=1.0,
                      line=dict(color=ACCENT, dash="dash"))
        fig.add_shape(type="line", x0=1, x1=300, y0=base_rate, y1=base_rate,
                      line=dict(color=GREEN, dash="dot"))
        fig.update_layout(title=f"Champion scores — first {head_n} highlighted (red); "
                                f"green line = {base_rate:.0%} base rate",
                          xaxis_title="Rank", yaxis_title="Predicted P(looking)", height=380,
                          margin=dict(t=48, l=4, r=4, b=4), showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption("Ranking is computed on the untouched test set — the same candidates shown in the "
               "notebook Section 16 bonus deliverable.")

# ------------------------------------------------------------------ Tab 5: Screen a Candidate
with tab_screen:
    st.subheader("Screen a candidate — champion-model scoring")
    st.caption("Enter a fresh applicant and see exactly how the model would prioritise them.")

    cities = list(CITY_FREQ.index[:30])
    with st.form("screen_form"):
        colA, colB, colC, colD = st.columns(4)
        city = colA.selectbox("City", cities + ["Other"], index=cities.index("city_103") if "city_103" in cities else 0)
        cdi = colA.slider("City development index", 0.448, 0.949, 0.903, 0.001,
                          help="0.448 (low development) → 0.949 (highest)")
        gender = colA.selectbox("Gender", ["Male", "Female", "Other"])
        rel_exp = colB.selectbox("Relevant experience", ["Has relevent experience", "No relevent experience"])
        major = colB.selectbox("Major discipline", ["STEM", "Business Degree", "Humanities", "Arts", "Other", "No Major"])
        edu = colB.selectbox("Education level", ["Primary School", "High School", "Graduate", "Masters", "Phd"])
        univ = colC.selectbox("University enrolment", ["no_enrollment", "Part time course", "Full time course"])
        train_hours = colC.slider("Training hours", 1, 336, 65)
        exp_raw = colC.selectbox("Years of experience", ["<1"] + [str(v) for v in range(1, 21)] + [">20"], index=5)
        lnj = colC.selectbox("Years since last job change", ["never", "1", "2", "3", "4", ">4"], index=1)
        ctype = colD.selectbox("Company type", ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Early Stage Startup", "Other"])
        csize = colD.selectbox("Company size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "UNKNOWN"])
        submitted = st.form_submit_button("Get probability", type="primary", width="stretch", key="screen_submit")

    exp_val = 0.0 if exp_raw == "<1" else (21.0 if exp_raw == ">20" else float(exp_raw))
    lnj_val = 0.0 if lnj == "never" else (5.0 if lnj == ">4" else float(lnj))
    city_val = city if city != "Other" else "NULL_outside_train"
    cand_raw = pd.DataFrame([{
        "city": city_val,
        "city_development_index": cdi,
        "gender": gender,
        "relevent_experience": rel_exp,
        "enrolled_university": univ,
        "education_level": edu,
        "major_discipline": major,
        "experience": exp_val,
        "company_size": csize,
        "company_type": ctype,
        "last_new_job": lnj_val,
        "training_hours": train_hours,
    }])[RAW_FEATURE_COLS]

    if submitted:
        proba, _ = score(cand_raw)
        p = float(proba[0])
        look_flag = p >= THRESHOLD
        rank_n = int((TEST_SCORES["predicted_probability"] < p).sum()) + 1

        b1, b2 = st.columns([1.4, 1])
        with b1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p,
                number={"valueformat": ".1%", "font": {"size": 52, "color": PRIMARY}},
                gauge={"axis": {"range": [0, 1], "tickformat": ".0%",
                                "tickvals": [0, 0.25, 0.5, 0.75, 1.0]},
                       "bar": {"color": PRIMARY, "thickness": 0.30},
                       "steps": [{"range": [0, THRESHOLD], "color": "#E7ECF6"},
                                 {"range": [THRESHOLD, 1], "color": "#FBE0E2"}],
                       "threshold": {"line": {"color": ACCENT, "width": 3}, "value": THRESHOLD}},
                title={"text": "P(candidate is looking for a change)"}))
            fig.update_layout(height=290, margin=dict(l=30, r=30, t=60, b=30))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b2:
            if look_flag:
                st.success(f"**High priority — likely actively looking**\n\n"
                           f"Probability {p:.1%} ≥ tuned threshold {THRESHOLD:.3f}")
            else:
                st.info(f"**Low priority**\n\n"
                        f"Probability {p:.1%} < tuned threshold {THRESHOLD:.3f}")
            st.metric("Approximate rank", f"#{rank_n:,}",
                      help=f"Out of {len(TEST_SCORES):,} unseen test candidates scored by the same champion model")

        base_row = {
            "city": CITY_FREQ.index[0], "city_development_index": float(RAW["city_development_index"].median()),
            "gender": RAW["gender"].mode()[0], "relevent_experience": RAW["relevent_experience"].mode()[0],
            "enrolled_university": RAW["enrolled_university"].mode()[0],
            "education_level": RAW["education_level"].mode()[0],
            "major_discipline": RAW["major_discipline"].mode()[0],
            "experience": exp_median(), "company_size": RAW["company_size"].mode()[0],
            "company_type": RAW["company_type"].mode()[0],
            "last_new_job": lnj_median(), "training_hours": float(RAW["training_hours"].median()),
        }
        display_map = {
            "city_development_index": "City development index", "gender": "Gender",
            "relevent_experience": "Relevant experience", "enrolled_university": "University enrolment",
            "education_level": "Education level", "major_discipline": "Major discipline",
            "experience": "Years of experience", "company_size": "Company size",
            "company_type": "Company type", "last_new_job": "Yrs since last job change",
            "training_hours": "Training hours", "city": "City",
        }
        drivers = []
        for dim in base_row:
            delta_row = cand_raw.iloc[0].to_dict()
            delta_row[dim] = base_row[dim]
            p_alt = float(score(pd.DataFrame([delta_row])[RAW_FEATURE_COLS])[0][0])
            drivers.append({"Input": display_map[dim], "Δ probability": round(p - p_alt, 4)})
        drv = (pd.DataFrame(drivers).reindex(
            pd.DataFrame(drivers)["Δ probability"].abs().sort_values(ascending=False).index).head(3))
        st.markdown("##### Heuristic drivers of this prediction")
        st.caption("Approximation: probability delta when this input is replaced by a typical "
                   "candidate's value. Direction: positive pushes towards “looking for a change”.")
        drv.insert(0, "", [f"{'▲' if v > 0 else '▼'}" for v in drv["Δ probability"]])
        st.dataframe(drv, width="stretch", hide_index=True)

    st.warning(DISCLAIMER, icon="🚫")

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown(f"# 🎯 Smart Recruitment\nAssistant")
    st.markdown(f"Champion model: **{MODEL_NAME}**")
    st.markdown(f"Tuned decision threshold: **{THRESHOLD:.3f}**")
    if "val_auprc" in bundle:
        st.markdown(f"Validation AUPRC: **{float(bundle['val_auprc']):.3f}**")
    if "trained_on" in bundle:
        st.markdown(f"`{bundle['trained_on']}`")
    st.divider()
    st.markdown("**Deliverable:** recruitment dashboard (bonus) \n\n"
                "Built as part of the AI Professional Diploma graduation project.")
    with st.expander("About the data"):
        st.caption("Dataset: “HR Analytics: Job Change of Data Scientists” (Kaggle) — 19,158 "
                   "labelled candidate profiles. Processed datasets shipped: processed_dataset.csv, "
                   "processed_candidates_encoded.csv, test_scores.csv, model_comparison.csv.")
    with st.expander("Reproduce artifacts"):
        st.code("python save_artifacts.py\nstreamlit run app.py", language="bash")
