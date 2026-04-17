import streamlit as st
import math

st.set_page_config(page_title="COVID-19 Prediction Study", layout="centered", page_icon="🦠")

st.markdown("""
<style>
    .stButton > button {
        background: #1e2130;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover { background: #2e3250; }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .hero-banner {
        border: 1px solid #2e3250;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        background: #1e2130;
    }
    .tag {
        display: inline-block;
        background: #2a2d3e;
        border: 1px solid #3e4260;
        color: #8b949e;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .result-positive {
        background: #2a1a1a;
        border: 1px solid #6e2020;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #ff6b6b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .result-negative {
        background: #1a2a1e;
        border: 1px solid #206e30;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #56d364;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h2 style="margin-bottom: 0.4rem;">🦠 COVID-19 Prediction Study</h2>
    <p style="color: #8b949e; margin-bottom: 1rem;">Answer the questions below based on the patient profile. All fields are optional.</p>
    <span class="tag">Logistic Regression Model</span>
    <span class="tag">100,000 Patient Records</span>
    <span class="tag">Phase 4 — User Study</span>
</div>
""", unsafe_allow_html=True)

st.subheader("1. What would you like to predict?")
outcome = st.radio("", ["COVID Positive", "Other Symptoms"], horizontal=True)

st.divider()

st.subheader("2. Patient Demographics")
st.caption("All fields are optional. Skip any you prefer not to answer.")
col1, col2 = st.columns(2)
with col1:
    age_input = st.text_input("Patient's age (optional)", placeholder="e.g. 45 or leave blank")
with col2:
    sex = st.selectbox("Gender identity", [
        "Prefer not to say", "Male", "Female", "Non-binary",
        "Transgender male", "Transgender female",
        "Genderqueer / Gender non-conforming", "Intersex", "Other", "Unknown"
    ])

st.divider()

st.subheader("3. Current Symptoms")
st.caption("Check all symptoms the patient is currently experiencing.")
col_s1, col_s2 = st.columns(2)
with col_s1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Dry cough")
    shortness_breath = st.checkbox("Shortness of breath")
    loss_taste_smell = st.checkbox("Loss of taste or smell")
with col_s2:
    fatigue = st.checkbox("Fatigue or body aches")
    sore_throat = st.checkbox("Sore throat")
    headache = st.checkbox("Headache")
    days_symptoms = st.selectbox("Days with symptoms", [
        "Prefer not to say", "No symptoms", "Less than 3 days",
        "3 to 5 days", "6 to 10 days", "More than 10 days"
    ])

st.divider()

st.subheader("4. Exposure and Medical History")
st.caption("All fields are optional.")
contact_covid = st.radio("Close contact with a confirmed COVID-19 case?",
    ["Prefer not to say", "Yes", "No", "Unknown"], horizontal=True)
traveled = st.radio("Traveled recently?",
    ["Prefer not to say", "Yes", "No"], horizontal=True)
vaccinated = st.radio("Vaccination status?",
    ["Prefer not to say", "Fully vaccinated", "Partially vaccinated", "Not vaccinated", "Unknown"])
tested_before = st.radio("Previously tested for COVID-19?",
    ["Prefer not to say", "Yes, tested positive", "Yes, tested negative", "Never tested"])

st.divider()

st.subheader("5. Pre-existing Conditions")
st.caption("Select all that apply. Leave blank if unknown.")
col3, col4 = st.columns(2)
with col3:
    diabetes = st.checkbox("Diabetes")
    obesity = st.checkbox("Obesity")
    cardiovascular = st.checkbox("Cardiovascular disease")
    copd = st.checkbox("Chronic lung disease (COPD)")
with col4:
    hypertension = st.checkbox("Hypertension")
    asthma = st.checkbox("Asthma")
    renal = st.checkbox("Chronic kidney disease")
    tobacco = st.checkbox("Tobacco use")

st.divider()

st.subheader("6. Severity Indicators")
st.caption("All fields are optional.")
col_v1, col_v2 = st.columns(2)
with col_v1:
    walk_breathless = st.radio("Can walk without breathlessness?",
        ["Prefer not to say", "Yes", "No"], horizontal=True)
    high_fever = st.radio("High fever above 38C?",
        ["Prefer not to say", "Yes", "No"], horizontal=True)
with col_v2:
    eat_drink = st.radio("Able to eat and drink normally?",
        ["Prefer not to say", "Yes", "No"], horizontal=True)
    confused = st.radio("Appears confused or disoriented?",
        ["Prefer not to say", "Yes", "No"], horizontal=True)

st.divider()

if outcome == "Other Symptoms":
    st.subheader("7. Clinical Status")
    st.caption("Indicate clinical interventions. Select Unknown if not applicable.")
    col5, col6 = st.columns(2)
    with col5:
        icu = st.selectbox("Admitted to ICU?", ["Unknown", "No", "Yes"])
        pneumonia = st.selectbox("Diagnosed with pneumonia?", ["Unknown", "No", "Yes"])
    with col6:
        intubed = st.selectbox("Intubated (on a ventilator)?", ["Unknown", "No", "Yes"])
        inmsupr = st.selectbox("Immunosuppressed?", ["Unknown", "No", "Yes"])
    st.divider()
else:
    icu = "No"
    intubed = "No"
    pneumonia = "No"
    inmsupr = "No"

if st.button("Submit and See Model Prediction"):

    age = int(age_input.strip()) if age_input.strip().isdigit() else 50

    score = 0
    top_factor = "General profile"

    if fever: score += 0.3
    if shortness_breath: score += 0.5
    if loss_taste_smell: score += 0.4
    if walk_breathless == "No": score += 0.4
    if confused == "Yes": score += 0.5
    if high_fever == "Yes": score += 0.3
    if contact_covid == "Yes": score += 0.5

    if outcome == "Other Symptoms":
        if intubed == "Yes":
            score += 1.9654
            top_factor = "Intubated"
        if icu == "Yes":
            score += 1.9399
            if intubed != "Yes":
                top_factor = "ICU admission"
        if pneumonia == "Yes":
            score += 0.8
            if intubed != "Yes" and icu != "Yes":
                top_factor = "Pneumonia"
        if renal: score += 0.5
        if inmsupr == "Yes": score += 0.45
        if diabetes: score += 0.3
        if hypertension: score += 0.25
        if age > 60:
            score += 0.4
            if intubed != "Yes" and icu != "Yes" and pneumonia != "Yes":
                top_factor = "Age over 60"
        if obesity: score += 0.2
    else:
        if diabetes:
            score += 0.6
            top_factor = "Diabetes"
        if obesity:
            score += 0.5
            if not diabetes:
                top_factor = "Obesity"
        if hypertension: score += 0.45
        if age > 50:
            score += 0.4
            if not diabetes and not obesity:
                top_factor = "Age over 50"
        if renal: score += 0.35

    prob = min(0.97, max(0.03, 1 / (1 + math.exp(-score + 1.2))))
    model_says_yes = prob >= 0.5
    model_label = "Yes" if model_says_yes else "No"
    model_confidence = round(prob * 100 if model_says_yes else (1 - prob) * 100)

    st.subheader("Results")
    col7, col8, col9 = st.columns(3)
    col7.metric("Model Prediction", model_label)
    col8.metric("Model Confidence", str(model_confidence) + "%")
    col9.metric("Top Risk Factor", top_factor)

    if model_says_yes:
        st.markdown('<div class="result-positive">⚠️ The model predicts this patient is at risk.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-negative">✅ The model predicts this patient is not at risk.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Risk probability: " + str(round(prob * 100)) + "% — Based on Logistic Regression model trained on 100,000 COVID-19 patient records.")