# hypertension_app.py
# 🌿 Hospital-Grade Hypertension Risk Prediction App

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Hypertension Risk Prediction", page_icon="🩺", layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"  # ensure the model file is in the same folder or specify full path
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------- HEADER ----------
st.title("🩺 Hypertension Risk Prediction System")
st.markdown("""
This AI-driven system assists healthcare providers in **predicting hypertension risk** 
based on patient vitals and clinical indicators.
""")

# ---------- INPUT FORM ----------
st.subheader("📋 Enter Patient Details")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45.0)
        gender = st.selectbox("Gender", ["M", "F"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=75.0)

    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.0)
        bp = st.text_input("BP (mmHg) [e.g. 135/85]", value="135/85")
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=10.0)

    with col3:
        diabetes = st.selectbox("Diabetes", [0, 1])
        both_dm_htn = st.selectbox("Both DM + HTN", [0.0, 1.0])
        treatment = st.selectbox("Treatment", ["None", "On Treatment"])

    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'AGE': [age],
        'GENDER': [1 if gender == "M" else 0],
        'WEIGHT(kg)': [weight],
        'BMI': [bmi],
        'BP(mmHg)': [bp],
        'BLOOD SUGAR(mmol/L)': [blood_sugar],
        'DIABETES': [int(diabetes)],
        'BOTH DM+HTN': [float(both_dm_htn)],
        'TREATMENT': [1 if treatment == "On Treatment" else 0]
    })

    try:
        # Align columns with model training features
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
        else:
            model_features = input_data.columns

        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_features]

        # Make prediction
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5

        # Interpret result
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = "Your predicted hypertension risk is **low**. Maintain a healthy lifestyle."
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = "Your predicted hypertension risk is **moderate**. Monitor BP and improve diet."
        else:
            risk_level = "🔴 High Risk"
            message = "Your predicted hypertension risk is **high**. Seek medical consultation."

        # ---------- DISPLAY RESULTS ----------
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Hypertensive' if pred == 1 else 'Normal'}")
        st.write(f"**Probability of Hypertension:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # ---------- SAVE LOG ----------
        record = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Age": [age],
            "Gender": [gender],
            "BMI": [bmi],
            "BP(mmHg)": [bp],
            "Blood_Sugar": [blood_sugar],
            "Probability": [prob],
            "Risk_Level": [risk_level]
        }
        record_df = pd.DataFrame(record)

        if not os.path.exists("prediction_logs.csv"):
            record_df.to_csv("prediction_logs.csv", index=False)
        else:
            record_df.to_csv("prediction_logs.csv", mode="a", header=False, index=False)

        st.success("✅ Prediction saved to log (prediction_logs.csv)")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Developed by Millicent Chesang | Powered by AI & Data Analytics for Public Health")
