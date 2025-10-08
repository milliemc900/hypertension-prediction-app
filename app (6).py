# app.py
# 🌿 Hospital-Grade Hypertension Risk Prediction App

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Hypertension Risk Prediction", page_icon="🩺", layout="wide")

# ---------- PASSWORD PROTECTION ----------
PASSWORD = "admin123"  # You can change this

def password_auth():
    """Simple password protection"""
    st.sidebar.title("🔒 Access Login")
    password = st.sidebar.text_input("Enter Password", type="password")
    if password != PASSWORD:
        st.sidebar.error("Incorrect password. Access denied.")
        st.stop()

password_auth()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "models/RandomForest_model.pkl"  # Ensure your model is here
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------- HEADER ----------
st.title("🩺 Hypertension Risk Prediction System")
st.markdown("""
This AI-driven system helps healthcare providers **predict hypertension risk** based on key clinical indicators.  
It supports early screening, decision-making, and follow-up planning.
""")

# ---------- INPUT FORM ----------
st.subheader("📋 Enter Patient Details")
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["M", "F"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=75.0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.0)

    with col2:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=135)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=85)
        glucose = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=10.0)

    with col3:
        diabetes = st.selectbox("Diabetes", [0, 1])
        both_dm_htn = st.selectbox("Both DM + HTN", [0.0, 1.0])
        treatment = st.selectbox(
            "Treatment Type",
            ['ab', 'abe', 'ae', 'ade', 'e', 'ad', 'aec', 'ace', 'ce', 'ebe', 'aw', 'ac', 'a']
        )

    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [1 if gender == 'M' else 0],
            'WEIGHT(kg)': [weight],
            'BMI': [bmi],
            'SYSTOLIC BP': [systolic_bp],
            'DIASTOLIC BP': [diastolic_bp],
            'BLOOD SUGAR(mmol/L)': [glucose],
            'DIABETES': [diabetes],
            'BOTH DM+HTN': [both_dm_htn],
            'TREATMENT': [treatment]
        })

        # Align with model features
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Interpret risk level
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = (
                "Predicted hypertension risk is **low**. Maintain a balanced diet, exercise, and regular checkups."
            )
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = (
                "Predicted hypertension risk is **moderate**. Monitor BP regularly and adopt heart-healthy habits."
            )
        else:
            risk_level = "🔴 High Risk"
            message = (
                "Predicted hypertension risk is **high**. Please seek medical evaluation for proper management."
            )

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
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
            "Probability": [prob],
            "Risk_Level": [risk_level],
            "Treatment": [treatment]
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
