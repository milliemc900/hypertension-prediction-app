# hypertension_app.py
# 🌿 Hospital-Grade Hypertension Risk Prediction App (with Password Protection)

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Hypertension Risk Prediction", page_icon="🩺", layout="wide")

# ---------- PASSWORD PROTECTION ----------
PASSWORD = "admin123"  # 🔒 Change this before deploying
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Secure Access Portal")
    password_input = st.text_input("Enter Password to Continue", type="password")
    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access granted. Welcome!")
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "random_forest_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------- HEADER ----------
st.title("🩺 Hypertension Risk Prediction System")
st.markdown("""
This AI-driven system assists healthcare providers in **predicting hypertension risk** based on patient vitals and clinical indicators.  
It supports early screening, decision-making, and follow-up planning.
""")

# ---------- INPUT FORM ----------
st.subheader("📋 Enter Patient Details")
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5)

    with col2:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=5.5)
        diabetes = st.selectbox("Has Diabetes?", ["No", "Yes"])

    with col3:
        cholesterol = st.number_input("Cholesterol (mmol/L)", min_value=2.0, max_value=15.0, value=5.0)
        smoker = st.selectbox("Smoking Status", ["No", "Yes"])
        physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    # ✅ Input data matches your requested structure
    input_data = pd.DataFrame({
        'AGE': [age],
        'GENDER': [1 if gender == "Male" else 0],
        'WEIGHT(kg)': [weight],
        'BMI': [bmi],
        'BP(mmHg)': [f"{systolic_bp}/{diastolic_bp}"],
        'BLOOD SUGAR(mmol/L)': [blood_sugar],
        'DIABETES': [1 if diabetes == "Yes" else 0]
    })

    try:
        # Align model columns if available
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
        else:
            model_features = input_data.columns

        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_features]

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Interpret results
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = (
                "Low hypertension risk. Maintain a healthy diet and regular exercise."
            )
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = (
                "Moderate hypertension risk. Monitor BP regularly and limit salt intake."
            )
        else:
            risk_level = "🔴 High Risk"
            message = (
                "High hypertension risk. Please seek medical advice for proper evaluation."
            )

        # Display results
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Hypertensive' if pred == 1 else 'Normal'}")
        st.write(f"**Probability of Hypertension:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # Save log
        record = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Age": [age],
            "Gender": [gender],
            "Weight(kg)": [weight],
            "BMI": [bmi],
            "BP(mmHg)": [f"{systolic_bp}/{diastolic_bp}"],
            "BloodSugar(mmol/L)": [blood_sugar],
            "RiskLevel": [risk_level],
            "Probability": [prob],
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
