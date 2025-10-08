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
    model_path = "random_forest_model.pkl"   # place model in models/ folder
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
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)

    with col2:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        cholesterol = st.number_input("Cholesterol (mmol/L)", min_value=2.0, max_value=15.0, value=5.0)
        glucose = st.number_input("Glucose (mmol/L)", min_value=2.0, max_value=30.0, value=5.5)

    with col3:
        smoker = st.selectbox("Smoking Status", ["No", "Yes"])
        physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        family_history = st.selectbox("Family History of Hypertension", ["No", "Yes"])
        alcohol_intake = st.selectbox("Alcohol Consumption", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    # Prepare input data with correct feature names
    input_data = pd.DataFrame({
        "AGE": [age],
        "GENDER": [1 if gender == "Male" else 0],
        "BMI": [bmi],
        "BLOOD SUGAR(mmol/L)": [glucose],
        "SYSTOLIC BP": [systolic_bp],
        "DIASTOLIC BP": [diastolic_bp],
        "HTN": [1],                 # placeholder if model expects it
        "DIABETES": [0],            # default if not in form
        "BOTH DM+HTN": [0],         # default if not in form
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

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Interpret results and generate message
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = (
                "Your predicted hypertension risk is **low**. "
                "This suggests normal blood pressure levels and minimal risk of related complications. "
                "Maintain a healthy lifestyle, balanced diet, and regular physical activity."
            )
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = (
                "Your predicted hypertension risk is **moderate**. "
                "This may indicate pre-hypertension or early signs of elevated blood pressure. "
                "It's advisable to monitor your BP regularly, reduce salt intake, and maintain a healthy weight."
            )
        else:
            risk_level = "🔴 High Risk"
            message = (
                "Your predicted hypertension risk is **high**. "
                "This may signal possible hypertension and higher risk of heart disease, kidney problems, or stroke. "
                "Please consult a healthcare provider for detailed evaluation and management."
            )

        # ---------- DISPLAY RESULTS ----------
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Hypertensive' if pred == 1 else 'Normal'}")
        st.write(f"**Probability of Hypertension:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # ---------- RELATED HEALTH RISKS ----------
        st.markdown("### 🫀 Related Risk Summary")

        if prob < 0.33:
            st.write("**Cardiovascular Strain:** Low")
            st.write("**Stroke Risk:** Low")
            st.write("**Diabetes Correlation:** Low")
        elif prob < 0.66:
            st.write("**Cardiovascular Strain:** Moderate")
            st.write("**Stroke Risk:** Moderate")
            st.write("**Diabetes Correlation:** Slightly Elevated")
        else:
            st.write("**Cardiovascular Strain:** High")
            st.write("**Stroke Risk:** High")
            st.write("**Diabetes Correlation:** High")
            st.warning("⚠️ Please consider full medical evaluation for cardiovascular and metabolic health.")

        # ---------- SAVE LOG ----------
        record = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Age": [age],
            "Gender": [gender],
            "BMI": [bmi],
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
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


