# hypertension_app.py
# 🩺 Hypertension Risk Prediction App

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Hypertension Risk Prediction", page_icon="🫀", layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"  # Ensure this file exists in your repo root

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please place your trained model here.")
        st.stop()

    return joblib.load(model_path)


# Load model
model = load_model()

# ---------- HEADER ----------
st.title("🩺 Hypertension Risk Prediction System")
st.markdown("""
This AI-powered system predicts the **risk of hypertension** based on patient vitals and clinical information.
""")

# ---------- INPUT FORM ----------
st.subheader("📋 Enter Patient Details")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    # --- Column 1 ---
    with col1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=45.0)
        gender = st.selectbox("Gender", ["M", "F"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)

    # --- Column 2 ---
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=27.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=135)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=85)
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=9.5)

    # --- Column 3 ---
    with col3:
        diabetes = st.selectbox("Diabetes (1=Yes, 0=No)", [0, 1])
        both_dm_htn = st.selectbox("Both DM + HTN (1=Yes, 0=No)", [0.0, 1.0])

        # ✅ Updated treatment options
        treatment = st.selectbox(
            "Treatment",
            [
                "a: Diet and physical activity",
                "b: Oral glucose lowering agents",
                "c: Insulin and oral glucose lowering agents",
                "d: Insulin",
                "e: Antihypertensive"
            ]
        )

    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    try:
        # Combine BP values for compatibility
        bp_combined = f"{systolic_bp}/{diastolic_bp}"

        # Prepare input data
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [1 if gender == "M" else 0],
            'WEIGHT(kg)': [weight],
            'BMI': [bmi],
            'BP(mmHg)': [bp_combined],
            'BLOOD SUGAR(mmol/L)': [blood_sugar],
            'DIABETES': [int(diabetes)],
            'BOTH DM+HTN': [float(both_dm_htn)],
            'TREATMENT': [treatment]
        })

        # Align columns with model if needed
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]

        # Make prediction
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5

        # Interpret probability
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = "Your predicted hypertension risk is **low**. Maintain a healthy lifestyle."
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = "Your predicted hypertension risk is **moderate**. Regular BP checks are advised."
        else:
            risk_level = "🔴 High Risk"
            message = "Your predicted hypertension risk is **high**. Seek medical review promptly."

        # ---------- DISPLAY RESULTS ----------
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Hypertensive' if pred == 1 else 'Normal'}")
        st.write(f"**Probability of Hypertension:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # ---------- SAVE PREDICTION LOG ----------
        record = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Age": [age],
            "Gender": [gender],
            "Weight": [weight],
            "BMI": [bmi],
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
            "Blood_Sugar": [blood_sugar],
            "Diabetes": [diabetes],
            "Treatment": [treatment],
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
st.caption("Developed by Millicent Chesang | Powered by AI for Public Health")
