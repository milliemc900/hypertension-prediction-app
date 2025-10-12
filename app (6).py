# hypertension_app.py
# 🩺 Hypertension Risk Prediction App (with Password Access)

import streamlit as st
import pandas as pd
import pickle # Changed from joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Hypertension Risk Prediction", page_icon="🫀", layout="wide")

# ---------- PASSWORD PROTECTION ----------
PASSWORD = "Millicent123"  # 🔐 Set your password here

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login screen
if not st.session_state.authenticated:
    st.title("🔒 Secure Login")
    password_input = st.text_input("Enter Password to Access the App:", type="password")
    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access Granted! Welcome.")
            st.rerun()
        else:
            st.error("❌ Incorrect Password. Please try again.")
    st.stop()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "RandomForest_model.pkl"  # Corrected filename and extension

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please upload it to your repository.")
        st.stop()

    with open(model_path, 'rb') as f:
        model = pickle.load(f) # Changed to pickle.load

    return model

model = load_model()

# ---------- HEADER ----------
st.title("🩺 Hypertension Risk Prediction System")
st.markdown("""
This AI-powered system predicts the **risk of hypertension** based on key patient indicators.
It helps clinicians identify at-risk patients for early intervention.
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


    # --- Column 3 ---
    with col3:
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=9.5)
        diabetes = st.selectbox("Diabetes (1=Yes, 0=No)", [0, 1])


    submitted = st.form_submit_button("🔍 Predict Hypertension Risk")

# ---------- PREDICTION ----------
if submitted:
    try:
        bp_combined = f"{systolic_bp}/{diastolic_bp}"

        # Prepare input for prediction
        # Removed 'BOTH DM+HTN' and 'TREATMENT'
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [1 if gender == "M" else 0],
            'WEIGHT(kg)': [weight],
            'BMI': [bmi],
            'BP(mmHg)': [bp_combined],
            'BLOOD SUGAR(mmol/L)': [blood_sugar],
            'DIABETES': [int(diabetes)],
            # 'BOTH DM+HTN': [float(both_dm_htn)], # Removed
            # 'TREATMENT': [treatment], # Removed
        })

        # Match columns to model - This part needs to be more robust after removing features.
        # It's better to recreate the processing pipeline here.
        # However, for a quick fix, we can just ensure the necessary columns are present.
        # A more complete solution would involve saving the preprocessing steps (like one-hot encoding categories)
        # and applying them here.

        # For now, we'll assume the model can handle missing columns or we manually add them with default values (like 0)
        # based on the expected features from the training data.

        # **Important Note:** This is a simplified approach. A production-ready app should save and reuse the exact
        # preprocessing steps from training.

        # Recreate dummy variables for GENDER and split BP
        input_data['SYSTOLIC BP'] = pd.to_numeric(input_data['BP(mmHg)'].str.split('/', expand=True)[0], errors='coerce')
        input_data['DIASTOLIC BP'] = pd.to_numeric(input_data['BP(mmHg)'].str.split('/', expand=True)[1], errors='coerce')
        input_data = input_data.drop('BP(mmHg)', axis=1)

        input_data = pd.get_dummies(input_data, columns=['GENDER'], drop_first=True)


        # Ensure all expected model features are present, adding missing ones with 0
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            # Ensure column order matches training data
            input_data = input_data[model_features]
        else:
            st.warning("Model does not have 'feature_names_in_'. Column order and presence might be an issue.")


        # Make prediction
        # Need to handle potential NaN after preprocessing/imputation if not done
        # Assuming the loaded model can handle NaNs or preprocessing in training included imputation
        # A more robust approach would be to save the imputer and apply it here.


        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5

        # Interpret risk
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = "Your predicted hypertension risk is **low**. Maintain healthy habits."
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = "Your predicted hypertension risk is **moderate**. Regular BP checks advised."
        else:
            risk_level = "🔴 High Risk"
            message = "Your predicted hypertension risk is **high**. Consult a healthcare provider."

        # ---------- DISPLAY RESULTS ----------
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Hypertensive' if pred == 1 else 'Normal'}")
        st.write(f"**Probability of Hypertension:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # ---------- SAVE PREDICTION LOG ----------
        # Removed 'BOTH DM+HTN' and 'TREATMENT' from logging
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
            # "Treatment": [treatment], # Removed
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
st.caption("Developed by Millicent Chesang | AI & Data Analytics for Public Health")
