# app.py - Diabetes Prediction App (Random Forest)

import streamlit as st
import pandas as pd
import joblib
import os

# --- PASSWORD PROTECTION ---
st.set_page_config(page_title="🩺 Diabetes Prediction App", page_icon="💉")

PASSWORD = "admin123"  # 🔒 Change this password to your own

# Ask for password before showing the app
password = st.text_input("Enter password to access the app:", type="password")
if password != PASSWORD:
    st.warning("🔐 Please enter the correct password to continue.")
    st.stop()

# --- Load trained Random Forest model safely ---
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "RandomForest_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()

    model = joblib.load(model_path)
    return model


# --- Load Model ---
model = load_model()

# --- App Title ---
st.title("🩺 Diabetes Prediction App")

# --- Input Fields ---
st.header("Enter Patient Details")

Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level", min_value=0)
SystolicBP = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=0)
DiastolicBP = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0, step=1)

# --- Treatment Dropdown ---
treatment_options = [
    'ab', 'abe', 'ae', 'ade', 'e', 'ad', 'aec', 'ace', 'ce', 'ebe', 'aw', 'ac', 'a'
]
Treatment = st.selectbox("Treatment Combination", options=treatment_options)

# --- Predict Button ---
if st.button("Predict Diabetes"):
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, SystolicBP, DiastolicBP, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Treatment]],
        columns=['Pregnancies', 'Glucose', 'SystolicBP', 'DiastolicBP', 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age', 'Treatment']
    )

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ The model predicts that this patient **has diabetes**.")
        else:
            st.success("✅ The model predicts that this patient **does not have diabetes**.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
