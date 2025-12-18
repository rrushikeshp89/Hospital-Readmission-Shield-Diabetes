import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# --- 1. LOAD ASSETS ---
# Removed @st.cache_resource to avoid further caching errors
def load_assets():
    # USAGE CHANGE: We use the native Booster instead of XGBClassifier
    # This avoids the "estimator_type" error caused by version mismatches.
    model = xgb.Booster()
    model.load_model("xgb_readmission_model.json")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_assets()

# --- 2. UI HEADER ---
st.title("🏥 30-Day Readmission Shield")
st.markdown("""
This tool predicts the risk of a diabetic patient being readmitted within 30 days.
*Built with XGBoost & SHAP*
""")

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("Patient Profile")

# Demographics
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
race = st.sidebar.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian"])
age_group = st.sidebar.slider("Age Group (0=0-10, 9=90-100)", 0, 9, 7)

# Medical History
time_in_hospital = st.sidebar.number_input("Days in Hospital", 1, 14, 3)
num_lab_procedures = st.sidebar.number_input("Num Lab Procedures", 0, 150, 40)
num_medications = st.sidebar.number_input("Num Medications", 0, 100, 15)
service_utilization = st.sidebar.number_input("Prior Visits (Emergency/Inpatient)", 0, 20, 1)

# Diagnoses
primary_diagnosis = st.sidebar.selectbox("Primary Diagnosis Group", 
    ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 
     'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'])

# Lab Results
a1c_result = st.sidebar.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
max_glu_serum = st.sidebar.selectbox("Max Glu Serum", ["None", "Norm", ">200", ">300"])

# --- 4. PREDICTION LOGIC ---
if st.sidebar.button("Predict Risk"):
    
    # A. Create a Dictionary of Inputs
    input_data = {
        'gender': gender,
        'race': race,
        'age_group': age_group,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'service_utilization': service_utilization,
        'primary_diagnosis': primary_diagnosis,
        'A1Cresult': a1c_result,
        'max_glu_serum': max_glu_serum,
        # Defaults
        'admission_type_id': '1', 
        'discharge_disposition_id': '1',
        'admission_source_id': '7',
        'insulin': 'No',
        'med_change': 0,
        'has_diabetes_med': 1
    }
    
    # B. Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # C. One-Hot Encoding
    df_input_encoded = pd.get_dummies(df_input)
    
    # D. Align with Training Features
    df_final = df_input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # E. Predict (UPDATED FOR BOOSTER)
    # The native Booster needs a "DMatrix" wrapper
    dtest = xgb.DMatrix(df_final)
    
    # Booster returns probability directly (not 0/1 class)
    probability = model.predict(dtest)[0]
    prediction = 1 if probability > 0.5 else 0

    # --- 5. DISPLAY RESULTS ---
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error(f"⚠️ High Risk of Readmission")
            st.metric("Risk Probability", f"{probability:.1%}")
        else:
            st.success(f"✅ Low Risk")
            st.metric("Risk Probability", f"{probability:.1%}")

    with col2:
        st.info("Key Risk Factors Detected:")
        if service_utilization > 1: st.write("- High History of Hospital Use")
        if num_medications > 20: st.write("- Polypharmacy (High Meds)")
        if age_group > 7: st.write("- Advanced Age")
        if primary_diagnosis == 'Circulatory': st.write("- Circulatory Diagnosis")