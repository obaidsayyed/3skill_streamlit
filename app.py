import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(page_title="Heart Failure Readmission Rate",page_icon="🫀",layout="wide")



# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("logistic_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Put it in the same folder as app.py.")
    st.stop()


# Get the feature names used during training
if not hasattr(model, "feature_names_in_"):
    st.error("The model does not contain the original feature names.")
    st.stop()

# Binary input helper
def yes_no(label, key):
    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key=key
    )


st.title("🫀 Heart Failure Readmission Prediction for 3 Skill")
st.write("Enter patient details to predict 30-day readmission.")

with st.form("patient_form"):

    # 1. Demographics and lifestyle
    st.subheader("Demographics & Lifestyle")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=0, max_value=120,
                              value=45, step=1)
        bmi = st.number_input("BMI", min_value=0.0,
                              value=25.56, step=0.1)
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        exercise = st.number_input("Exercise Frequency",
                                   min_value=0, value=0, step=1)

    with c2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])

    # 2. Medical history
    st.divider()
    st.subheader("Medical History")
    c1, c2 = st.columns(2)

    with c1:
        hypertension = yes_no("Hypertension", "hypertension")
        diabetes = yes_no("Diabetes", "diabetes")
        ckd = yes_no("Chronic Kidney Disease", "ckd")
        previous_hf = st.number_input(
            "Previous HF Admissions", min_value=0, value=1, step=1
        )
        previous_hospital = st.number_input(
            "Previous Hospital Admissions", min_value=0, value=1, step=1
        )

    with c2:
        cad = yes_no("Coronary Artery Disease", "cad")
        stroke = yes_no("Previous Stroke", "stroke")
        af = yes_no("Atrial Fibrillation", "af")
        hf_type = st.selectbox(
            "Heart Failure Type",
            ["HFmrEF", "HFrEF", "HFpEF"]
        )
        nyha = st.number_input(
            "NYHA Class", min_value=1, max_value=4, value=2, step=1
        )

    # 3. Clinical measurements
    st.divider()
    st.subheader("Clinical Measurements")
    c1, c2 = st.columns(2)

    with c1:
        ef = st.number_input("Ejection Fraction (%)",
                             min_value=0.0, max_value=100.0,
                             value=42.75, step=0.5)
        sbp = st.number_input("Systolic BP",
                              min_value=0.0, value=119.47, step=1.0)
        dbp = st.number_input("Diastolic BP",
                              min_value=0.0, value=85.55, step=1.0)
        heart_rate = st.number_input("Heart Rate",
                                     min_value=0.0, value=79.42, step=1.0)
        oxygen = st.number_input("Oxygen Saturation (%)",
                                 min_value=0.0, max_value=100.0,
                                 value=94.86, step=0.5)

    with c2:
        creatinine = st.number_input("Creatinine",
                                     min_value=0.0, value=3.05, step=0.1)
        sodium = st.number_input("Sodium",
                                 min_value=0.0, value=137.54, step=0.5)
        potassium = st.number_input("Potassium",
                                    min_value=0.0, value=4.50, step=0.1)
        hemoglobin = st.number_input("Hemoglobin",
                                     min_value=0.0, value=11.84, step=0.1)
        glucose = st.number_input("Blood Glucose",
                                  min_value=0.0, value=95.59, step=1.0)
        bnp = st.number_input("BNP",
                              min_value=0.0, value=118.41, step=1.0)

    # 4. Admission details and treatment
    st.divider()
    st.subheader("Admission & Treatment")
    c1, c2 = st.columns(2)

    with c1:
        los = st.number_input("Length of Stay (days)",
                              min_value=0, value=13, step=1)
        icu = yes_no("ICU Admission", "icu")
        emergency = yes_no("Emergency Admission", "emergency")
        beta_blocker = yes_no("Beta Blocker", "beta_blocker")

    with c2:
        ace_arb = yes_no("ACE/ARB", "ace_arb")
        diuretic = yes_no("Diuretic", "diuretic")
        sglt2 = yes_no("SGLT2 Inhibitor", "sglt2")

    submitted = st.form_submit_button(
        "Predict 30 Days Readmission",
        use_container_width= True
    )


# Prediction
if submitted:

    # Create input using the original, unencoded column names
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Smoking_Status": smoking,
        "Alcohol_Consumption": alcohol,
        "Exercise_Frequency": exercise,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Chronic_Kidney_Disease": ckd,
        "Coronary_Artery_Disease": cad,
        "Previous_Stroke": stroke,
        "Atrial_Fibrillation": af,
        "Previous_HF_Admissions": previous_hf,
        "Previous_Hospital_Admissions": previous_hospital,
        "Heart_Failure_Type": hf_type,
        "NYHA_Class": nyha,
        "Ejection_Fraction": ef,
        "Systolic_BP": sbp,
        "Diastolic_BP": dbp,
        "Heart_Rate": heart_rate,
        "Oxygen_Saturation": oxygen,
        "Creatinine": creatinine,
        "Sodium": sodium,
        "Potassium": potassium,
        "Hemoglobin": hemoglobin,
        "Blood_Glucose": glucose,
        "BNP": bnp,
        "Length_of_Stay": los,
        "ICU_Admission": icu,
        "Emergency_Admission": emergency,
        "Beta_Blocker": beta_blocker,
        "ACE_ARB": ace_arb,
        "Diuretic": diuretic,
        "SGLT2_Inhibitor": sglt2
    }])

    try:
        # Apply the same encoding as the training code
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Match the exact feature order used during training
        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Predict
        prediction = model.predict(input_encoded)[0]

        # Probability of class 1
        positive_index = list(model.classes_).index(1)
        probability = model.predict_proba(input_encoded)[0][positive_index]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Predicted outcome: Readmitted within 30 days")
        else:
            st.success("Predicted outcome: Not readmitted within 30 days")

        st.metric(
            "Model-estimated probability of readmission",
            f"{probability * 100:.2f}%"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")