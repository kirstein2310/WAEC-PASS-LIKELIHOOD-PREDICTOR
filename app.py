
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="WAEC Pass Likelihood Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
MODEL_PATH = Path("/content/waec_model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please save waec_model.pkl first.")
    st.stop()

model = load_model()

# Application heading
st.title("🎓 WAEC PASS LIKELIHOOD PREDICTION SYSTEM")

st.markdown("""
### Student Academic Performance Assessment

This application predicts a student's likelihood of passing
the West African Examinations Council (WAEC) examination
based on study habits, attendance and mock examination performance.
""")

st.divider()

# Sidebar
st.sidebar.header("About the System")
st.sidebar.info("""
This system uses a trained Random Forest Classifier
to estimate WAEC pass likelihood.

Enter the student's academic information to generate
a prediction.
""")

# Student information
st.header("📋 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Weekly Study Hours",
        min_value=2.0,
        max_value=40.0,
        value=15.0,
        step=1.0
    )

with col2:
    attendance = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

with col3:
    mock_score = st.number_input(
        "Mock Examination Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

# Feature engineering
study_attendance_interaction = study_hours * attendance
academic_risk_flag = int(mock_score < 50)

# Prepare input data
input_data = pd.DataFrame([{
    "Study_Hours_Weekly": study_hours,
    "Attendance_Rate": attendance,
    "Mock_Exam_Score": mock_score,
    "Study_Attendance_Interaction": study_attendance_interaction,
    "Academic_Risk_Flag": academic_risk_flag
}])

st.divider()

# Prediction button
st.header("📊 WAEC Prediction")

if st.button("Predict Pass Likelihood", type="primary",
             use_container_width=True):

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        pass_index = list(model.classes_).index(1)
        pass_probability = probabilities[pass_index]

        fail_probability = 1 - pass_probability

        # Display results
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Pass Probability",
                f"{pass_probability:.2%}"
            )

        with result_col2:
            st.metric(
                "Failure Probability",
                f"{fail_probability:.2%}"
            )

        st.progress(float(pass_probability))

        if prediction == 1:
            st.success(
                "SUCCESS: The model predicts that this student "
                "is likely to PASS WAEC."
            )
        else:
            st.error(
                "WARNING: The model predicts that this student "
                "may be at risk of FAILING WAEC."
            )

        # Academic risk assessment
        st.subheader("Academic Risk Assessment")

        if academic_risk_flag == 1:
            st.warning(
                "HIGH ACADEMIC RISK: Mock examination score "
                "is below 50%. Additional academic support "
                "is recommended."
            )
        else:
            st.success(
                "Mock examination score is at or above "
                "the 50% risk threshold."
            )

        # Display student information
        st.subheader("Student Performance Summary")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.caption(
            "Disclaimer: This prediction is an experimental estimate "
            "from a model trained on synthetic data. It is not a "
            "guarantee of actual WAEC examination performance."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.markdown(
    "<center>WAEC Pass Likelihood Prediction System | "
    "Machine Learning Project</center>",
    unsafe_allow_html=True
)
