import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")


# -------------------------------
# Page config
# -------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction")
st.write("Enter the patient health details and predict the heart disease class.")


# -------------------------------
# Load model & scaler
# -------------------------------

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model("heart-disease-ann.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()


# -------------------------------
# Input fields
# -------------------------------

st.subheader("Patient Health Details")

input_data = {

    'age': st.number_input(
        'Age',
        min_value=1,
        max_value=120,
        value=55
    ),

    'sex': st.number_input(
        'Sex (0 = Female, 1 = Male)',
        min_value=0,
        max_value=1,
        value=1
    ),

    'cp': st.number_input(
        'Chest Pain Type (0-3)',
        min_value=0,
        max_value=3,
        value=1
    ),

    'trestbps': st.number_input(
        'Resting Blood Pressure',
        min_value=50,
        max_value=250,
        value=140
    ),

    'chol': st.number_input(
        'Cholesterol',
        min_value=50,
        max_value=600,
        value=250
    ),

    'fbs': st.number_input(
        'Fasting Blood Sugar (0 = No, 1 = Yes)',
        min_value=0,
        max_value=1,
        value=0
    ),

    'restecg': st.number_input(
        'Resting ECG (0-2)',
        min_value=0,
        max_value=2,
        value=1
    ),

    'thalach': st.number_input(
        'Maximum Heart Rate',
        min_value=50,
        max_value=250,
        value=150
    ),

    'exang': st.number_input(
        'Exercise Induced Angina (0 = No, 1 = Yes)',
        min_value=0,
        max_value=1,
        value=0
    ),

    'oldpeak': st.number_input(
        'ST Depression',
        min_value=0.0,
        max_value=10.0,
        value=1.0
    ),

    'slope': st.number_input(
        'Slope (0-2)',
        min_value=0,
        max_value=2,
        value=1
    ),

    'ca': st.number_input(
        'Number of Major Vessels (0-4)',
        min_value=0,
        max_value=4,
        value=0
    ),

    'thal': st.number_input(
        'Thal (0-3)',
        min_value=0,
        max_value=3,
        value=2
    )
}


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([input_data])


    # Safety check
    if list(input_df.columns) != list(scaler.feature_names_in_):

        st.error("Feature mismatch between input and trained scaler.")

        st.stop()


    # Scale input data
    input_scaled = scaler.transform(input_df)


    # Make prediction
    prediction = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # Convert probability to class
    predicted_class = (
        "Heart Disease"
        if prediction > 0.5
        else "No Heart Disease"
    )


    # Display result
    st.subheader("Result")

    st.write(
        f"**Prediction:** {predicted_class}"
    )

    st.write(
        f"**Probability:** {prediction:.4f}"
    )