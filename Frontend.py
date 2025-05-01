import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import random

# Set page configuration
st.set_page_config(page_title="GRF Prediction", layout="centered")

# Custom CSS styling
def set_custom_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.orthoneurophysioclinic.com/wp-content/uploads/2021/07/ortho-physio.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: black;
            text-align: center;
            padding: 20px 0;
        }
        .quote {
            font-size: 22px;
            font-style: italic;
            color: #000000;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Brown color for all labels */
        label, .stTextInput label, .stNumberInput label, .stRadio label {
            color: #8B4513 !important;
            font-weight: bold !important;
        }
        /* Male/Female radio text to brown */
        .stRadio div div {
            color: #8B4513 !important;
            font-weight: bold !important;
        }
        /* Red button with white font */
        .stButton > button {
            background-color: red !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 0.5em 2em !important;
        }
        /* Input fields styled with light blue background and black border */
        input[type="text"], input[type="number"] {
            background-color: #ADD8E6 !important;
            color: black !important;
            border: 2px solid black !important;
            border-radius: 8px !important;
            padding: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown('<div class="main-title">🦵 Ground Reaction Force (GRF) Predictor by using Machine Learning Model</div>', unsafe_allow_html=True)

    # All quotes shown
    quotes = [
        "“The body achieves what the mind believes.”",
        "“Physiotherapy is not just a treatment — it's a way back to life.”",
        "“Healing is a matter of time, but it is sometimes also a matter of opportunity.”",
        "“Every step you take in therapy is one closer to strength.”",
    ]
    for q in quotes:
        st.markdown(f'<div class="quote">{q}</div>', unsafe_allow_html=True)

set_custom_background()

# Load models
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}")
        return None

linear_model_path = r"C:\Users\M Swathi\OneDrive\Desktop\Major_Project\linear_regression_model.pkl"
random_forest_model_path = r"C:\Users\M Swathi\OneDrive\Desktop\Major_Project\random_forest_model.pkl"

linear_regressor = load_model(linear_model_path)
random_forest_regressor = load_model(random_forest_model_path)

# Input fields
name = st.text_input("Enter your Name")
age = st.number_input("Enter your Age", min_value=0, max_value=100, step=1)
gender = st.radio("Select Gender", ('Male', 'Female'))
weight = st.number_input("Enter your Weight (kg)", min_value=10.0, max_value=200.0, step=0.1)
speed = st.number_input("Enter your Speed (m/s)", min_value=0.0, max_value=10.0, step=0.1)
imu_peak = st.number_input("Enter IMU Peak", min_value=0.0, max_value=100.0, step=0.1)
imu_impulse = st.number_input("Enter IMU Impulse", min_value=0.0, max_value=100.0, step=0.1)
imu_tc = st.number_input("Enter IMU tc", min_value=0.0, max_value=10.0, step=0.01)

# Predict button
if st.button("Predict GRF"):
    if linear_regressor is None or random_forest_regressor is None:
        st.error("One or both models are not loaded.")
    else:
        gender_numeric = 1 if gender == 'Male' else 0
        input_data = np.array([[speed, imu_peak, weight, imu_impulse]])

        try:
            linear_pred = linear_regressor.predict(input_data)[0]
            rf_pred = random_forest_regressor.predict(input_data)[0]

            st.markdown(f"<h4 style='color:black;'>📈 Predicted GRF Peak:</h4>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:black;'>Linear Regression Model: <b>{linear_pred:.2f}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:black;'>Random Forest Model: <b>{rf_pred:.2f}</b></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
        # Function to classify GRF
# Move classify_grf() above try block
def classify_grf(value):
    if value < 2.6:
        return "🟢 <b>Low GRF</b> – Possible muscle weakness or fatigue."
    elif value < 2.9:
        return "🟡 <b>Moderate GRF</b> – Normal for walking/jogging. Monitor for discomfort."
    elif value < 3.2:
        return "🟠 <b>High GRF</b> – Watch for knee/ankle strain. Consider physiotherapy support."
    else:
        return "🔴 <b>Very High GRF</b> – Risk of joint overload. Strongly advised to consult a physiotherapist."

try:
    linear_pred = linear_regressor.predict(input_data)[0]
    rf_pred = random_forest_regressor.predict(input_data)[0]

    

    # Show classification and suggestion in black
    st.markdown("<h4 style='color:black;'>⚕️ GRF Health Insight:</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:black;'>{classify_grf(linear_pred)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:black;'>{classify_grf(rf_pred)}</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Prediction error: {e}")




