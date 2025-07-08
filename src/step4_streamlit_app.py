# src/step4_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model/insurance_model.pkl", "rb"))

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter the following details to predict insurance cost:")

# Input fields
age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input("BMI (e.g. 24.5)", min_value=10.0, max_value=50.0, value=24.5)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))

# Convert categorical inputs to numeric
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_map = {"southwest": 3, "southeast": 2, "northwest": 1, "northeast": 0}
region_encoded = region_map[region]

# Make prediction
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Insurance Cost: ₹{prediction:.2f}")

