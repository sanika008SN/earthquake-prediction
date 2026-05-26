import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('earthquake_model.pkl')
region_encoder = joblib.load('region_encoder.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
risk_encoder = joblib.load('risk_encoder.pkl')

# App title
st.title("🌍 Earthquake Prediction System")

# Input fields
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
depth = st.number_input("Depth")

region = st.selectbox(
    "Region",
    ['Maharashtra','Delhi','Tamil Nadu',
     'West Bengal','Rajasthan','Goa','Telangana']
)

tremors = st.number_input("Previous Tremors", step=1)

soil = st.selectbox(
    "Soil Type",
    ['Rocky','Sandy','Clay']
)

# Predict button
if st.button("Predict"):

    # Encode input
    region_encoded = region_encoder.transform([region])[0]
    soil_encoded = soil_encoder.transform([soil])[0]

    # Prepare input
    input_data = np.array([[latitude,
                            longitude,
                            depth,
                            region_encoded,
                            tremors,
                            soil_encoded]])

    # Predict
    prediction = model.predict(input_data)

    # Convert output
    result = risk_encoder.inverse_transform(prediction)[0]

    # Show result
    st.success(f"Earthquake Risk Level: {result}")