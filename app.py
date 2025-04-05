import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load full pipeline (includes preprocessor + model)
model = joblib.load("car_price_pipeline.pkl")

st.set_page_config(page_title="Car Price Predictor 🚗", layout="centered")
st.title("🚗 Car Price Prediction App")

# Default input
car_name = st.text_input("Car Name (not used in prediction)", "ritz")
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2014)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, format="%.2f", value=5.59)
driven_kms = st.number_input("KMs Driven", min_value=0, value=27000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3], index=0)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'], index=0)
selling_type = st.selectbox("Selling Type", ['Dealer', 'Individual'], index=0)
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'], index=0)

if st.button("Predict Selling Price 💰"):
    input_df = pd.DataFrame([{
        'Year': year,
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Owner': owner,
        'Fuel_Type': fuel_type,
        'Selling_type': selling_type,
        'Transmission': transmission
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"💵 Estimated Selling Price: ₹ {prediction:.2f} lakhs")
