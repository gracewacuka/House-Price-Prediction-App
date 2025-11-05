# heart_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("house_price_model.h5")
#scaler = joblib.load("scaler2.pkl")

# Streamlit UI
st.title("🏠 House Price Prediction App")
st.write("Enter the details below to estimate the house price.")

# User input
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=2000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
parking = st.number_input("Parking", min_value=0, max_value=100, value=10)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, parking]])
    #input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")
