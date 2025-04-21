import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("wine_price_model.pkl")

st.title("🍷 Wine Price Predictor")
st.markdown("Predict the price of wine based on its chemical properties.")

# Sidebar for inputs
st.sidebar.header("Input Wine Features")

def user_input():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.6, 0.5)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 16.0, 2.5)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05)
    free_sulfur = st.sidebar.slider("Free Sulfur Dioxide", 1, 75, 15)
    total_sulfur = st.sidebar.slider("Total Sulfur Dioxide", 6, 300, 50)
    density = st.sidebar.slider("Density", 0.9900, 1.0050, 0.9950)
    pH = st.sidebar.slider("pH", 2.8, 4.0, 3.3)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)
    quality = st.sidebar.slider("Quality (0–10)", 0, 10, 6)

    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur,
                          total_sulfur, density, pH, sulphates,
                          alcohol, quality]])
    return features

input_data = user_input()

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Wine Price: ₹{prediction:.2f}")
