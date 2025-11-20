import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    metadata = joblib.load("model_metadata.pkl")
    return model, metadata

model, metadata = load_model()

st.set_page_config(page_title="Wind Turbine Power Prediction", layout="wide")
st.title("Wind Turbine Power Prediction Dashboard")

st.subheader("Upload Dataset (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("Preview Uploaded Data")
    st.dataframe(df_uploaded.head())

st.subheader("Live Sensor Simulation")

col1, col2, col3 = st.columns(3)

wind_speed = col1.number_input("Wind Speed (m/s)", 0.0, 30.0, 10.0)
theoretical_power = col2.number_input("Theoretical Power (kW)", 0.0, 5000.0, 1500.0)
wind_direction = col3.number_input("Wind Direction (°)", 0, 360, 150)

generate_btn = st.button("Simulate Sensor Data")

if generate_btn:
    simulated_data = {
        "Wind Speed": wind_speed,
        "Theoretical Power": theoretical_power,
        "Wind Direction": wind_direction
    }
    st.write("Sensor Data")
    st.json(simulated_data)

st.subheader("Virtual IoT Gateway")

gateway_btn = st.button("Send Data to Virtual Gateway")

if gateway_btn:
    os.makedirs("gateway_data", exist_ok=True)
    df_gateway = pd.DataFrame([simulated_data])
    df_gateway.to_csv("gateway_data/latest_sensor.csv", index=False)
    st.write("Data sent to virtual gateway")
    st.dataframe(df_gateway)

st.subheader("Model Prediction")

predict_btn = st.button("Predict Power Output")

if predict_btn:
    arr = np.array([
        simulated_data["Wind Speed"],
        simulated_data["Theoretical Power"],
        simulated_data["Wind Direction"]
    ]).reshape(1, -1)

    prediction = model.predict(arr)[0]

    st.write(f"Predicted Active Power: {prediction:.2f} kW")

st.subheader("Model Metadata")
st.json(metadata)

importance = pd.DataFrame(metadata["feature_importance"])
st.bar_chart(importance.set_index("Feature"))

st.markdown("---")
st.write("Wind Turbine Simulation and ML Prediction System")
