import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('battery_model.pkl')
scaler = joblib.load('scaler.pkl')  

# App Title
st.title("🔋 EV Battery Health Prediction")
st.markdown("Predict the **State of Health (SoH)** of an EV battery using key parameters.")

# Input Form
with st.form("Battery Input Form"):
    Cycle = st.number_input("Cycle", min_value=0, max_value=1000)
    Temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=100.0)
    Voltage = st.number_input("Voltage (V)", min_value=2.0, max_value=5.0, step=0.01)
    Current = st.number_input("Current (A)", min_value=0.0, max_value=100.0)
   
    submit = st.form_submit_button("Predict SoH")

# Prediction
if submit:
    input_df = pd.DataFrame([[Cycle, Temperature, Voltage, Current]],
                        columns=['Cycle', 'Temperature', 'Voltage (V)', 'Current (A)'])
    
    input_data_scaled = scaler.transform(input_df)

    prediction = model.predict(input_data_scaled)[0]
    st.success(f"🧠 Predicted SoH: **{prediction:.2f}%**")

    # Optional Chart
    st.subheader("📉 SoH vs Cycle Trend")
    fig, ax = plt.subplots()
    ax.plot([0, Cycle], [100, prediction], marker='o')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (%)")
    ax.set_title("Battery SoH Degradation")
    st.pyplot(fig)
