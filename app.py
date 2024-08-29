import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('my_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the feature names (replace with actual feature names)
feature_names = ['anchor_ratio', 'trans_range', 'node_density', 'iterations']

# Streamlit app
st.title("Average Localization Error (ALE) Prediction")
st.write("Input the relevant features to predict the ALE:")

# Create input fields for user inputs
st.header("Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input data to DataFrame for processing
input_df = pd.DataFrame([input_data])

# Ensure the DataFrame has the correct columns and order
input_df = input_df[feature_names]

# Button to predict ALE
if st.button("Predict ALE"):
    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the ALE
    prediction = model.predict(input_scaled)
    predicted_ale = prediction[0][0]  # Since the model predicts a single output

    # Display the result
    st.subheader("Prediction")
    st.markdown(
        f"""
    <div style="background-color: #121212; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: #4CAF50; font-family: Arial, sans-serif;">Prediction Result</h3>
        <p style="font-size: 20px; color: #ffffff; font-family: Arial, sans-serif;">The predicted Average Localization Error (ALE) is:</p>
        <h1 style="color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">{predicted_ale:.4f}</h1>
    </div>
    """,
        unsafe_allow_html=True
    )
