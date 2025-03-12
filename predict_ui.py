import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model/xgboost_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.title("💳 Credit Card Default Prediction")
st.write("Paste a comma-separated list of values to check the risk of default.")

# User input
user_input = st.text_area("Enter features (comma-separated)", "")

if st.button("Predict"):
    try:
        # Convert input to array
        input_data = np.array([list(map(float, user_input.split(',')))])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction] * 100
        
        # Display result
        if prediction == 1:
            st.error(f"⚠️ Likely to Default (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ No Default Expected (Confidence: {confidence:.2f}%)")
    
    except Exception as e:
        st.error("Invalid input format. Please check and try again.")
