import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import os
import json

# Add parent directory to path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import prepare_data

# Load model and preprocessing artifacts
model = joblib.load('deployment/spaceship_model.pkl')
scaler = joblib.load('deployment/scaler.pkl')
with open('deployment/feature_names.json', 'r') as f:
    expected_features = json.load(f)

st.title("🚀 Spaceship Titanic Predictor")
st.write("Will you be transported to another dimension?")

# Input form
with st.form("passenger_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        home_planet = st.selectbox("Home Planet", ["Earth", "Mars", "Europa"])
        cryo_sleep = st.checkbox("CryoSleep")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        vip = st.checkbox("VIP")
        
    with col2:
        cabin_deck = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
        cabin_side = st.selectbox("Cabin Side", ["P", "S"])
        destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        
    st.subheader("Spending on Amenities ($)")
    room_service = st.number_input("RoomService", min_value=0.0, value=0.0)
    food_court = st.number_input("FoodCourt", min_value=0.0, value=0.0)
    shopping_mall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
    spa = st.number_input("Spa", min_value=0.0, value=0.0)
    vr_deck = st.number_input("VRDeck", min_value=0.0, value=0.0)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create dataframe
    input_data = pd.DataFrame({
        'HomePlanet': [home_planet],
        'CryoSleep': [cryo_sleep],
        'Cabin': [f"{cabin_deck}/0/{cabin_side}"],
        'Destination': [destination],
        'Age': [age],
        'VIP': [vip],
        'RoomService': [room_service],
        'FoodCourt': [food_court],
        'ShoppingMall': [shopping_mall],
        'Spa': [spa],
        'VRDeck': [vr_deck],
        'PassengerId': ['0000_01'],  # Dummy
        'Name': ['Test Passenger']  # Dummy
    })
    
    # Preprocess (will need train_stats - load or pass empty dict)
    train_stats = {
        'age_median': 27.0, 
        'spending_medians': {
            'RoomService': 0.0,
            'FoodCourt': 0.0,
            'ShoppingMall': 0.0,
            'Spa': 0.0,
            'VRDeck': 0.0
        },
        'cryo_dist': pd.Series(index=[False, True], data=[0.641694, 0.358306])
    }
    
    processed = prepare_data(input_data, train_stats=train_stats, is_train=False)

    # One-hot encode categorical features (same as training)
    categorical_features = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_Side", "Age_Group"]
    processed_encoded = pd.get_dummies(processed, columns=categorical_features, drop_first=True, dtype=int)

    # Convert binary columns to int
    processed_encoded["CryoSleep"] = processed_encoded["CryoSleep"].astype(int)
    processed_encoded["VIP"] = processed_encoded["VIP"].astype(int)

    # Ensure all expected features are present and in the correct order
    for feature in expected_features:
        if feature not in processed_encoded.columns:
            processed_encoded[feature] = 0
    processed_encoded = processed_encoded[expected_features]

    # Scale numerical features
    numerical_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Total_Spendings", "Group_Size"]
    processed_scaled = processed_encoded.copy()
    processed_scaled[numerical_cols] = scaler.transform(processed_encoded[numerical_cols])

    # Make prediction
    prediction = model.predict(processed_scaled)
    proba = model.predict_proba(processed_scaled)
    
    # Display result
    if prediction[0] == 1:
        st.success(f"🌀 TRANSPORTED! ({proba[0][1]:.1%} confidence)")
        st.write("You'll be sent to another dimension!")
    else:
        st.info(f"✅ NOT TRANSPORTED ({proba[0][0]:.1%} confidence)")
        st.write("You'll stay in this dimension!")