# streamlit_app.py

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from utils.model_loader import load_ensemble_model, load_feature_names
st.set_page_config(
    page_title="Smart Accident Risk Classifier 🚦",
    page_icon="🚦",
    layout="wide",
)
# --- Model loader ---
@st.cache_resource
def load_model_and_features():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "ensemble_model.pkl")
    feature_path = os.path.join(base_dir, "models", "feature_names.pkl")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)
    return model, feature_names

model, feature_names = load_model_and_features()

# ✅ Set page config as first Streamlit command

# --- App Title ---
st.title("🚦 Smart Accident Risk Classifier")
st.markdown("Predict accident risk based on traffic and signal data.")

st.markdown("---")
st.header("📋 Input Traffic & Signal Data")

# --- Dynamic UI Inputs ---
input_data = {}

# Define categorical dropdowns if known
categorical_options = {
    "state": ['Telangana', 'Andhra Pradesh', 'Karnataka', 'Tamil Nadu'],
    "road_type": ['urban', 'rural', 'highway'],
    "enforcement_level": ['low', 'medium', 'high'],
    "season": ['summer', 'rainy', 'winter'],
    "day_of_week": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    "lighting": ['daylight', 'night'],
    "weather": ['clear', 'rainy', 'foggy']
}

boolean_features = ["has_signal", "is_peak"]

for feature in feature_names:
    if feature in categorical_options:
        input_data[feature] = st.selectbox(f"{feature}", categorical_options[feature])
    elif feature in boolean_features:
        input_data[feature] = st.checkbox(f"{feature}")
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

# --- Prediction ---
if st.button("🔍 Predict Accident Risk"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical/boolean to match training phase
        for col in input_df.columns:
            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype(int)
            elif input_df[col].dtype == object:
                input_df[col] = pd.factorize(input_df[col])[0]

        # Reorder columns as trained
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("### 🎯 Prediction Result")
        if prediction == 1:
            st.error("🚨 **High Risk of Accident**")
        else:
            st.success("✅ **Low Risk of Accident**")

        st.metric("📊 Risk Probability", f"{probability:.2%}")
        st.progress(int(probability * 100))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.markdown("""---""")
st.markdown("""
#### 🚀 Built by **Jyothsna Suneel**  
🔗 [Connect on LinkedIn](https://www.linkedin.com/in/jyothsna-suneel-a588b15b/)  
""")





