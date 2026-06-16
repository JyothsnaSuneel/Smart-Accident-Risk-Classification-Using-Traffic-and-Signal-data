import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Accident Risk Classifier 🚦",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model_and_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(
        os.path.join(base_dir, "models", "ensemble_model.pkl")
    )

    feature_names = joblib.load(
        os.path.join(base_dir, "models", "feature_names.pkl")
    )

    encoders = joblib.load(
        os.path.join(base_dir, "models", "label_encoders.pkl")
    )

    return model, feature_names, encoders


model, feature_names, encoders = load_model_and_files()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🚦 Smart Accident Risk Classifier")
st.markdown("Predict accident risk based on traffic and signal data.")

st.markdown("---")
st.header("📋 Input Traffic & Signal Data")

# ---------------------------------------------------
# Features
# ---------------------------------------------------
boolean_features = ["has_signal", "is_peak"]

input_data = {}

# ---------------------------------------------------
# Dynamic Inputs
# ---------------------------------------------------
for feature in feature_names:

    if feature in encoders:

        input_data[feature] = st.selectbox(
            feature,
            encoders[feature].classes_
        )

    elif feature in boolean_features:

        input_data[feature] = st.checkbox(feature)

    else:

        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.2f"
        )

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🔍 Predict Accident Risk"):

    try:

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in encoders:

            if col in input_df.columns:

               input_df[col] = encoders[col].transform(input_df[col])

        # Convert booleans
        for col in input_df.columns:

            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype(int)

        # Match training order
        input_df = input_df[feature_names]

        # Debug (optional)
        with st.expander("View Encoded Input"):
            st.dataframe(input_df)

        # Predict
        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        st.markdown("### 🎯 Prediction Result")

        if prediction == 1:

            st.error(
                f"🚨 High Risk of Accident\n\nRisk Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ Low Risk of Accident\n\nRisk Probability: {probability:.2%}"
            )

        st.metric(
            "📊 Accident Probability",
            f"{probability:.2%}"
        )

        st.progress(int(probability * 100))

    except Exception as e:

        st.error("Prediction Error")
        st.exception(e)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
#### 🚀 Built by **Jyothsna Suneel**

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/jyothsna-suneel-a588b15b/)
""")



