# ============================================================
# 🌾 SMART AGRICULTURE AI SYSTEM
# Developed by Sri 💚
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import joblib
import json
import matplotlib.pyplot as plt
import base64

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="centered"
)

# ============================================================
# BACKGROUND IMAGE
# ============================================================

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        pass

set_background("backgroud_image.jpg")

# ============================================================
# LOAD MODELS (SAFE)
# ============================================================

@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model("model.keras")
    soil_model = tf.keras.models.load_model("smart_soil_crop_model.h5")
    scaler = joblib.load("soil_preprocessor.pkl")
    encoder = joblib.load("crop_label_encoder.pkl")
    return disease_model, soil_model, scaler, encoder

try:
    disease_model, soil_model, scaler, encoder = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ============================================================
# LOAD CLASS NAMES (WORKS FOR DICT OR LIST)
# ============================================================

try:
    with open("class_names.json", "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        class_names = [None] * len(data)
        for name, index in data.items():
            class_names[index] = name

    elif isinstance(data, list):
        class_names = data

    else:
        st.error("Unsupported class_names.json format")
        st.stop()

except Exception as e:
    st.error(f"Class file error: {e}")
    st.stop()

# ============================================================
# DISEASE DATABASE
# ============================================================

disease_info = {

    "Apple___Apple_scab": {
        "description": "Fungal disease causing olive-green or brown leaf spots.",
        "remedy": "Spray Mancozeb or Captan fungicide every 7–10 days.",
        "prevention": "Remove fallen leaves and ensure air circulation."
    },

    "Apple___Black_rot": {
        "description": "Dark circular lesions on fruit and leaves.",
        "remedy": "Prune infected areas and apply Thiophanate-methyl.",
        "prevention": "Maintain orchard hygiene."
    },

    "Apple___Cedar_apple_rust": {
        "description": "Yellow-orange leaf spots due to fungal infection.",
        "remedy": "Apply Myclobutanil fungicide.",
        "prevention": "Remove nearby cedar plants."
    },

    "Potato___Early_blight": {
        "description": "Brown concentric ring patterns on leaves.",
        "remedy": "Apply Chlorothalonil fungicide.",
        "prevention": "Practice crop rotation."
    },

    "Potato___Late_blight": {
        "description": "Dark water-soaked lesions spreading quickly.",
        "remedy": "Apply Metalaxyl immediately.",
        "prevention": "Avoid overwatering."
    }
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌿 Agri-Smart Menu")
option = st.sidebar.radio(
    "Select Service",
    ["Home", "Crop Disease Detection", "Soil-to-Crop Recommendation", "Climate Modelling"]
)

# ============================================================
# HOME
# ============================================================

if option == "Home":
    st.title("🌾 Smart Agriculture AI System")
    st.write("AI-powered Crop Disease Detection and Smart Farming Solutions.")

# ============================================================
# CROP DISEASE DETECTION
# ============================================================

elif option == "Crop Disease Detection":

    st.header("🔍 Crop Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = disease_model.predict(img_array)
        class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100

        if class_index >= len(class_names):
            st.error("Model class mismatch detected.")
            st.stop()

        predicted_class = class_names[class_index]

        st.success(f"🌾 Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("🧾 Disease Details")

        if "healthy" in predicted_class.lower():
            st.success("🌿 The plant appears healthy.")
            st.info("Maintain proper irrigation and fertilization.")

        elif predicted_class in disease_info:
            info = disease_info[predicted_class]
            disease_name = predicted_class.split("___")[1].replace("_", " ")

            st.write(f"### 🦠 Disease: {disease_name}")
            st.write("### 📌 Description")
            st.write(info["description"])
            st.write("### 💊 Remedy")
            st.success(info["remedy"])
            st.write("### 🌿 Prevention")
            st.info(info["prevention"])

        else:
            st.warning("Detailed remedy not available yet.")

# ============================================================
# SOIL TO CROP RECOMMENDATION
# ============================================================

elif option == "Soil-to-Crop Recommendation":

    st.header("🌱 Soil-to-Crop AI Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("Humidity (%)", 0, 100, 60)
        moisture = st.number_input("Moisture (%)", 0, 100, 40)
        nitrogen = st.number_input("Nitrogen (ppm)", 0, 150, 50)

    with col2:
        phosphorous = st.number_input("Phosphorous (ppm)", 0, 150, 50)
        potassium = st.number_input("Potassium (ppm)", 0, 250, 50)
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clayey", "Red", "Black"])
        ph = st.number_input("pH Level", 0.0, 14.0, 7.0)

    if st.button("🌾 Recommend Crop"):

        input_dict = {
            "Temparature": [temperature],
            "Humidity": [humidity],
            "Moisture": [moisture],
            "Nitrogen": [nitrogen],
            "Phosphorous": [phosphorous],
            "Potassium": [potassium],
            "Soil Type": [soil_type],
            "pH_Category": ["Neutral"],
            "Moisture_Level": ["Medium"],
            "Soil_Suitability_Score": [50]
        }

        input_df = pd.DataFrame(input_dict)
        input_scaled = scaler.transform(input_df)

        prediction = soil_model.predict(input_scaled)
        result_index = np.argmax(prediction)
        crop_name = encoder.inverse_transform([result_index])[0]

        st.success(f"🌾 Recommended Crop: {crop_name}")
        st.balloons()

# ============================================================
# CLIMATE MODELLING
# ============================================================

elif option == "Climate Modelling":

    st.header("🌍 Climate Modelling & Future Prediction")

    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("Analyze Climate"):

        future_temp = temperature + 1.5
        future_rain = rainfall * 0.9

        st.subheader("📊 2050 Projection")
        st.write(f"Future Temperature: {round(future_temp,2)}°C")
        st.write(f"Future Rainfall: {round(future_rain,2)} mm")

        fig, ax = plt.subplots()
        ax.plot(["Current", "2050"], [temperature, future_temp], marker="o")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature Trend")

        st.pyplot(fig)
