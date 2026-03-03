# ====================================================
# 🌾 CROP RECOMMENDATION WITH RANGE IN INPUT MESSAGE
# ====================================================

import numpy as np
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & DATASET
# ===============================

model = joblib.load("crop_model.pkl")
scaler = joblib.load("crop_scaler.pkl")
data = pd.read_csv("Crop_recommendation.csv")

print("✅ Model Loaded Successfully!")

# ===============================
# GET PARAMETER RANGES
# ===============================

parameters = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
ranges = {}

for param in parameters:
    min_val = data[param].min()
    max_val = data[param].max()
    ranges[param] = (min_val, max_val)

# ===============================
# FUNCTION TO TAKE SAFE INPUT
# ===============================

def get_input(param_name):
    min_val, max_val = ranges[param_name]
    
    while True:
        try:
            value = float(
                input(f"\nEnter the value of {param_name} ({min_val:.2f} to {max_val:.2f}): ")
            )
            
            if min_val <= value <= max_val:
                return value
            else:
                print("⚠ Value out of range! Please enter within allowed limits.")
                
        except ValueError:
            print("❌ Invalid input! Please enter numeric value.")

# ===============================
# TAKE USER INPUT
# ===============================

print("\n📊 Please Enter Soil & Climate Details\n")

N = get_input("N")
P = get_input("P")
K = get_input("K")
temperature = get_input("temperature")
humidity = get_input("humidity")
ph = get_input("ph")
rainfall = get_input("rainfall")

# ===============================
# PREPARE DATA FOR MODEL
# ===============================

input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
input_scaled = scaler.transform(input_data)

# ===============================
# PREDICT CROP
# ===============================

prediction = model.predict(input_scaled)
recommended_crop = prediction[0]

print("\n🌾 Recommended Crop:", recommended_crop)
