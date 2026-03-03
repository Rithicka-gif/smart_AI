import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Load Assets
model = load_model('smart_soil_crop_model.h5')
# Note: 'soil_preprocessor.pkl' is the ColumnTransformer that handles scaling + encoding
preprocessor = joblib.load('soil_preprocessor.pkl') 
le = joblib.load('crop_label_encoder.pkl')

def run_prediction(input_dict):
    # Create a raw DataFrame from the dictionary
    # Column names MUST match the CSV headers exactly
    df = pd.DataFrame([input_dict])

    # 2. Feature Engineering (Matches Training)
    def get_ph_cat(ph):
        if ph < 6.5: return 'Acidic'
        elif 6.5 <= ph <= 7.5: return 'Neutral'
        else: return 'Alkaline'

    def get_moist_lvl(m):
        if m < 35: return 'Low'
        elif 35 <= m <= 60: return 'Medium'
        else: return 'High'

    # Apply engineering to the raw columns
    df['pH_Category'] = df['Phosphorous'].apply(get_ph_cat)
    df['Moisture_Level'] = df['Moisture'].apply(get_moist_lvl)
    
    # Calculate Suitability Score
    n, p, k = df['Nitrogen'][0], df['Phosphorous'][0], df['Potassium'][0]
    score = 40 if df['pH_Category'][0] == 'Neutral' else 20
    score += 30 if df['Moisture_Level'][0] == 'Medium' else 10
    score += min((n + p + k) / 3, 30)
    df['Soil_Suitability_Score'] = score

    # 3. Transform and Predict
    # DO NOT use pd.get_dummies here. The preprocessor handles it.
    try:
        # This will scale the numbers and one-hot encode the categories automatically
        processed_data = preprocessor.transform(df)
        
        prediction = model.predict(processed_data)
        result_idx = np.argmax(prediction)
        return le.inverse_transform([result_idx])[0]
    except Exception as e:
        return f"Error during transformation: {e}"

# Example Test
if __name__ == "__main__":
    test_input = {
        'Temparature': 30.0, 
        'Humidity': 60.0, 
        'Moisture': 45.0, 
        'Soil Type': 'Loamy', 
        'Nitrogen': 40, 
        'Phosphorous': 20, 
        'Potassium': 10
    }
    result = run_prediction(test_input)
    print(f"\n🚀 Recommended Crop: {result}")
