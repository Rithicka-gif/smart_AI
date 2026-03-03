import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Dataset
data = pd.read_csv("data_core.csv") 

# 2. Feature Engineering (Adjusted for your specific headers)
def moisture_level(moisture):
    if moisture < 35: return 'Low'
    elif moisture <= 60: return 'Medium'
    else: return 'High'

def soil_score(row):
    score = 0
    # Adjusted score logic since pH is missing
    score += 30 if row['Moisture_Level'] == "Medium" else 10
    # Nutrient contribution
    nutrient_sum = row['Nitrogen'] + row['Phosphorous'] + row['Potassium']
    score += min(nutrient_sum / 3, 30)
    return score

# Apply updated logic
data['Moisture_Level'] = data['Moisture'].apply(moisture_level)
data['Soil_Suitability_Score'] = data.apply(soil_score, axis=1)

# 3. Prepare Features & Target
# Note: 'Temparature' matches the typo in your header
numeric_features = ['Temparature','Humidity','Moisture','Nitrogen','Phosphorous','Potassium','Soil_Suitability_Score']
categorical_features = ['Soil Type','Moisture_Level']

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
        ("scaler", StandardScaler(), numeric_features)
    ]
)

X = preprocessor.fit_transform(data)
y = data['Crop Type'] # Matches your header

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save transformation tools
joblib.dump(preprocessor, "soil_preprocessor.pkl")
joblib.dump(le, "crop_label_encoder.pkl")

# 4. Build Neural Network
num_classes = len(le.classes_)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(X, y_encoded, epochs=10, batch_size=4, validation_split=0.1)

# 6. Save Model
model.save("smart_soil_crop_model.h5")
print("✅ Training Complete! Model saved as 'smart_soil_crop_model.h5'")
