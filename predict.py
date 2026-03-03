# predict.py
import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ----------- CONFIG ----------- #
MODEL_PATH = "model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)  # same as training

# ----------- CHECK FILES ----------- #
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    sys.exit(1)

if not os.path.exists(CLASS_NAMES_PATH):
    print(f"Error: Class names file '{CLASS_NAMES_PATH}' not found!")
    sys.exit(1)

# ----------- LOAD MODEL AND CLASS NAMES ----------- #
print(f"Loading model from '{MODEL_PATH}'...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ----------- HELPER FUNCTION ----------- #
def load_and_preprocess(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found.")
        sys.exit(1)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# ----------- MAIN ----------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    # Preprocess image
    img_array = load_and_preprocess(img_path)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"Prediction: {CLASS_NAMES[class_index]} ({confidence:.2f}% confidence)")
