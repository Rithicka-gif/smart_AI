import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# =========================
# 1️⃣ Load Datasets
# =========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/valid",
    image_size=(224, 224),
    batch_size=32
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32
)
# =========================
# 2️⃣ Save Class Names
# =========================

class_names = train_ds.class_names

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

num_classes = len(class_names)
print("Number of classes:", num_classes)

# =========================
# 3️⃣ Normalize Data
# =========================

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Improve performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# 4️⃣ Build CNN Model
# =========================

model = keras.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

# =========================
# 5️⃣ Compile Model
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 6️⃣ Train Model
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# =========================
# 7️⃣ Evaluate on Test Set
# =========================

test_loss, test_acc = model.evaluate(test_ds)
print("\nTest Accuracy:", test_acc)

# =========================
# 8️⃣ Save Model
# =========================

model.save("model.keras")

print("\n✅ Training Complete. Model Saved Successfully.")
