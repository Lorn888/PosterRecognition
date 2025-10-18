import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "poster_model.h5"
DATASET_PATH = "dataset"  # root dataset folder

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# List all class folders
class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith(".")
])

print(f"Found {len(class_names)} classes")

# Choose an image to test
TEST_IMAGE = input("Enter path to an image to test (e.g., dataset/123/img1.jpg): ").strip()

if not os.path.exists(TEST_IMAGE):
    print("❌ File not found!")
    exit()

# Load and preprocess the image
img = image.load_img(TEST_IMAGE, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Predict
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"🎯 Predicted folder: {class_names[predicted_class]} (confidence: {confidence:.2f})")