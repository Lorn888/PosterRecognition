import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths
DATASET_PATH = "dataset"
OUTPUT_PATH = "prepared_data"

IMG_SIZE = (224, 224)

os.makedirs(OUTPUT_PATH, exist_ok=True)

X = []
y = []

poster_folders = sorted(os.listdir(DATASET_PATH))
label_map = {}

label_counter = 0

for folder in poster_folders:
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    images = [
        img for img in os.listdir(folder_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < 2:
        print(f"Skipping folder {folder} because it has less than 2 images")
        continue

    label_map[folder] = label_counter
    label_counter += 1

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        # ✅ Skip if it's a folder (e.g., thumbs)
        if os.path.isdir(img_path):
            continue

        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(label_map[folder])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# Convert to numpy arrays
X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print("Images per class:", Counter(y))

# ✅ Use one image per class for validation (if possible)
test_size = 1 / 3  # 1 image validation, 2 for training

try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
except ValueError as e:
    print("⚠️ Stratified split failed, falling back to random split:", e)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

# Save prepared data
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_PATH, "label_map.npy"), label_map)

print("✅ Dataset prepared and saved successfully!")
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")