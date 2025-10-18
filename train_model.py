import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Paths
DATA_PATH = "prepared_data"
MODEL_PATH = "poster_model.h5"

# Load prepared data
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))

num_classes = len(np.unique(y_train))
print(f"Loaded data — {len(X_train)} training, {len(X_val)} validation samples, {num_classes} classes")

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    callbacks=callbacks
)

# Unfreeze part of the base model for fine-tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2  # Unfreeze top half

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune training
fine_tune_history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")