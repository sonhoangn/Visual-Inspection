# train_model.py

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandomFlip, RandomRotation
import os
import sys

# --- Configuration ---
# *** CRITICAL MODIFICATION: Read paths from environment variables ***
# If run outside Streamlit, use a default path, otherwise use the path provided by the app.py
DATA_DIR = os.getenv('DATA_DIR', 'default/data/train')  # Fallback if not set
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'default_model.keras')
# Convert environment variable string back to integer
try:
    EPOCHS = int(os.getenv('EPOCHS', 10))
except ValueError:
    EPOCHS = 10  # Default fallback

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
# ------------------------------------------------------------------

print("--- Running Model Training Script ---")
print(f"Data Source: {DATA_DIR}")
print(f"Epochs: {EPOCHS}")
print(f"Save Path: {MODEL_SAVE_PATH}")

# 1. Load Data using the modern tf.keras.utils method
# Make sure the data directory exists before trying to load
if not os.path.isdir(DATA_DIR):
    print(f"FATAL ERROR: Data directory not found at {DATA_DIR}")
    # Exit the script with an error code
    sys.exit(1)
# splitting and labeling
print("Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get the class names (should be ['defective', 'good'] or similar)
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected Classes: {class_names}")

# 2. Define Data Preprocessing and Augmentation Layers
# Note: Augmentation is applied only to the training data implicitly during fit()
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
])

# Rescaling layer (convert pixel values from 0-255 to 0-1)
rescale = Rescaling(1. / 255)

# 3. Create the Model Architecture
model = Sequential([
    # Input layer handles resizing and rescaling (if not done in tf.data pipeline)
    # Preprocessing layers are placed first
    rescale,
    data_augmentation,

    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    # Final output layer: sigmoid for binary classification (2 classes)
    Dense(1, activation='sigmoid')
])

# 4. Compile the Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 6. Save the Model
print(f"\nTraining complete. Saving model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("Model successfully saved.")