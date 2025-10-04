# inspect_unit.py

import tensorflow as tf
import numpy as np
import sys
import os
import json  # <-- NEW IMPORT

# --- Configuration (MUST MATCH training script) ---
MODEL_PATH = 'defect_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['defective', 'good']
PROBABILITY_THRESHOLD = 0.5


# 1. Image Preprocessing Function
def preprocess_image(image_path):
    """
    Loads an image, resizes it, converts it to a numpy array,
    and adds the batch dimension.
    """
    # Use Keras utility to load the image and resize it
    try:
        img = tf.keras.utils.load_img(
            image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
    except Exception as e:
        # Print error and return None if loading fails
        print(f"ERROR: Image loading failed: {e}", file=sys.stderr)
        return None

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


# 2. Main Inspection Function (MODIFIED TO RETURN JSON)
def inspect_image(image_path, model_path):  # <-- Added model_path argument
    """Loads the model, preprocesses the image, and makes a prediction."""

    # Set up a dictionary to hold the result
    result = {
        "success": False,
        "message": "",
        "filename": os.path.basename(image_path),
        "prediction_class": None,
        "confidence": None,
        "raw_score_good": None,
    }

    if not os.path.exists(model_path):
        result["message"] = f"Model file not found at '{model_path}'"
        print(json.dumps(result))
        return

    try:
        # Use st.cache_resource if this function was in app.py, but here we load it fresh
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        result["message"] = f"Error loading model: {e}"
        print(json.dumps(result))
        return

    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        result["message"] = f"Failed to preprocess image at '{image_path}'"
        print(json.dumps(result))
        return

    # Perform prediction (Inference)
    predictions = model.predict(image_tensor, verbose=0)
    probability_good = float(predictions[0][0])  # Convert numpy float to standard Python float for JSON

    # 3. Interpret the result
    if probability_good >= PROBABILITY_THRESHOLD:
        prediction_class = CLASS_NAMES[1]  # 'good'
        confidence = probability_good
    else:
        prediction_class = CLASS_NAMES[0]  # 'defective'
        confidence = 1.0 - probability_good

    # 4. Populate and return results
    result["success"] = True
    result["message"] = "Inspection complete."
    result["prediction_class"] = prediction_class
    result["confidence"] = round(confidence, 4)
    result["raw_score_good"] = round(probability_good, 4)

    print(json.dumps(result))  # Print the JSON string to STDOUT


# 5. Execution Block (MODIFIED TO ACCEPT ARGS AND PASS THEM)
if __name__ == '__main__':
    # Expect 2 command-line arguments: image_path and model_path
    if len(sys.argv) < 3:
        # Fallback error message (printed to STDERR so it doesn't interfere with JSON output)
        print("Usage: python inspect_unit.py <path_to_image> <path_to_model>", file=sys.stderr)
        sys.exit(1)

    input_image_path = sys.argv[1]
    input_model_path = sys.argv[2]

    # Run the inspection
    inspect_image(input_image_path, input_model_path)