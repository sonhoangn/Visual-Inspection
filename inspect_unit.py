import tensorflow as tf
import numpy as np
import sys
import os

# --- Configuration (MUST MATCH training script) ---
MODEL_PATH = 'defect_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Class names list - order matters!
# The model outputs a probability for the class at index 1 ('good').
CLASS_NAMES = ['defective', 'good']
PROBABILITY_THRESHOLD = 0.5  # Boundary to decide between the two classes


# 1. Image Preprocessing Function
def preprocess_image(image_path):
    """
    Loads an image, resizes it, converts it to a numpy array,
    normalizes it (0-1), and adds the batch dimension.
    """
    # Use Keras utility to load the image and resize it
    try:
        img = tf.keras.utils.load_img(
            image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Convert to numpy array
    img_array = tf.keras.utils.img_to_array(img)

    # Keras models expect a batch dimension (even for one image)
    # Shape changes from (128, 128, 3) to (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # The training model had a Rescaling(1./255) layer built-in,
    # but for standalone prediction, it's safer to ensure normalization.
    # However, since the Rescaling layer is part of the saved model,
    # rely on the saved model,
    # but the input must be a float type.

    return img_array


# 2. Main Inspection Function
def inspect_image(image_path):
    """Loads the model, preprocesses the image, and makes a prediction."""

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}...")
    # Load the entire model, including the architecture, weights, and optimizer state
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Inspecting image: {image_path}")
    image_tensor = preprocess_image(image_path)

    if image_tensor is None:
        return

    # Perform prediction (Inference)
    # The output is a single probability value between 0 and 1
    predictions = model.predict(image_tensor)

    # The model uses sigmoid output, giving the probability of the positive class ('good')
    # predictions[0][0] is the scalar probability
    probability_good = predictions[0][0]

    # 3. Interpret the result
    if probability_good >= PROBABILITY_THRESHOLD:
        prediction_class = CLASS_NAMES[1]  # 'good'
        confidence = probability_good
    else:
        prediction_class = CLASS_NAMES[0]  # 'defective'
        # Confidence in 'defective' is 1 - probability_good
        confidence = 1.0 - probability_good

    # 4. Print results
    print("-" * 30)
    print(f"Inspection Result for: {os.path.basename(image_path)}")
    print(f"Predicted Class: {prediction_class}")
    print(f"Confidence: {confidence:.2f} ({confidence * 100:.2f}%)")
    print(f"Raw Score (P('good')): {probability_good:.4f}")
    print("-" * 30)


# 5. Execution Block
if __name__ == '__main__':
    # Check if an image path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python inspect_unit.py <path_to_image>")
        print("Example: python inspect_unit.py ./test_images/sample_defect.jpg")
        sys.exit(1)

    input_image_path = sys.argv[1]

    if not os.path.exists(input_image_path):
        print(f"Error: Input image file not found at '{input_image_path}'")
        sys.exit(1)

    # Run the inspection
    inspect_image(input_image_path)