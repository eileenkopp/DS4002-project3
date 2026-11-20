
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Model configuration
LABELS = ["rock", "paper", "scissors"]
IMG_SIZE = 224

# Locate the saved model relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "OUTPUT" / "rps_mobilenetv2.keras"


def load_classifier():
    """Load the trained CNN once when the game starts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )
    return model


def predict_gesture(model, hand_crop):
    """
    Predict rock, paper, or scissors from an OpenCV image.

    Returns:
        predicted_label: str
        confidence: float
    """

    # OpenCV uses BGR; the CNN was trained on RGB images
    rgb_image = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

    # Resize to the model's expected dimensions
    resized = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE))

    # Convert to float32 (retain pixel range 0-255)
    image_array = resized.astype(np.float32)

    # Add batch dimension: (224,224,3) -> (1,224,224,3)
    batch = np.expand_dims(image_array, axis=0)

    # Run CNN inference
    probabilities = model(batch, training=False).numpy()[0]

    # Choose class with highest probability
    predicted_index = int(np.argmax(probabilities))
    predicted_label = LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_label, confidence
