import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'drowsiness_model_full.keras')

# Load model once at import
model = load_model(MODEL_PATH, compile=False)

def preprocess(frame, target_size=(224,224)):
    img = cv2.resize(frame, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(frame):
    """
    Returns (label, confidence)
    """
    img = preprocess(frame)
    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    label = ['Awake', 'Drowsy'][idx]
    confidence = float(preds[idx])
    return label, confidence
