import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# ---------- MODEL LOADING (DO NOT TOUCH) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tree_model.h5")

model = tf.keras.models.load_model(MODEL_PATH, compile=False,safe_mode=False)

CLASS_NAMES = [
    "Acacia", "Bamboo-shoots", "Banana-plant", "Banyan-Tree",
    "Coconut-tree", "Mango-Tree", "Mangrove", "Neem-Tree",
    "Palm-tree", "Papaya-Tree", "Pine", "lemon tree", "peepal tree"
]

# ---------- PREDICTION ----------
def predict_tree(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])

    return CLASS_NAMES[idx], confidence

