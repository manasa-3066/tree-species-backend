import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
from openai import OpenAI

# ---------- MODEL LOADING (CRITICAL FIX) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tree_model.h5")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "Acacia", "Bamboo-shoots", "Banana-plant", "Banyan-Tree",
    "Coconut-tree", "Mango-Tree", "Mangrove", "Neem-Tree",
    "Palm-tree", "Papaya-Tree", "Pine", "lemon tree", "peepal tree"
]

def predict_tree(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])

    return CLASS_NAMES[idx], confidence


# ---------- OPENAI (SAFE INIT) ----------
load_dotenv()

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_tree_awareness(tree_name: str):
    if client is None:
        return "Awareness service not available."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert botanist."},
            {"role": "user", "content": f"""
            Explain the tree species {tree_name} for public awareness.
            Include:
            - Scientific name
            - Physical features
            - Medicinal/ecological importance
            - Environmental benefits
            - Why it should be protected
            """}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
