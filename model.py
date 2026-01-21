import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
from openai import OpenAI

# ---------- MODEL LOADING ----------
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
    confidence = round(float(pred[0][idx]) * 100, 2)

    return CLASS_NAMES[idx], confidence


# ---------- OPENAI ----------
load_dotenv()

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_tree_awareness(tree_name: str):
    if client is None:
        return (
         f"{tree_name} is a valuable and ecologically significant tree species. "
        "Trees like this play a crucial role in maintaining environmental balance "
        "by absorbing carbon dioxide, releasing oxygen, and improving air quality.\n\n"

        "They support biodiversity by providing food and shelter to birds, insects, "
        "and other living organisms, helping sustain healthy ecosystems. "
        "Many tree species also contribute to soil conservation by preventing erosion "
        "and maintaining soil fertility.\n\n"

        "Beyond environmental benefits, such trees often hold cultural, medicinal, "
        "and social importance in local communities. They have been traditionally used "
        "for shade, healing practices, and as natural resources that support livelihoods.\n\n"

        "Protecting and preserving trees like this is essential for combating climate change, "
        "supporting biodiversity, and ensuring a healthier and more sustainable planet "
        "for present and future generations."
         )


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

