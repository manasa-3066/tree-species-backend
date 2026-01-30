from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_tree
from awareness import (
    get_tree_awareness_english,
    get_tree_awareness_telugu
)

import shutil
import os

app = FastAPI()

# Allow Flutter to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Tree API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ✅ always use a safe temp filename
    image_path = "temp_image.jpg"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tree_name, confidence = predict_tree(image_path)

    # awareness (offline / fallback safe)
    awareness_english = get_tree_awareness_english(tree_name)
    awareness_telugu = get_tree_awareness_telugu(tree_name)

    os.remove(image_path)

    return {
        "tree": tree_name,
        "confidence": round(confidence * 100, 2),
        "awareness_english": awareness_english,
        "awareness_telugu": awareness_telugu
    }

