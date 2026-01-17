from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_tree
from awareness import get_tree_awareness
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
    image_path = f"temp_{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tree_name, confidence = predict_tree(image_path)
    awareness = get_tree_awareness(tree_name)

    os.remove(image_path)

    return {
        "tree": tree_name,
        "confidence": round(confidence *100, 2),
        "awareness": awareness
    }
