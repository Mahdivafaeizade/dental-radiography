"""FastAPI service for the dental X-ray multi-label classifier."""

import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.data_loader import CLASSES
from src.dataset import EVAL_TRANSFORMS
from src.train import build_model, DEVICE

app = FastAPI(title="Dental X-ray Classifier API")

model = build_model()
model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
model.eval()


class PredictionResponse(BaseModel):
    predictions: dict[str, float]


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = EVAL_TRANSFORMS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().tolist()

    return PredictionResponse(predictions=dict(zip(CLASSES, probs)))


@app.get("/")
def root():
    return {"message": "Dental X-ray Classifier API. POST an image to /predict."}