from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import predict_letter
from utils import normalize_landmarks
import numpy as np

app = FastAPI(title="TSL Letter Prediction API")

class LandmarkRequest(BaseModel):
    landmarks: list  # list of [x, y, z] points (21 elements)

@app.post("/predict")
def predict(landmarks_req: LandmarkRequest):
    try:
        landmarks = landmarks_req.landmarks
        if len(landmarks) != 21:
            raise HTTPException(status_code=400, detail="Must provide 21 hand landmarks.")
        norm = normalize_landmarks(landmarks)
        letter = predict_letter(norm)
        return {"letter": letter}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

