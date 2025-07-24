"""
Simple FastAPI application for penguin species prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Input model
class PenguinData(BaseModel):
    bill_length: float
    bill_depth: float  
    flipper_length: float
    body_mass: float
    sex: str  # "male" or "female"
    island: str  # "Torgersen", "Biscoe", or "Dream"

# Response model
class PredictionResult(BaseModel):
    species: str
    probability: float

# Initialize FastAPI
app = FastAPI(title="Penguin Predictor", version="1.0")

# Global variables
model = None
species_names = ["Adelie", "Chinstrap", "Gentoo"]

def load_or_create_model():
    """Load existing model or create a simple one."""
    global model
    
    model_path = "penguin_model.json"
    
    if os.path.exists(model_path):
        # Load existing model
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        # Create simple model with dummy data
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=200, n_features=6, n_classes=3, random_state=42)
        
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X, y)
        model.save_model(model_path)

def preprocess_input(data: PenguinData) -> np.ndarray:
    """Convert input to model format."""
    # Create feature array
    features = [
        data.bill_length,
        data.bill_depth, 
        data.flipper_length,
        data.body_mass,
        1 if data.sex.lower() == "male" else 0,  # sex encoding
        1 if data.island == "Biscoe" else 0      # island encoding (simplified)
    ]
    
    return np.array([features])

# Load model on startup
@app.on_event("startup")
async def startup():
    load_or_create_model()

# Endpoints
@app.get("/")
def home():
    return {"message": "Penguin Species Predictor", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResult)
def predict(data: PenguinData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        features = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get result
        species = species_names[prediction]
        confidence = float(max(probabilities))
        
        return PredictionResult(species=species, probability=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/species")
def get_species():
    return {"available_species": species_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
