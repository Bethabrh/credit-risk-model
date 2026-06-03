from fastapi import FastAPI
import pandas as pd
import joblib
import os

from src.data_processing import create_features
from src.api.pydantic_models import CreditRequest

app = FastAPI()

# -------------------------
# LOAD ARTIFACTS
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)


@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}


@app.post("/predict")
def predict(data: CreditRequest):

    try:
        # Convert request to DataFrame
        df = pd.DataFrame([data.dict()])

        # Feature engineering
        df = create_features(df)

        # Drop target columns if they exist
        X = df.drop(columns=["FraudResult", "is_high_risk"], errors="ignore")

        # Preprocess
        X_processed = pipeline.transform(X)

        # Prediction
        probability = model.predict_proba(X_processed)[0][1]
        prediction = model.predict(X_processed)[0]

        return {
            "high_risk_prediction": int(prediction),
            "risk_probability": float(probability)
        }

    except Exception as e:
        return {
            "error": str(e)
        }