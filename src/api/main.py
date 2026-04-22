from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    RiskPredictionRequest,
)
from src.ml.inference import load_model, predict_readmission_risk

APP_VERSION = "1.0.0"
_model = None

app = FastAPI(
    title="Healthcare Readmission Risk API",
    description="Predicts 30-day readmission risk from claims and member features.",
    version=APP_VERSION,
)


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.on_event("startup")
def startup_event() -> None:
    try:
        get_model()
        app.state.model_loaded = True
    except FileNotFoundError:
        app.state.model_loaded = False


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Healthcare Readmission Risk API is running."}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "version": APP_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: RiskPredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        prediction = predict_readmission_risk(model, payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(model_version=APP_VERSION, **prediction)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        model = get_model()
        predictions = [
            PredictionResponse(
                model_version=APP_VERSION,
                **predict_readmission_risk(model, record.model_dump()),
            )
            for record in payload.records
        ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BatchPredictionResponse(predictions=predictions)
