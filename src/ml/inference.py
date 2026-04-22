from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "chronic_conditions_count",
    "risk_score_baseline",
    "length_of_stay",
    "prior_ed_visits_6m",
    "medication_adherence",
    "allowed_amount",
    "paid_amount",
    "followup_within_7d",
    "admission_type",
    "diagnosis_group",
    "provider_specialty",
    "plan_type",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "readmission_model.joblib"


def load_model(model_path: str | Path | None = None) -> Any:
    resolved = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not resolved.exists():
        raise FileNotFoundError(f"Model artifact not found at {resolved}")
    return joblib.load(resolved)


def _risk_bucket(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


def predict_readmission_risk(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Missing required fields for prediction: {missing}")

    ordered_payload = {column: payload[column] for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([ordered_payload])
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "risk_probability": round(probability, 4),
        "risk_bucket": _risk_bucket(probability),
    }
