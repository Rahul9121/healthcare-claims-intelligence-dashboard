from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    chronic_conditions_count: int = Field(ge=0, le=15)
    risk_score_baseline: float = Field(ge=0, le=1)
    length_of_stay: int = Field(ge=1, le=60)
    prior_ed_visits_6m: int = Field(ge=0, le=30)
    medication_adherence: float = Field(ge=0, le=1)
    allowed_amount: float = Field(ge=0)
    paid_amount: float = Field(ge=0)
    followup_within_7d: int = Field(ge=0, le=1)
    admission_type: Literal["Emergency", "Urgent", "Elective"]
    diagnosis_group: Literal[
        "Cardiology",
        "Pulmonology",
        "Orthopedics",
        "Oncology",
        "Endocrinology",
        "Nephrology",
    ]
    provider_specialty: Literal[
        "Internal Medicine",
        "Cardiology",
        "Pulmonology",
        "Orthopedics",
        "Hospitalist",
        "Family Medicine",
    ]
    plan_type: Literal["Bronze", "Silver", "Gold", "Platinum"]


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_bucket: Literal["low", "medium", "high"]
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[RiskPredictionRequest] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
