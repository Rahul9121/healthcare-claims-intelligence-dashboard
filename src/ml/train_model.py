from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "chronic_conditions_count",
    "risk_score_baseline",
    "length_of_stay",
    "prior_ed_visits_6m",
    "medication_adherence",
    "allowed_amount",
    "paid_amount",
    "followup_within_7d",
]
CATEGORICAL_FEATURES = [
    "admission_type",
    "diagnosis_group",
    "provider_specialty",
    "plan_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "readmitted_30d"


def _to_native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_training_frame(members: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    merged = claims.merge(
        members[["member_id", "age", "plan_type", "chronic_conditions_count", "risk_score_baseline"]],
        on="member_id",
        how="left",
    )
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    frame = merged[required].dropna().copy()
    frame["followup_within_7d"] = frame["followup_within_7d"].astype(int)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_model_pipeline(
    members_path: Path,
    claims_path: Path,
    model_path: Path,
    metrics_path: Path,
    sample_payload_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    members = pd.read_csv(members_path)
    claims = pd.read_csv(claims_path)
    frame = build_training_frame(members, claims)

    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_used_for_training": int(len(X_train)),
        "rows_used_for_testing": int(len(X_test)),
        "target_prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "feature_columns": FEATURE_COLUMNS,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sample_payload_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sample_payload = {
        key: _to_native(value) for key, value in X_test.head(1).to_dict(orient="records")[0].items()
    }
    sample_payload_path.write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train readmission risk model.")
    parser.add_argument(
        "--members-path", type=Path, default=Path("data/raw/members.csv"), help="Members CSV path."
    )
    parser.add_argument(
        "--claims-path", type=Path, default=Path("data/raw/claims.csv"), help="Claims CSV path."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/readmission_model.joblib"),
        help="Output model artifact path.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("models/model_metrics.json"),
        help="Output model metrics path.",
    )
    parser.add_argument(
        "--sample-payload-path",
        type=Path,
        default=Path("models/sample_prediction_payload.json"),
        help="Output path for sample API payload.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model_pipeline(
        members_path=args.members_path,
        claims_path=args.claims_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        sample_payload_path=args.sample_payload_path,
        random_state=args.random_state,
    )
    print(f"Model artifact saved: {args.model_path}")
    print(f"Metrics saved: {args.metrics_path}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
