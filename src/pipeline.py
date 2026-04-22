from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.build_dashboard_assets import build_dashboard_assets
from src.data.generate_synthetic_data import write_synthetic_datasets
from src.data.validate_data import run_validations
from src.ml.train_model import train_model_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full healthcare intelligence pipeline.")
    parser.add_argument("--members", type=int, default=3000, help="Number of synthetic members.")
    parser.add_argument("--claims", type=int, default=18000, help="Number of synthetic claims.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    validation_report_path = Path("data/validation/validation_report.json")
    model_path = Path("models/readmission_model.joblib")
    metrics_path = Path("models/model_metrics.json")
    sample_payload_path = Path("models/sample_prediction_payload.json")

    members_path, claims_path = write_synthetic_datasets(
        members_count=args.members,
        claims_count=args.claims,
        seed=args.seed,
        out_dir=raw_dir,
    )
    print("Step 1/4: synthetic datasets generated.")

    members_df = pd.read_csv(members_path)
    claims_df = pd.read_csv(claims_path)
    validation_report = run_validations(members_df, claims_df)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    if validation_report["summary"]["status"] != "PASS":
        raise RuntimeError("Validation failed. Check data/validation/validation_report.json.")
    print("Step 2/4: data validation passed.")

    metrics = train_model_pipeline(
        members_path=members_path,
        claims_path=claims_path,
        model_path=model_path,
        metrics_path=metrics_path,
        sample_payload_path=sample_payload_path,
        random_state=args.seed,
    )
    print(f"Step 3/4: model trained. ROC-AUC={metrics['roc_auc']:.4f}")

    build_dashboard_assets(
        members_path=members_path,
        claims_path=claims_path,
        output_dir=processed_dir,
    )
    print("Step 4/4: dashboard datasets generated.")


if __name__ == "__main__":
    main()
