from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

VALID_ADMISSION_TYPES = {"Emergency", "Urgent", "Elective"}
VALID_DIAGNOSIS_GROUPS = {
    "Cardiology",
    "Pulmonology",
    "Orthopedics",
    "Oncology",
    "Endocrinology",
    "Nephrology",
}


def run_validations(members_df: pd.DataFrame, claims_df: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    add_check(
        "members_unique_member_id",
        members_df["member_id"].is_unique and members_df["member_id"].notna().all(),
        "Member IDs must be unique and non-null.",
    )
    add_check(
        "claims_unique_claim_id",
        claims_df["claim_id"].is_unique and claims_df["claim_id"].notna().all(),
        "Claim IDs must be unique and non-null.",
    )

    missing_values = int(members_df.isna().sum().sum() + claims_df.isna().sum().sum())
    add_check(
        "no_missing_values",
        missing_values == 0,
        f"Total missing values across both tables: {missing_values}.",
    )

    fk_missing = int((~claims_df["member_id"].isin(set(members_df["member_id"]))).sum())
    add_check(
        "claims_member_fk_integrity",
        fk_missing == 0,
        f"Claims with unknown member_id: {fk_missing}.",
    )

    age_in_range = bool(members_df["age"].between(18, 100).all())
    add_check("member_age_in_range", age_in_range, "Member age must be between 18 and 100.")

    financial_logic_valid = bool(
        (claims_df["allowed_amount"] >= claims_df["paid_amount"]).all()
        and (claims_df["paid_amount"] >= 0).all()
    )
    add_check(
        "financial_amount_logic",
        financial_logic_valid,
        "paid_amount must be non-negative and <= allowed_amount.",
    )

    service_dates = pd.to_datetime(claims_df["service_date"], errors="coerce")
    discharge_dates = pd.to_datetime(claims_df["discharge_date"], errors="coerce")
    date_logic_valid = bool((discharge_dates >= service_dates).all())
    add_check("date_order_valid", date_logic_valid, "discharge_date must be on/after service_date.")

    admission_values_valid = bool(claims_df["admission_type"].isin(VALID_ADMISSION_TYPES).all())
    add_check(
        "admission_type_valid",
        admission_values_valid,
        f"admission_type must be one of {sorted(VALID_ADMISSION_TYPES)}.",
    )

    diagnosis_values_valid = bool(claims_df["diagnosis_group"].isin(VALID_DIAGNOSIS_GROUPS).all())
    add_check(
        "diagnosis_group_valid",
        diagnosis_values_valid,
        f"diagnosis_group must be one of {sorted(VALID_DIAGNOSIS_GROUPS)}.",
    )

    readmission_binary_valid = bool(claims_df["readmitted_30d"].isin({0, 1}).all())
    add_check(
        "readmitted_30d_binary",
        readmission_binary_valid,
        "readmitted_30d must be binary (0/1).",
    )

    failed_checks = [check for check in checks if not check["passed"]]
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "members": int(len(members_df)),
            "claims": int(len(claims_df)),
        },
        "checks": checks,
        "summary": {
            "status": "PASS" if not failed_checks else "FAIL",
            "passed_checks": len(checks) - len(failed_checks),
            "failed_checks": len(failed_checks),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data quality checks for healthcare datasets.")
    parser.add_argument(
        "--members-path", type=Path, default=Path("data/raw/members.csv"), help="Members CSV path."
    )
    parser.add_argument(
        "--claims-path", type=Path, default=Path("data/raw/claims.csv"), help="Claims CSV path."
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validation/validation_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 1 if validation report status is FAIL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members_df = pd.read_csv(args.members_path)
    claims_df = pd.read_csv(args.claims_path)
    report = run_validations(members_df, claims_df)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation report written: {args.report_path}")
    print(f"Validation status: {report['summary']['status']}")

    if args.fail_on_error and report["summary"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
