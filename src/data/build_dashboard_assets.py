from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _age_band(age: pd.Series) -> pd.Series:
    binned = pd.cut(
        age,
        bins=[17, 34, 49, 64, 120],
        labels=["18-34", "35-49", "50-64", "65+"],
        include_lowest=True,
    )
    return binned.astype("string").fillna("Unknown")


def _chronic_band(counts: pd.Series) -> pd.Series:
    binned = pd.cut(
        counts,
        bins=[-1, 1, 3, 99],
        labels=["0-1", "2-3", "4+"],
        include_lowest=True,
    )
    return binned.astype("string").fillna("Unknown")


def build_dashboard_assets(
    members_path: Path, claims_path: Path, output_dir: Path
) -> dict[str, Path]:
    members = pd.read_csv(members_path, parse_dates=["enrollment_date"])
    claims = pd.read_csv(claims_path, parse_dates=["service_date", "discharge_date"])
    output_dir.mkdir(parents=True, exist_ok=True)

    claims["service_month"] = claims["service_date"].dt.to_period("M").astype(str)

    utilization = (
        claims.groupby("service_month", as_index=False)
        .agg(
            total_claims=("claim_id", "count"),
            unique_members=("member_id", "nunique"),
            avg_length_of_stay=("length_of_stay", "mean"),
            readmission_rate=("readmitted_30d", "mean"),
            total_paid=("paid_amount", "sum"),
        )
        .sort_values("service_month")
    )
    utilization["avg_length_of_stay"] = utilization["avg_length_of_stay"].round(2)
    utilization["readmission_rate"] = (utilization["readmission_rate"] * 100).round(2)
    utilization["total_paid"] = utilization["total_paid"].round(2)

    cost_drivers = (
        claims.groupby("diagnosis_group", as_index=False)
        .agg(
            claim_count=("claim_id", "count"),
            total_allowed=("allowed_amount", "sum"),
            total_paid=("paid_amount", "sum"),
            avg_paid_per_claim=("paid_amount", "mean"),
            readmission_rate=("readmitted_30d", "mean"),
        )
        .sort_values("total_paid", ascending=False)
    )
    cost_drivers["readmission_rate"] = (cost_drivers["readmission_rate"] * 100).round(2)
    cost_drivers["avg_paid_per_claim"] = cost_drivers["avg_paid_per_claim"].round(2)

    provider_performance = (
        claims.groupby(["provider_id", "provider_name", "provider_specialty"], as_index=False)
        .agg(
            claim_count=("claim_id", "count"),
            avg_paid_per_claim=("paid_amount", "mean"),
            avg_length_of_stay=("length_of_stay", "mean"),
            readmission_rate=("readmitted_30d", "mean"),
            followup_rate=("followup_within_7d", "mean"),
        )
        .query("claim_count >= 30")
        .sort_values(["readmission_rate", "avg_paid_per_claim"], ascending=[True, False])
    )
    provider_performance["avg_paid_per_claim"] = provider_performance["avg_paid_per_claim"].round(2)
    provider_performance["avg_length_of_stay"] = provider_performance["avg_length_of_stay"].round(2)
    provider_performance["readmission_rate"] = (provider_performance["readmission_rate"] * 100).round(2)
    provider_performance["followup_rate"] = (provider_performance["followup_rate"] * 100).round(2)

    member_rollup = (
        claims.groupby("member_id", as_index=False)
        .agg(
            total_claims=("claim_id", "count"),
            total_paid=("paid_amount", "sum"),
            avg_length_of_stay=("length_of_stay", "mean"),
            readmission_rate=("readmitted_30d", "mean"),
            avg_prior_ed_visits=("prior_ed_visits_6m", "mean"),
        )
        .merge(
            members[
                ["member_id", "region", "plan_type", "age", "chronic_conditions_count", "risk_score_baseline"]
            ],
            on="member_id",
            how="left",
        )
    )
    member_rollup["age_band"] = _age_band(member_rollup["age"])
    member_rollup["chronic_band"] = _chronic_band(member_rollup["chronic_conditions_count"])
    member_rollup["readmission_rate"] = (member_rollup["readmission_rate"] * 100).round(2)
    member_rollup["avg_length_of_stay"] = member_rollup["avg_length_of_stay"].round(2)
    member_rollup["avg_prior_ed_visits"] = member_rollup["avg_prior_ed_visits"].round(2)
    member_rollup["total_paid"] = member_rollup["total_paid"].round(2)

    cohort_analysis = (
        member_rollup.groupby(["plan_type", "region", "age_band", "chronic_band"], as_index=False)
        .agg(
            members=("member_id", "count"),
            total_cost=("total_paid", "sum"),
            avg_cost_per_member=("total_paid", "mean"),
            avg_readmission_rate=("readmission_rate", "mean"),
        )
        .sort_values("total_cost", ascending=False)
    )
    cohort_analysis["total_cost"] = cohort_analysis["total_cost"].round(2)
    cohort_analysis["avg_cost_per_member"] = cohort_analysis["avg_cost_per_member"].round(2)
    cohort_analysis["avg_readmission_rate"] = cohort_analysis["avg_readmission_rate"].round(2)

    outputs = {
        "utilization_monthly": output_dir / "dashboard_utilization_monthly.csv",
        "cost_drivers": output_dir / "dashboard_cost_drivers.csv",
        "provider_performance": output_dir / "dashboard_provider_performance.csv",
        "cohort_analysis": output_dir / "dashboard_cohort_analysis.csv",
        "member_rollup": output_dir / "dashboard_member_rollup.csv",
    }
    utilization.to_csv(outputs["utilization_monthly"], index=False)
    cost_drivers.to_csv(outputs["cost_drivers"], index=False)
    provider_performance.to_csv(outputs["provider_performance"], index=False)
    cohort_analysis.to_csv(outputs["cohort_analysis"], index=False)
    member_rollup.to_csv(outputs["member_rollup"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard-ready analytical datasets.")
    parser.add_argument(
        "--members-path", type=Path, default=Path("data/raw/members.csv"), help="Members CSV path."
    )
    parser.add_argument(
        "--claims-path", type=Path, default=Path("data/raw/claims.csv"), help="Claims CSV path."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="Output directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_dashboard_assets(
        members_path=args.members_path,
        claims_path=args.claims_path,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
