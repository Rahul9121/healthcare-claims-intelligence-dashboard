from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIAGNOSIS_GROUPS = [
    "Cardiology",
    "Pulmonology",
    "Orthopedics",
    "Oncology",
    "Endocrinology",
    "Nephrology",
]
PROCEDURE_GROUPS = [
    "Medication Management",
    "Lab and Diagnostics",
    "Surgery",
    "Imaging",
    "Rehabilitation",
]
ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]
PLAN_TYPES = ["Bronze", "Silver", "Gold", "Platinum"]
SPECIALTIES = [
    "Internal Medicine",
    "Cardiology",
    "Pulmonology",
    "Orthopedics",
    "Hospitalist",
    "Family Medicine",
]
REGIONS = ["Northeast", "Midwest", "South", "West"]
GENDERS = ["F", "M", "Other"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def _provider_reference(rng: np.random.Generator, n_providers: int = 80) -> pd.DataFrame:
    provider_ids = [f"P{idx:04d}" for idx in range(1, n_providers + 1)]
    provider_names = [f"Provider {idx:03d}" for idx in range(1, n_providers + 1)]
    provider_specialties = rng.choice(SPECIALTIES, size=n_providers)
    return pd.DataFrame(
        {
            "provider_id": provider_ids,
            "provider_name": provider_names,
            "provider_specialty": provider_specialties,
        }
    )


def generate_members(n_members: int, rng: np.random.Generator) -> pd.DataFrame:
    ages = np.clip(rng.normal(loc=57, scale=16, size=n_members).round(), 18, 95).astype(int)
    chronic_conditions_count = np.clip(rng.poisson(lam=2.2, size=n_members), 0, 8)
    risk_score_baseline = np.clip(
        0.12 + (ages * 0.004) + (chronic_conditions_count * 0.08) + rng.normal(0, 0.07, n_members),
        0.05,
        0.98,
    )
    enrollment_start = pd.Timestamp("2021-01-01")
    enrollment_dates = enrollment_start + pd.to_timedelta(rng.integers(0, 1200, n_members), unit="D")

    members = pd.DataFrame(
        {
            "member_id": [f"M{idx:06d}" for idx in range(1, n_members + 1)],
            "age": ages,
            "gender": rng.choice(GENDERS, p=[0.5, 0.48, 0.02], size=n_members),
            "region": rng.choice(REGIONS, size=n_members),
            "plan_type": rng.choice(PLAN_TYPES, p=[0.28, 0.37, 0.24, 0.11], size=n_members),
            "chronic_conditions_count": chronic_conditions_count,
            "risk_score_baseline": risk_score_baseline.round(4),
            "enrollment_date": enrollment_dates,
        }
    )
    return members


def generate_claims(
    members: pd.DataFrame, n_claims: int, rng: np.random.Generator
) -> pd.DataFrame:
    member_sample = members.sample(
        n=n_claims, replace=True, random_state=int(rng.integers(1, 1_000_000))
    ).reset_index(drop=True)
    provider_ref = _provider_reference(rng)
    provider_sample = provider_ref.sample(
        n=n_claims, replace=True, random_state=int(rng.integers(1, 1_000_000))
    ).reset_index(drop=True)

    service_dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, size=n_claims), unit="D"
    )
    length_of_stay = np.clip(rng.poisson(lam=3.2, size=n_claims) + 1, 1, 21)
    discharge_dates = service_dates + pd.to_timedelta(length_of_stay, unit="D")

    admission_types = rng.choice(ADMISSION_TYPES, p=[0.53, 0.2, 0.27], size=n_claims)
    diagnosis_groups = rng.choice(
        DIAGNOSIS_GROUPS, p=[0.2, 0.17, 0.22, 0.12, 0.19, 0.1], size=n_claims
    )
    procedure_groups = rng.choice(PROCEDURE_GROUPS, size=n_claims)
    prior_ed_visits_6m = np.clip(rng.poisson(lam=1.4, size=n_claims), 0, 12)
    medication_adherence = np.clip(rng.normal(loc=0.76, scale=0.15, size=n_claims), 0.2, 1.0)

    followup_probability = np.clip(
        0.85
        - (prior_ed_visits_6m * 0.04)
        - (member_sample["chronic_conditions_count"].to_numpy() * 0.03),
        0.25,
        0.93,
    )
    followup_within_7d = rng.binomial(1, followup_probability)

    diagnosis_cost_map = {
        "Cardiology": 2400,
        "Pulmonology": 1750,
        "Orthopedics": 2300,
        "Oncology": 3000,
        "Endocrinology": 1600,
        "Nephrology": 2100,
    }
    diagnosis_cost_factor = pd.Series(diagnosis_groups).map(diagnosis_cost_map).to_numpy()
    allowed_amount = np.clip(
        850
        + (length_of_stay * 520)
        + (prior_ed_visits_6m * 115)
        + diagnosis_cost_factor
        + rng.normal(0, 500, n_claims),
        300,
        50000,
    )
    paid_ratio = np.clip(rng.normal(0.79, 0.09, n_claims), 0.48, 0.98)
    paid_amount = allowed_amount * paid_ratio

    age = member_sample["age"].to_numpy()
    chronic = member_sample["chronic_conditions_count"].to_numpy()
    plan_type = member_sample["plan_type"].to_numpy()
    is_emergency = (admission_types == "Emergency").astype(int)
    is_cardiology = (diagnosis_groups == "Cardiology").astype(int)
    is_bronze = (plan_type == "Bronze").astype(int)

    risk_logit = (
        -3.0
        + (age * 0.018)
        + (chronic * 0.34)
        + (prior_ed_visits_6m * 0.24)
        + (length_of_stay * 0.16)
        + (is_emergency * 0.95)
        + (is_cardiology * 0.38)
        + (is_bronze * 0.22)
        - (followup_within_7d * 1.05)
        - (medication_adherence * 1.25)
    )
    readmission_prob = np.clip(_sigmoid(risk_logit + rng.normal(0, 0.55, n_claims)), 0.01, 0.98)
    readmitted_30d = rng.binomial(1, readmission_prob)

    claims = pd.DataFrame(
        {
            "claim_id": [f"C{idx:07d}" for idx in range(1, n_claims + 1)],
            "member_id": member_sample["member_id"],
            "service_date": service_dates,
            "discharge_date": discharge_dates,
            "provider_id": provider_sample["provider_id"],
            "provider_name": provider_sample["provider_name"],
            "provider_specialty": provider_sample["provider_specialty"],
            "diagnosis_group": diagnosis_groups,
            "procedure_group": procedure_groups,
            "admission_type": admission_types,
            "length_of_stay": length_of_stay,
            "prior_ed_visits_6m": prior_ed_visits_6m,
            "medication_adherence": medication_adherence.round(3),
            "followup_within_7d": followup_within_7d,
            "allowed_amount": allowed_amount.round(2),
            "paid_amount": paid_amount.round(2),
            "readmitted_30d": readmitted_30d,
        }
    )
    return claims.sort_values("service_date").reset_index(drop=True)


def write_synthetic_datasets(
    members_count: int, claims_count: int, seed: int, out_dir: Path
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    members = generate_members(members_count, rng)
    claims = generate_claims(members, claims_count, rng)

    members_path = out_dir / "members.csv"
    claims_path = out_dir / "claims.csv"
    members.to_csv(members_path, index=False)
    claims.to_csv(claims_path, index=False)
    return members_path, claims_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic members and claims datasets.")
    parser.add_argument("--members", type=int, default=3000, help="Number of members to generate.")
    parser.add_argument("--claims", type=int, default=18000, help="Number of claims to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw"), help="Output directory for CSV datasets."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members_path, claims_path = write_synthetic_datasets(
        members_count=args.members,
        claims_count=args.claims,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"Generated members: {members_path}")
    print(f"Generated claims: {claims_path}")


if __name__ == "__main__":
    main()
