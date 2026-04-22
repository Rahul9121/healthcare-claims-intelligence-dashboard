import pandas as pd

from src.data.validate_data import run_validations


def _sample_frames():
    members = pd.DataFrame(
        {
            "member_id": ["M000001", "M000002"],
            "age": [45, 67],
            "gender": ["F", "M"],
            "region": ["West", "South"],
            "plan_type": ["Gold", "Silver"],
            "chronic_conditions_count": [1, 3],
            "risk_score_baseline": [0.2, 0.6],
            "enrollment_date": ["2022-01-01", "2022-06-15"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["C0000001", "C0000002"],
            "member_id": ["M000001", "M000002"],
            "service_date": ["2024-01-10", "2024-02-01"],
            "discharge_date": ["2024-01-13", "2024-02-04"],
            "provider_id": ["P0001", "P0002"],
            "provider_name": ["Provider 001", "Provider 002"],
            "provider_specialty": ["Cardiology", "Hospitalist"],
            "diagnosis_group": ["Cardiology", "Nephrology"],
            "procedure_group": ["Imaging", "Surgery"],
            "admission_type": ["Emergency", "Urgent"],
            "length_of_stay": [3, 4],
            "prior_ed_visits_6m": [1, 2],
            "medication_adherence": [0.8, 0.65],
            "followup_within_7d": [1, 0],
            "allowed_amount": [4200, 5100],
            "paid_amount": [3300, 4000],
            "readmitted_30d": [0, 1],
        }
    )
    return members, claims


def test_validation_passes_for_clean_data():
    members, claims = _sample_frames()
    report = run_validations(members, claims)
    assert report["summary"]["status"] == "PASS"
    assert report["summary"]["failed_checks"] == 0


def test_validation_fails_for_unknown_member_id():
    members, claims = _sample_frames()
    claims.loc[0, "member_id"] = "UNKNOWN_MEMBER"
    report = run_validations(members, claims)
    assert report["summary"]["status"] == "FAIL"
    failing_checks = [check["name"] for check in report["checks"] if not check["passed"]]
    assert "claims_member_fk_integrity" in failing_checks
