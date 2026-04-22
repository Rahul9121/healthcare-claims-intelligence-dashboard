CREATE TABLE IF NOT EXISTS members (
    member_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    region VARCHAR(50) NOT NULL,
    plan_type VARCHAR(20) NOT NULL,
    chronic_conditions_count INT NOT NULL,
    risk_score_baseline NUMERIC(5, 4) NOT NULL,
    enrollment_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR(20) PRIMARY KEY,
    member_id VARCHAR(20) NOT NULL REFERENCES members(member_id),
    service_date DATE NOT NULL,
    discharge_date DATE NOT NULL,
    provider_id VARCHAR(20) NOT NULL,
    provider_name VARCHAR(120) NOT NULL,
    provider_specialty VARCHAR(80) NOT NULL,
    diagnosis_group VARCHAR(80) NOT NULL,
    procedure_group VARCHAR(80) NOT NULL,
    admission_type VARCHAR(20) NOT NULL,
    length_of_stay INT NOT NULL,
    prior_ed_visits_6m INT NOT NULL,
    medication_adherence NUMERIC(4, 3) NOT NULL,
    followup_within_7d INT NOT NULL,
    allowed_amount NUMERIC(12, 2) NOT NULL,
    paid_amount NUMERIC(12, 2) NOT NULL,
    readmitted_30d INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_member_id ON claims(member_id);
CREATE INDEX IF NOT EXISTS idx_claims_service_date ON claims(service_date);
CREATE INDEX IF NOT EXISTS idx_claims_provider_id ON claims(provider_id);
