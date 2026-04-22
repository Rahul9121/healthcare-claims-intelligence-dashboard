# Healthcare Claims Intelligence Dashboard + Readmission Risk API

Production-style healthcare analytics project that combines:
- synthetic claims + member data engineering,
- quality validation checks,
- executive analytics datasets for BI dashboards,
- readmission risk modeling,
- and a deployable FastAPI prediction service.

This project is built to mirror care management + payer analytics workflows you can confidently discuss in interviews.

## Live links
- GitHub Repository: `https://github.com/Rahul9121/healthcare-claims-intelligence-dashboard`
- Live API Docs (Render): `https://healthcare-claims-risk-api-rahul.onrender.com/docs`
- Live API Health: `https://healthcare-claims-risk-api-rahul.onrender.com/health`
- Live Dashboard: `https://healthcare-claims-dashboard-rahul.onrender.com/`

## Why this project matters
Healthcare operations teams need fast answers to:
- Where utilization and spend are increasing.
- Which diagnosis groups and providers are cost drivers.
- Which cohorts show elevated readmission risk.
- Which members should be prioritized for follow-up interventions.

This repo solves both sides:
- **Executive analytics** through dashboard-ready curated datasets.
- **Operational action** through a risk scoring API.

## Project outputs (already included in repo)
- Synthetic source data: `data/raw/members.csv`, `data/raw/claims.csv`
- Validation report: `data/validation/validation_report.json` (10/10 checks passed)
- Dashboard datasets:
  - `data/processed/dashboard_utilization_monthly.csv`
  - `data/processed/dashboard_cost_drivers.csv`
  - `data/processed/dashboard_provider_performance.csv`
  - `data/processed/dashboard_cohort_analysis.csv`
  - `data/processed/dashboard_member_rollup.csv`
- Model artifacts:
  - `models/readmission_model.joblib`
  - `models/model_metrics.json`
  - `models/sample_prediction_payload.json`

## Model performance snapshot
From the latest training run on 18,000 synthetic claims:
- ROC-AUC: **0.7428**
- Accuracy: **0.6808**
- Precision: **0.4284**
- Recall: **0.6649**
- F1-score: **0.5211**

## Architecture
1. Generate synthetic claims/member data (`src/data/generate_synthetic_data.py`)
2. Validate data quality (`src/data/validate_data.py`)
3. Train readmission model (`src/ml/train_model.py`)
4. Build dashboard marts (`src/data/build_dashboard_assets.py`)
5. Serve predictions via FastAPI (`src/api/main.py`)

Orchestrated by: `python -m src.pipeline`

## Tech stack
- **Python**: Pandas, NumPy, Scikit-learn, FastAPI
- **SQL**: PostgreSQL schema + analytics queries
- **BI**: Power BI Public / Tableau Public (CSV-ready datasets)
- **DevOps**: Docker, Docker Compose, GitHub Actions
- **Cloud deploy**: Render or Railway

## Quick start (local)
### 1) Install dependencies
```bash
py -m pip install -r requirements.txt
```

### 2) Run end-to-end pipeline
```bash
py -m src.pipeline --members 3000 --claims 18000 --seed 42
```

### 3) Start API
```bash
py -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4) Score one sample payload
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @models/sample_prediction_payload.json
```

## API endpoints
- `GET /` - service status
- `GET /health` - app + model readiness
- `POST /predict` - single-record prediction
- `POST /predict/batch` - batch scoring (up to 500 rows)

## Data validation checks
Automated checks cover:
- unique primary keys,
- missing values,
- foreign key integrity,
- age/domain constraints,
- date chronology,
- financial logic (`paid_amount <= allowed_amount`),
- binary readmission target integrity.

## PostgreSQL analytics layer
- DDL: `src/sql/schema.sql`
- Analytics queries: `src/sql/analytics_queries.sql`
- CSV-to-Postgres loader: `src/data/load_to_postgres.py`

Load to Postgres:
```bash
py -m src.data.load_to_postgres --db-url "postgresql+psycopg2://postgres:postgres@localhost:5432/claims_db"
```

## BI dashboard build guide (Power BI / Tableau Public)
Use files from `data/processed/` to create:
1. **Executive Overview** (utilization + readmission trend + total paid)
2. **Cost Driver Analysis** (diagnosis/procedure impact)
3. **Provider Scorecard** (cost vs readmission vs follow-up)
4. **Cohort Intelligence** (plan/region/chronic burden segments)

Publish to Power BI Public or Tableau Public and place your link in the **Live links** section.

## Deploy as a live API
### Deploy to Render
1. Push this repository to GitHub.
2. In Render, click **New +** → **Blueprint**.
3. Select your repo. Render reads `render.yaml` and `Dockerfile`.
4. Deploy and copy your URL:
   - `https://<service-name>.onrender.com/docs`

### Deploy to Railway
1. Create a new Railway project from GitHub.
2. Railway uses `railway.json` + `Dockerfile`.
3. Deploy and open:
   - `https://<service-name>.up.railway.app/docs`

## Run with Docker locally
```bash
docker compose up --build
```

## CI/CD
GitHub Actions workflow: `.github/workflows/ci.yml`
- install dependencies,
- run full pipeline,
- run Ruff lint checks,
- run Pytest.

## Suggested interview talking points
- Designed a healthcare claims intelligence pipeline with strong data quality controls.
- Built cohort-aware analytics for readmission-focused care management.
- Productized ML with a documented API and deployment-ready infrastructure.
- Delivered reproducible analytics + model outputs suitable for executive and operational use.

## Next enhancements
- SHAP explainability endpoint.
- Scheduled retraining and drift monitoring.
- Authentication + role-based access for production API.
- dbt layer for lineage and governance.
