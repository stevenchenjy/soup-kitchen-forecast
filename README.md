# Soup Kitchen Visitor Forecast and Meal Prep Assistant

## What this project does

Meal preparation starts before a kitchen knows how many guests will arrive. This project turns historical attendance into visitor forecasts and suggested meal counts, giving staff a starting point for balancing enough food with avoidable over-preparation.

There are two Streamlit dashboards: one for staff to view recommendations and record attendance, and one for administrators to manage locations, accounts, data, and model monitoring. The repository contains the application, forecasting code, a bundled model for `ny_12550`, and backtest artifacts. Backtests evaluate forecasts; measured food-waste reduction has not been established here.

If you are visiting from my [personal website](https://stevenchenjy.github.io/), start with the workflow below. For implementation and local setup, continue to [Run locally](#run-locally).

## A typical workflow

1. Choose an authorized location and a Saturday or Sunday service date.
2. View the attendance forecast and suggested number of meals.
3. Record actual attendance after the service.
4. Compare predictions with actual attendance in the monitoring dashboard.

Attendance, model packages, and generated outputs are organized by location. Local SQLite and JSON storage support development; optional Supabase storage supports shared deployments.

## How recommendations work

The current dashboards require a valid locked **F6** model package. For the bundled `ny_12550` model, F6 uses attendance-history and calendar features with separate Saturday and Sunday models. Its **C0** recommendation policy rounds the model's raw 80th-percentile prediction up to a whole meal count; it does not add an adjustable percentage or residual buffer.

Weather-based models and the earlier percentage-buffer calculation remain in the repository for separate research and legacy workflows. Open-Meteo weather access needs no API key, but live weather is not an input to the locked F6 recommendation path. An optional prospective weather comparison records forecast snapshots and checks preparation cutoffs alongside F6; see the [weather study documentation](docs/weather_shadow_study.md).

## Features

- Role-based Streamlit apps for admin and staff workflows.
- Multi-location configuration through `data/locations.json`.
- Per-location attendance storage using local SQLite or Supabase.
- Visitor forecasts and meal recommendations with model-package integrity checks.
- Saturday and Sunday model segmentation with rolling backtests.
- Separate weather-feature research through Open-Meteo APIs.
- Backtest outputs, metrics, and charts for model review.
- Prediction logging and actual-attendance reconciliation.
- Nightly retraining workflow for Supabase-backed deployments.
- Prediction provenance and comparison of backtest evidence with live observations.

## Repository Structure

```text
.
├── app.py                         # Admin Streamlit app
├── app_staff.py                   # Staff Streamlit app
├── src/                           # Forecasting, auth, storage, weather, and app support modules
├── scripts/                       # Training, retraining, and migration scripts
├── data/                          # Seed/config data used for local demos
├── models/                        # Generated model files
├── artifacts/                     # Generated backtest outputs and charts
├── .github/workflows/             # GitHub Actions automation
└── DEPLOYMENT.md                  # Runtime and Streamlit Cloud deployment notes
```

## Requirements

- Python 3.12
- pip
- Streamlit
- Dependencies listed in `requirements.txt`

Use Python 3.12 for local development and deployment. See `DEPLOYMENT.md` for additional runtime notes.

## Run locally

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with the bundled F6 model; no retraining is needed just to open the dashboards. Run the admin dashboard:

```bash
streamlit run app.py --server.port 8501
```

Run the staff dashboard:

```bash
streamlit run app_staff.py --server.port 8502
```

Open `http://localhost:8501` for the admin dashboard and `http://localhost:8502` for the staff dashboard. Sign-in is required; the staff view is restricted to assigned locations. These are local development entry points, not a public live-demo link.

Generated SQLite databases, weather caches, models, and artifacts should be treated as local outputs unless intentionally published.

## Local Demo Access

This repository includes seeded local-demo user records so the apps can be explored without setting up a production identity system first. Treat any bundled demo access as disposable development data only.

Before deploying the app for shared or public use:

- Create fresh admin and staff accounts.
- Rotate or replace any seeded local-demo users.
- Store production credentials and Supabase keys in Streamlit secrets or environment variables.
- Do not publish real passwords, service-role keys, or `.env` files.

## Data and model maintenance

Location settings live in `data/locations.json`. Each location has an ID, display name, zip code, country code, and timezone.

For local development, attendance records are stored under:

```text
data/locations/<location_id>/attendance.db
```

Weather caches are stored under:

```text
data/locations/<location_id>/weather_daily.csv
```

Training outputs are written to:

```text
models/visitor_model_<location_id>.joblib
artifacts/<location_id>/
```

For the current F6 workflow, read the [candidate verification](docs/f6_stage2_candidate_verification.md), [parity/readiness](docs/f6_stage3_4_activation_readiness.md), and [activation record](docs/f6_stage5_activation.md) before changing an active model. Candidate generation and model publication are separate operations.

`scripts/train_backtest.py` is the legacy schema-v1 training path. It writes directly to a location's active model file. Do not run it over the active `ny_12550` F6 package as a setup or refresh step: the current dashboard integrity check will not accept that legacy package as F6. The same distinction matters when adding a location; configuration alone does not provide a dashboard-ready F6 model.

The compatibility entry point `scripts/retrain_incremental.py` delegates to that same legacy training path and carries the same restriction.

For Supabase-backed deployments, the nightly retraining script can check dirty locations and retrain only when attendance changes:

```bash
python scripts/nightly_retrain.py --all
```

## Location configuration

Edit `data/locations.json` and add a location:

```json
{
  "id": "la_90012",
  "name": "Los Angeles, CA 90012",
  "zip_code": "90012",
  "country_code": "US",
  "timezone": "America/Los_Angeles"
}
```

The location appears in the configured location list, but attendance history, authorized users, and an appropriate validated model package are also needed for a usable forecast. The example above changes configuration only.

## Deployment Notes

- Use Python 3.12 in Streamlit Cloud and GitHub Actions.
- Deploy `app.py` as the admin app and `app_staff.py` as the staff app.
- Configure Supabase secrets for shared storage and nightly retraining.
- Run `scripts/create_attendance_change_log.sql` in Supabase before deploying the Staff latest-entry deletion feature.
- Keep service-role keys and other secrets out of Git.
- Review generated model and artifact publishing intentionally; they may be large, stale, or environment-specific.

Expected Supabase-related environment variables or Streamlit secrets include:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
SUPABASE_ANON_KEY
SUPABASE_USERS_TABLE
SUPABASE_ATTENDANCE_TABLE
SUPABASE_ATTENDANCE_CHANGE_LOG_TABLE
SUPABASE_PREDICTION_LOGS_TABLE
SUPABASE_MODEL_TRAINING_RUNS_TABLE
SUPABASE_MODEL_RETRAIN_STATE_TABLE
```

See `DEPLOYMENT.md` for the current deployment checklist.
