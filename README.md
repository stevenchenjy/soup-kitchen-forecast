# Soup Kitchen Visitor Forecast and Meal Prep Assistant

## Project Description

This system helps soup kitchens forecast visitor attendance and make meal prep recommendations so teams can prepare enough food for guests while reducing avoidable food waste. It combines historical attendance data, location-specific weather context, and backtested forecasting models to support practical daily decisions for admin and staff users.

## Overview

Soup kitchens often need to plan meals before they know exactly how many visitors will arrive. Preparing too little can leave guests underserved, while preparing too much can waste food, staff time, and budget. This project provides a lightweight forecasting workflow for multi-location meal programs:

- Admin users manage locations, attendance records, users, training runs, and diagnostics.
- Staff users view authorized locations, record attendance, and receive meal prep recommendations.
- Forecasts are trained per location so each kitchen can keep its own attendance, weather cache, model file, and backtest artifacts.
- Optional Supabase configuration supports shared deployment storage; local SQLite and JSON storage support local demos and development.

## Features

- Role-based Streamlit apps for admin and staff workflows.
- Multi-location configuration through `data/locations.json`.
- Per-location attendance storage using local SQLite or Supabase.
- Visitor forecasts with suggested meal counts and configurable safety buffer.
- Saturday and Sunday model segmentation with rolling backtests.
- Weather feature integration through Open-Meteo APIs, with no API key required.
- Backtest outputs, metrics, and charts for model review.
- Prediction logging and actual-attendance reconciliation.
- Nightly retraining workflow for Supabase-backed deployments.

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

## Screenshots

Add updated screenshots before publishing or presenting the project:

- Admin dashboard overview: `docs/screenshots/admin-dashboard.png`
- Staff meal prep recommendation: `docs/screenshots/staff-recommendation.png`
- Backtest metrics and charts: `docs/screenshots/backtest-results.png`

## Requirements

- Python 3.12
- pip
- Streamlit
- Dependencies listed in `requirements.txt`

Use Python 3.12 for local development and deployment. See `DEPLOYMENT.md` for additional runtime notes.

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train or refresh a model for a configured location:

```bash
python scripts/train_backtest.py --location ny_12550
```

Run the admin dashboard:

```bash
streamlit run app.py --server.port 8501
```

Run the staff dashboard:

```bash
streamlit run app_staff.py --server.port 8502
```

The app can run locally with the bundled demo data and generated local files. Generated SQLite databases, weather caches, models, and artifacts should be treated as local outputs unless intentionally published.

## Local Demo Access

This repository includes seeded local-demo user records so the apps can be explored without setting up a production identity system first. Treat any bundled demo access as disposable development data only.

Before deploying the app for shared or public use:

- Create fresh admin and staff accounts.
- Rotate or replace any seeded local-demo users.
- Store production credentials and Supabase keys in Streamlit secrets or environment variables.
- Do not publish real passwords, service-role keys, or `.env` files.

## Data and Model Workflow

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

Run a one-location training job:

```bash
python scripts/train_backtest.py --location ny_12550
```

Run the incremental retraining entrypoint for one location:

```bash
python scripts/retrain_incremental.py --location ny_12550
```

For Supabase-backed deployments, the nightly retraining script can check dirty locations and retrain only when attendance changes:

```bash
python scripts/nightly_retrain.py --all
```

## Add a Location

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

Then train that location:

```bash
python scripts/train_backtest.py --location la_90012
```

After training, the location will be available in the app location list.

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
