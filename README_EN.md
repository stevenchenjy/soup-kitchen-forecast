# Visitor Forecasting and Meal Prep Assistant (Multi-Location)

## Overview
- Login and role-based access (`admin` / `staff`)
- Switch locations after login
- Per-location isolated storage:
  - SQLite database: `data/locations/<location_id>/attendance.db`
  - Weather cache: `data/locations/<location_id>/weather_daily.csv`
  - Model file: `models/visitor_model_<location_id>.joblib`
  - Backtest artifacts and charts: `artifacts/<location_id>/...`
- Per-location forecasting and meal prep recommendation
- Separate Saturday/Sunday modeling and backtesting

## Default Accounts
- `admin / admin123`
- `staff / staff123`

## 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Start the System (Streamlit)
Admin dashboard:
```bash
streamlit run app.py --server.port 8501
```

Staff dashboard:
```bash
streamlit run app_staff.py --server.port 8502
```

## 3) Data Locations
Raw historical source file (used for initial seeding):
- `team_amounts_2023_2026.xlsx`

Location configuration:
- `data/locations.json`

Per-location data storage:
- `data/locations/<location_id>/attendance.db`
- `data/locations/<location_id>/weather_daily.csv`

Seed historical data into one location (from Excel):
```bash
python scripts/prepare_data.py --location ny_12550
```

## 4) Model Training / Incremental Update
Train one location (backtest + model output):
```bash
python scripts/train_backtest.py --location ny_12550
```

Incremental update (current implementation reuses the same training pipeline):
```bash
python scripts/retrain_incremental.py --location ny_12550
```

## 5) Add a New Location
1. Edit `data/locations.json` and add a new item under `locations`, for example:

```json
{
  "id": "la_90012",
  "name": "Los Angeles, CA 90012",
  "zip_code": "90012",
  "country_code": "US",
  "timezone": "America/Los_Angeles"
}
```

2. Seed data for the new location:
```bash
python scripts/prepare_data.py --location la_90012
```

3. Train model for the new location:
```bash
python scripts/train_backtest.py --location la_90012
```

After that, the new `location` will be available in the location list after login.
