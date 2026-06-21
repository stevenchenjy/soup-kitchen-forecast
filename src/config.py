from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "team_amounts_2023_2026.xlsx"
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "visitors_clean.csv"
WEATHER_DATA_FILE = PROJECT_ROOT / "data" / "weather_daily_12550.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "visitor_model.joblib"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOCATIONS_FILE = PROJECT_ROOT / "data" / "locations.json"
LOCATIONS_ROOT = PROJECT_ROOT / "data" / "locations"

ZIP_CODE = "12550"
COUNTRY_CODE = "US"
TIMEZONE = "America/New_York"
WEATHER_OBS_START_HOUR = 10
WEATHER_OBS_END_HOUR = 13
FORECAST_MAX_DAYS_AHEAD = 16

TARGET_COL = "visitors"
DATE_COL = "service_date"

BASELINE_MEALS_PREPARED = 150
ESTIMATED_WASTE_REDUCTION_RATE = 0.10
MEAL_WEIGHT_KG = 0.45
KG_CO2E_PER_KG_FOOD_WASTE = 3.8


class ForecastTargetDateError(ValueError):
    """Raised when a live forecast date is outside the supported service window."""


def forecast_today(timezone: str = TIMEZONE) -> date:
    return datetime.now(ZoneInfo(timezone)).date()


def validate_forecast_target_date(target_date: Any, timezone: str = TIMEZONE) -> date:
    try:
        if isinstance(target_date, datetime):
            localized_target = target_date.astimezone(ZoneInfo(timezone)) if target_date.tzinfo else target_date
            service_date = localized_target.date()
        elif isinstance(target_date, date):
            service_date = target_date
        elif hasattr(target_date, "date"):
            service_date = target_date.date()
        else:
            service_date = date.fromisoformat(str(target_date)[:10])
    except (TypeError, ValueError) as exc:
        raise ForecastTargetDateError("target_date must be a valid ISO date") from exc

    if not isinstance(service_date, date):
        raise ForecastTargetDateError("target_date must be a valid ISO date")
    if service_date.weekday() not in {5, 6}:
        raise ForecastTargetDateError("target_date must be Saturday or Sunday")

    today = forecast_today(timezone)
    last_forecast_date = today + timedelta(days=FORECAST_MAX_DAYS_AHEAD - 1)
    if service_date < today:
        raise ForecastTargetDateError("target_date cannot be in the past")
    if service_date > last_forecast_date:
        raise ForecastTargetDateError(
            f"target_date must be within {FORECAST_MAX_DAYS_AHEAD} days, through {last_forecast_date.isoformat()}"
        )
    return service_date


def _safe_location_id(location_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in location_id.strip().lower())


def location_dir(location_id: str) -> Path:
    return LOCATIONS_ROOT / _safe_location_id(location_id)


def location_db_file(location_id: str) -> Path:
    return location_dir(location_id) / "attendance.db"


def location_weather_file(location_id: str) -> Path:
    return location_dir(location_id) / "weather_daily.csv"


def model_file_for_location(location_id: str) -> Path:
    return PROJECT_ROOT / "models" / f"visitor_model_{_safe_location_id(location_id)}.joblib"


def artifact_dir_for_location(location_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / _safe_location_id(location_id)
