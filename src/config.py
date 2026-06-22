from datetime import date, datetime, timedelta
from pathlib import Path
import re
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
    """Base class for invalid live forecast service dates."""


class ServiceDateParseError(ForecastTargetDateError):
    """Raised when a service date is not an accepted year-first date."""


class ServiceDateWeekdayError(ForecastTargetDateError):
    """Raised when a service date is not Saturday or Sunday."""


class ServiceDatePastError(ForecastTargetDateError):
    """Raised when a service date is before today."""


class ForecastHorizonError(ForecastTargetDateError):
    """Raised when a service date is beyond the supported forecast horizon."""


SERVICE_DATE_FORMAT_MESSAGE = "Please enter the service date in YYYY-MM-DD format, for example 2026-07-04."
_SERVICE_DATE_PATTERN = re.compile(r"^(\d{4})([-/])(\d{1,2})\2(\d{1,2})$")


def forecast_today(timezone: str = TIMEZONE) -> date:
    return datetime.now(ZoneInfo(timezone)).date()


def parse_service_date(value: Any, timezone: str = TIMEZONE) -> date:
    try:
        if isinstance(value, datetime):
            localized_value = value.astimezone(ZoneInfo(timezone)) if value.tzinfo else value
            service_date = localized_value.date()
            if not isinstance(service_date, date):
                raise ServiceDateParseError(SERVICE_DATE_FORMAT_MESSAGE)
            return service_date
        if isinstance(value, date):
            return value

        match = _SERVICE_DATE_PATTERN.fullmatch(str(value).strip())
        if match is None:
            raise ServiceDateParseError(SERVICE_DATE_FORMAT_MESSAGE)
        return date(int(match.group(1)), int(match.group(3)), int(match.group(4)))
    except ServiceDateParseError:
        raise
    except (TypeError, ValueError) as exc:
        raise ServiceDateParseError(SERVICE_DATE_FORMAT_MESSAGE) from exc


def validate_forecast_target_date(target_date: Any, timezone: str = TIMEZONE) -> date:
    service_date = parse_service_date(target_date, timezone=timezone)
    if service_date.weekday() not in {5, 6}:
        raise ServiceDateWeekdayError("Service date must be Saturday or Sunday.")

    today = forecast_today(timezone)
    last_forecast_date = today + timedelta(days=FORECAST_MAX_DAYS_AHEAD - 1)
    if service_date < today:
        raise ServiceDatePastError("Service date cannot be in the past.")
    if service_date > last_forecast_date:
        raise ForecastHorizonError(
            "Forecasts are only available within 16 days because weather forecasts are not reliable beyond that range."
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
