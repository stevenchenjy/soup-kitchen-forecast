from pathlib import Path

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

TARGET_COL = "visitors"
DATE_COL = "service_date"


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
