"""Fit a versioned, inactive weather challenger from realized historical weather."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATE_COL, PROJECT_ROOT
from src.location_config import get_location
from src.predictor import VisitorPredictor
from src.weather_candidate import (
    MODEL_PACKAGE_NAME,
    build_weather_candidate_package,
    load_weather_candidate,
    sha256_file,
)
from src.weather_snapshots import extract_weather_features


def candidate_package_dir(output_dir: str | Path, package_id: str) -> Path:
    """Weather challengers may only be written beneath models/candidates."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*[-_]v[1-9][0-9]*", package_id):
        raise ValueError("package_id must contain only safe characters and end in _v1 or another explicit version")
    root = Path(output_dir).expanduser().resolve()
    allowed = (PROJECT_ROOT / "models" / "candidates").resolve()
    if root != allowed and allowed not in root.parents:
        raise ValueError(f"Weather candidate output must be inside {allowed}")
    destination = root / package_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Weather candidate already exists: {destination}")
    return destination


def _validate_weather_location(payload: dict[str, Any], *, timezone: str, latitude: float, longitude: float) -> None:
    if not np.isfinite([latitude, longitude]).all() or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Expected latitude/longitude must be finite valid coordinates")
    if payload.get("timezone") != timezone:
        raise ValueError("Historical weather timezone does not match location")
    try:
        actual_lat = float(payload["latitude"])
        actual_lon = float(payload["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Historical weather must identify its grid latitude and longitude") from exc
    if not np.isfinite([actual_lat, actual_lon]).all() or abs(actual_lat - latitude) > 0.25 or abs(actual_lon - longitude) > 0.25:
        raise ValueError("Historical weather grid coordinates do not match expected location (0.25-degree tolerance)")


def train_weather_candidate(
    *, location_id: str, baseline_model: str | Path, weather_input: str | Path,
    package_id: str, latitude: float, longitude: float,
    output_dir: str | Path = PROJECT_ROOT / "models" / "candidates",
) -> Path:
    destination = candidate_package_dir(output_dir, package_id)
    location = get_location(location_id)
    baseline_path = Path(baseline_model).expanduser().resolve()
    weather_path = Path(weather_input).expanduser().resolve()
    baseline_sha256 = sha256_file(baseline_path)
    weather_sha256 = sha256_file(weather_path)
    predictor = VisitorPredictor(str(baseline_path))
    baseline_package = joblib.load(baseline_path)
    if not predictor.uses_locked_f6 or baseline_package.get("location_id") != location_id:
        raise ValueError("Baseline must be a locked F6 package for the requested location")
    if any(baseline_package.get("weather_context", {}).get(key) != expected for key, expected in (
        ("zip_code", location.zip_code), ("country_code", location.country_code), ("timezone", location.timezone),
    )):
        raise ValueError("Baseline weather context must match the requested location")
    payload = json.loads(weather_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Historical weather input must be a JSON object")
    _validate_weather_location(payload, timezone=location.timezone, latitude=latitude, longitude=longitude)
    rows = []
    for value in predictor.history_df[DATE_COL]:
        service_date = pd.Timestamp(value).date()
        rows.append({DATE_COL: service_date, **extract_weather_features(payload, service_date, timezone=location.timezone)})
    package = build_weather_candidate_package(
        location_id=location_id, attendance=predictor.history_df,
        weather_features=pd.DataFrame(rows), package_id=package_id,
        baseline_source={"path": str(baseline_path), "sha256": baseline_sha256, "package_id": predictor.package_id},
        weather_input={"path": str(weather_path), "sha256": weather_sha256,
                       "data_kind": "realized_historical_bootstrap", "source": "Open-Meteo historical archive",
                       "response_grid_latitude": payload["latitude"], "response_grid_longitude": payload["longitude"]},
        weather_context={"zip_code": location.zip_code, "country_code": location.country_code,
                         "timezone": location.timezone, "latitude": float(latitude), "longitude": float(longitude)},
    )
    if sha256_file(baseline_path) != baseline_sha256 or sha256_file(weather_path) != weather_sha256:
        raise ValueError("Training source changed during candidate training; rerun with stable inputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_id}.tmp-", dir=destination.parent))
    destination_created = False
    try:
        package_path = temporary / MODEL_PACKAGE_NAME
        metadata_path = temporary / "metadata.json"
        joblib.dump(package, package_path)
        metadata_path.write_text(json.dumps(package["metadata"], indent=2, allow_nan=False) + "\n")
        checksums = {"algorithm": "sha256", "files": {
            MODEL_PACKAGE_NAME: sha256_file(package_path), "metadata.json": sha256_file(metadata_path),
        }}
        (temporary / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        destination.mkdir()  # Atomic reservation: never replace an existing package.
        destination_created = True
        for source in temporary.iterdir():
            source.rename(destination / source.name)
        temporary.rmdir()
        load_weather_candidate(destination, expected_location_id=location_id)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", required=True)
    parser.add_argument("--baseline-model", required=True, help="Trusted locked F6 package; read only")
    parser.add_argument("--weather-input", required=True, help="Historical Open-Meteo hourly JSON; never represented as archived forecasts")
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--latitude", required=True, type=float, help="Verified service-location latitude for checking the historical grid")
    parser.add_argument("--longitude", required=True, type=float, help="Verified service-location longitude for checking the historical grid")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "candidates"))
    args = parser.parse_args()
    destination = train_weather_candidate(
        location_id=args.location, baseline_model=args.baseline_model, weather_input=args.weather_input,
        package_id=args.package_id, latitude=args.latitude, longitude=args.longitude, output_dir=args.output_dir,
    )
    print(json.dumps({"candidate_directory": str(destination), "package_status": "shadow_only_not_validated", "active_model_changed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
