"""Strict 11 a.m.–1 p.m. weather inputs for the prospective F6 shadow study.

This module never calls an archive API or substitutes missing weather. A live
forecast's receipt time proves availability; Open-Meteo's seamless response does
not supply one reliable model issue time, so ``provider_issued_at`` stays null.
Variable definitions: https://open-meteo.com/en/docs#hourly-parameter-definition
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timezone as dt_timezone
import hashlib
import json
import math
from numbers import Real
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src.config import FORECAST_MAX_DAYS_AHEAD, TIMEZONE, parse_service_date


FEATURE_CONTRACT_ID = "weather_11_13_v1"
WEATHER_FEATURE_COLUMNS = (
    "wx_apparent_min_c", "wx_apparent_max_c", "wx_precip_mm",
    "wx_gust_max_kmh", "wx_snowfall_cm", "wx_snowdepth_max_m",
)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY_UNITS = {
    "time": "iso8601",
    "apparent_temperature": "°C",
    "precipitation": "mm",
    "wind_gusts_10m": "km/h",
    "snowfall": "cm",
    "snow_depth": "m",
}


class WeatherSnapshotError(ValueError):
    """Weather or its availability evidence does not meet the study contract."""


def _aware_utc(value: str | datetime, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(dt_timezone.utc)
    except (TypeError, ValueError) as exc:
        raise WeatherSnapshotError(f"{name} must be an aware ISO timestamp with a UTC offset.") from exc


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise WeatherSnapshotError(f"{name} must contain a finite numeric value; missing weather is not imputed.")
    result = float(value)
    if nonnegative and result < 0:
        raise WeatherSnapshotError(f"{name} cannot be negative.")
    return result


def _coordinates(latitude: Any, longitude: Any) -> tuple[float, float]:
    lat = _finite(latitude, "latitude")
    lon = _finite(longitude, "longitude")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise WeatherSnapshotError("Weather coordinates are outside WGS84 bounds.")
    return lat, lon


def extract_weather_features(
    payload: Mapping[str, Any], service_date: str | date, timezone: str = TIMEZONE,
) -> dict[str, float]:
    """Aggregate a complete, unit-checked Open-Meteo hourly response.

    Apparent temperature and snow depth are instantaneous at 11, 12 and 13.
    Precipitation, snowfall and maximum gusts describe the preceding hour, so
    only their values stamped 12 and 13 cover the exact two-hour service window.
    This pure extractor also accepts historical responses for explicitly labeled
    exploratory training; it makes no claim about when that weather was known.
    """
    zone = ZoneInfo(timezone)
    target = parse_service_date(service_date, timezone=timezone)
    if not isinstance(payload, Mapping) or payload.get("error"):
        raise WeatherSnapshotError("Weather provider returned an invalid response.")
    if payload.get("timezone") != timezone:
        raise WeatherSnapshotError(f"Weather timezone must be {timezone!r}.")
    units = payload.get("hourly_units")
    if not isinstance(units, Mapping):
        raise WeatherSnapshotError("Weather response is missing hourly units.")
    for variable, expected in HOURLY_UNITS.items():
        if units.get(variable) != expected:
            raise WeatherSnapshotError(f"Expected {variable} unit {expected!r}, got {units.get(variable)!r}.")
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or not isinstance(hourly.get("time"), list) or not hourly["time"]:
        raise WeatherSnapshotError("Weather response is missing hourly timestamps.")
    times = hourly["time"]
    for variable in HOURLY_UNITS:
        if not isinstance(hourly.get(variable), list) or len(hourly[variable]) != len(times):
            raise WeatherSnapshotError(f"Hourly {variable} length does not match hourly timestamps.")

    indexes: dict[datetime, int] = {}
    for index, stamp in enumerate(times):
        try:
            if not isinstance(stamp, str) or "T" not in stamp:
                raise ValueError("not an ISO datetime")
            parsed = datetime.fromisoformat(stamp)
            if parsed.minute or parsed.second or parsed.microsecond:
                raise ValueError("not an exact hour")
            if parsed.tzinfo is not None:
                local = parsed.astimezone(zone)
                if local.replace(tzinfo=None) != parsed.replace(tzinfo=None):
                    raise ValueError("offset does not match response timezone")
                parsed = local.replace(tzinfo=None)
        except (TypeError, ValueError) as exc:
            raise WeatherSnapshotError(f"Invalid local hourly timestamp: {stamp!r}.") from exc
        if parsed in indexes:
            raise WeatherSnapshotError(f"Duplicate hourly timestamp: {stamp}.")
        indexes[parsed] = index

    needed: dict[int, int] = {}
    for hour in (11, 12, 13):
        stamp = datetime.combine(target, time(hour))
        if stamp not in indexes:
            raise WeatherSnapshotError(f"Missing required weather hour {stamp.isoformat(timespec='minutes')}.")
        needed[hour] = indexes[stamp]

    def values(variable: str, hours: tuple[int, ...]) -> list[float]:
        return [
            _finite(hourly[variable][needed[hour]], f"{variable} at {hour}:00",
                    nonnegative=variable != "apparent_temperature")
            for hour in hours
        ]

    apparent = values("apparent_temperature", (11, 12, 13))
    features = dict(zip(WEATHER_FEATURE_COLUMNS, (
        min(apparent), max(apparent), sum(values("precipitation", (12, 13))),
        max(values("wind_gusts_10m", (12, 13))), sum(values("snowfall", (12, 13))),
        max(values("snow_depth", (11, 12, 13))),
    )))
    if not all(math.isfinite(value) for value in features.values()):
        raise WeatherSnapshotError("Aggregated weather values must be finite.")
    return features


def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    """Hash canonical JSON of the complete envelope, excluding its own digest."""
    content = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_location(location: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = dict(location) if isinstance(location, Mapping) else {
        name: getattr(location, name, None)
        for name in ("id", "name", "zip_code", "country_code", "timezone", "latitude", "longitude")
    }
    if not fields.get("id") or not fields.get("timezone"):
        raise WeatherSnapshotError("Location must have an id and an explicit IANA timezone.")
    ZoneInfo(fields["timezone"])
    has_lat, has_lon = fields.get("latitude") is not None, fields.get("longitude") is not None
    if has_lat != has_lon:
        raise WeatherSnapshotError("Explicit latitude and longitude must be supplied together.")
    if has_lat:
        lat, lon = _coordinates(fields["latitude"], fields["longitude"])
        return fields, {"source": "explicit_location_coordinates", "latitude": lat, "longitude": lon}

    postal_code = str(fields.get("zip_code") or "").strip()
    country = str(fields.get("country_code") or "").strip().upper()
    if not postal_code or len(country) != 2:
        raise WeatherSnapshotError("Location requires explicit coordinates or a postal code and country code.")
    params = {"name": postal_code, "countryCode": country, "count": 10, "language": "en", "format": "json"}
    response = requests.get(GEOCODING_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", []) if isinstance(payload, Mapping) else []
    matches = [row for row in results if isinstance(row, Mapping)
               and row.get("country_code") == country
               and row.get("timezone") == fields["timezone"]
               and postal_code in (row.get("postcodes") or [])]
    if len(matches) != 1:
        raise WeatherSnapshotError("Postal code did not resolve to one location in the configured country/timezone; provide explicit coordinates.")
    match = matches[0]
    lat, lon = _coordinates(match.get("latitude"), match.get("longitude"))
    return fields, {"source": GEOCODING_URL, "request_params": params, "result": dict(match),
                    "latitude": lat, "longitude": lon}


def fetch_weather_snapshot(
    service_date: str | date, cutoff_at: str | datetime, location: Any,
    *, now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Fetch a live forecast that finishes before the supplied preparation cutoff.

    ``location`` accepts the project's Location dataclass or a mapping; mappings
    may supply explicit latitude/longitude. Otherwise postal-code geocoding must
    yield one country/timezone/postal-code match. There is no location fallback.
    ``now`` is injectable for tests and must return an aware datetime.
    """
    clock = now or (lambda: datetime.now(dt_timezone.utc))
    cutoff = _aware_utc(cutoff_at, "cutoff_at")
    started = _aware_utc(clock(), "request_started_at")
    if started > cutoff:
        raise WeatherSnapshotError("Preparation cutoff has passed; a current forecast cannot be backdated.")
    timezone = location.get("timezone") if isinstance(location, Mapping) else getattr(location, "timezone", None)
    if not timezone:
        raise WeatherSnapshotError("Location must have an explicit IANA timezone.")
    zone = ZoneInfo(timezone)
    target = parse_service_date(service_date, timezone=timezone)
    service_start = datetime.combine(target, time(11), zone).astimezone(dt_timezone.utc)
    if cutoff >= service_start:
        raise WeatherSnapshotError("Preparation cutoff must be before 11:00 local time on the service date.")
    days_ahead = (target - started.astimezone(zone).date()).days
    if days_ahead < 0 or days_ahead >= FORECAST_MAX_DAYS_AHEAD:
        raise WeatherSnapshotError("Service date is outside the live weather forecast horizon.")
    fields, geolocation = _resolve_location(location)
    params = {
        "latitude": geolocation["latitude"], "longitude": geolocation["longitude"],
        "hourly": ",".join(variable for variable in HOURLY_UNITS if variable != "time"),
        "timezone": timezone, "start_hour": f"{target.isoformat()}T11:00",
        "end_hour": f"{target.isoformat()}T13:00", "temperature_unit": "celsius",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm", "timeformat": "iso8601",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    raw = response.json()
    retrieved = _aware_utc(clock(), "retrieved_at")
    if retrieved < started:
        raise WeatherSnapshotError("Receipt timestamp precedes request start; clock is inconsistent.")
    if retrieved > cutoff:
        raise WeatherSnapshotError("Forecast was received after the preparation cutoff and is ineligible.")
    features = extract_weather_features(raw, target, timezone)
    response_lat, response_lon = _coordinates(raw.get("latitude"), raw.get("longitude"))
    result = {
        "feature_contract_id": FEATURE_CONTRACT_ID, "service_date": target.isoformat(),
        "location_id": str(fields["id"]), "timezone": timezone,
        "request_started_at": started.isoformat(), "retrieved_at": retrieved.isoformat(),
        "cutoff_at": cutoff.isoformat(), "provider_issued_at": None,
        "source": "open_meteo_live_forecast", "source_url": FORECAST_URL,
        "request_params": params, "geolocation": geolocation,
        "response_location": {"latitude": response_lat, "longitude": response_lon,
                              "timezone": raw["timezone"], "utc_offset_seconds": raw.get("utc_offset_seconds")},
        "raw_response": raw, "features": features,
    }
    result["snapshot_sha256"] = snapshot_sha256(result)
    return result
