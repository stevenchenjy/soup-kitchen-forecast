from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    COUNTRY_CODE,
    TIMEZONE,
    WEATHER_OBS_END_HOUR,
    WEATHER_OBS_START_HOUR,
    ZIP_CODE,
)


@dataclass
class GeoPoint:
    latitude: float
    longitude: float
    name: str


class WeatherClient:
    """
    Weather module for NY 12550 and observation window 10am-1pm local time.
    Uses Open-Meteo APIs (no API key required).
    """

    def __init__(self, zip_code: str = ZIP_CODE, country: str = COUNTRY_CODE, timezone: str = TIMEZONE):
        self.zip_code = zip_code
        self.country = country
        self.timezone = timezone

    def _geocode(self) -> GeoPoint:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": self.zip_code, "country": self.country, "count": 1, "language": "en", "format": "json"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("results") or []
        if data:
            p = data[0]
            return GeoPoint(latitude=p["latitude"], longitude=p["longitude"], name=p.get("name", self.zip_code))

        return GeoPoint(latitude=41.5034, longitude=-74.0104, name="Newburgh")

    @staticmethod
    def _aggregate_10_to_13(df_hourly: pd.DataFrame) -> pd.DataFrame:
        if df_hourly.empty:
            return pd.DataFrame(columns=["date", "temp_10_13", "apparent_temp_10_13", "humidity_10_13", "wind_10_13", "precip_10_13"])

        d = df_hourly.copy()
        d["time"] = pd.to_datetime(d["time"])
        d["hour"] = d["time"].dt.hour
        d = d[(d["hour"] >= WEATHER_OBS_START_HOUR) & (d["hour"] <= WEATHER_OBS_END_HOUR)]
        d["date"] = d["time"].dt.date

        agg = (
            d.groupby("date", as_index=False)
            .agg(
                temp_10_13=("temperature_2m", "mean"),
                apparent_temp_10_13=("apparent_temperature", "mean"),
                humidity_10_13=("relative_humidity_2m", "mean"),
                wind_10_13=("wind_speed_10m", "mean"),
                precip_10_13=("precipitation", "sum"),
            )
            .sort_values("date")
        )
        return agg

    def fetch_historical_daily(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        loc = self._geocode()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation",
            "timezone": self.timezone,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        hourly = pd.DataFrame(r.json().get("hourly", {}))
        return self._aggregate_10_to_13(hourly)

    def fetch_forecast_daily(self, target_date: str | date) -> pd.DataFrame:
        loc = self._geocode()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation",
            "timezone": self.timezone,
            "start_date": str(target_date),
            "end_date": str(target_date),
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        hourly = pd.DataFrame(r.json().get("hourly", {}))
        return self._aggregate_10_to_13(hourly)


def build_weather_dataset(
    service_dates: pd.Series,
    out_csv: str | Path | None = None,
    zip_code: str = ZIP_CODE,
    country: str = COUNTRY_CODE,
    timezone: str = TIMEZONE,
) -> pd.DataFrame:
    client = WeatherClient(zip_code=zip_code, country=country, timezone=timezone)
    start = pd.to_datetime(service_dates.min()).date()
    end = pd.to_datetime(service_dates.max()).date()
    weather = client.fetch_historical_daily(start, end)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(out_csv, index=False)

    return weather
