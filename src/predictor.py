from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import src.config as config
from src.config import DATE_COL, validate_forecast_target_date
from src.data_processing import infer_next_service_date
from src.features import add_basic_calendar_features, add_lag_features, merge_weather_features
from src.production_features import (
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
    build_locked_f6_feature_row,
    service_horizon_between,
    validate_package_feature_contract,
)
from src.weather import WeatherClient


@dataclass
class PredictionOutput:
    service_date: pd.Timestamp
    predicted_visitors: float
    predicted_quantile: float
    residual_buffer: float
    suggested_meals: int
    meal_buffer_pct: float
    model_segment: str
    package_id: str | None = None
    model_package_schema_version: int | None = None
    feature_set_id: str | None = None
    feature_order_sha256: str | None = None
    recommendation_policy_id: str | None = None
    forecast_origin: pd.Timestamp | None = None
    calendar_days_ahead: int | None = None
    service_horizon: int | None = None


class WeatherForecastUnavailableError(RuntimeError):
    """Raised when live forecast weather is unavailable for the requested service date."""


class VisitorPredictor:
    def __init__(self, model_path: str):
        pack = joblib.load(model_path)
        self.model_path = str(Path(model_path))
        self.models = pack["models"]
        self.quantile_models = pack["quantile_models"]
        self.feature_cols = pack["feature_cols"]
        self.history_df = pack["history_df"].copy()
        self.default_meal_buffer_pct = pack.get("default_meal_buffer_pct", 0.08)
        self.residual_buffer_by_day = pack.get("residual_buffer_by_day", {"sat": 0.0, "sun": 0.0})
        weather_context = pack.get("weather_context", {})
        self.weather_zip_code = weather_context.get("zip_code", "12550")
        self.weather_country_code = weather_context.get("country_code", "US")
        self.weather_timezone = weather_context.get("timezone", "America/New_York")
        self.model_package_schema_version = int(pack.get("model_package_schema_version", 1))
        self.package_id = str(pack.get("package_id") or Path(model_path).name)
        self.feature_contract = pack.get("feature_contract")
        self.preprocessors = pack.get("preprocessors", {})
        if self.model_package_schema_version not in {1, MODEL_PACKAGE_SCHEMA_VERSION}:
            raise ValueError(
                "Unsupported model-package schema version: "
                f"{self.model_package_schema_version}"
            )
        self.uses_locked_f6 = self.model_package_schema_version == MODEL_PACKAGE_SCHEMA_VERSION
        if self.uses_locked_f6:
            if not isinstance(self.feature_contract, dict):
                raise ValueError("F6 model package feature_contract must be a mapping")
            validate_package_feature_contract(self.feature_contract, self.feature_cols)
            expected_segments = {"sat", "sun"}
            if set(self.models) != expected_segments:
                raise ValueError(
                    "F6 model package must contain exactly Saturday and Sunday point models"
                )
            if set(self.quantile_models) != expected_segments:
                raise ValueError(
                    "F6 model package must contain exactly Saturday and Sunday quantile models"
                )
            if set(self.preprocessors) != expected_segments:
                raise ValueError(
                    "F6 model package must contain exactly Saturday and Sunday preprocessors"
                )
            required_history_columns = {DATE_COL, "visitors"}
            missing_history_columns = required_history_columns.difference(
                self.history_df.columns
            )
            if missing_history_columns:
                raise ValueError(
                    "F6 model package history is missing columns: "
                    f"{sorted(missing_history_columns)}"
                )
            if self.history_df.empty:
                raise ValueError("F6 model package attendance history is empty")
            self.recommendation_policy_id = pack.get("recommendation_policy_id")
            if self.recommendation_policy_id != RECOMMENDATION_POLICY_ID:
                raise ValueError("F6 model package recommendation policy is not locked C0")
        else:
            if self.feature_contract is not None:
                raise ValueError("Schema-v1 model packages cannot declare an F6 feature contract")
            self.recommendation_policy_id = "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS"

    @staticmethod
    def _segment_for_date(target_date: pd.Timestamp) -> str:
        wd = pd.to_datetime(target_date).weekday()
        if wd == 5:
            return "sat"
        if wd == 6:
            return "sun"
        raise ValueError("target_date must be Saturday or Sunday")

    def _prepare_one_row(self, target_date: pd.Timestamp) -> pd.DataFrame:
        if getattr(self, "uses_locked_f6", False):
            today = pd.Timestamp(config.forecast_today(self.weather_timezone)).normalize()
            origin = min(today, pd.Timestamp(target_date).normalize() - pd.Timedelta(days=1))
            return build_locked_f6_feature_row(
                self.history_df,
                target_date,
                origin,
            )

        history = self.history_df.sort_values(DATE_COL).copy().reset_index(drop=True)
        row = {DATE_COL: pd.to_datetime(target_date), "visitors": np.nan}
        tmp = pd.concat([history[[DATE_COL, "visitors"]], pd.DataFrame([row])], ignore_index=True)

        tmp = add_basic_calendar_features(tmp)
        tmp = add_lag_features(tmp)

        one = tmp.iloc[[-1]].copy()

        client = WeatherClient(
            zip_code=self.weather_zip_code,
            country=self.weather_country_code,
            timezone=self.weather_timezone,
        )
        try:
            weather = client.fetch_forecast_daily(target_date.date())
        except Exception as exc:
            raise WeatherForecastUnavailableError(
                f"Weather forecast data is unavailable for {target_date:%Y-%m-%d}."
            ) from exc
        if weather.empty or "date" not in weather.columns:
            raise WeatherForecastUnavailableError(
                f"Weather forecast data is unavailable for {target_date:%Y-%m-%d}."
            )

        weather_dates = pd.to_datetime(weather["date"], errors="coerce").dt.date
        target_weather = weather.loc[weather_dates == target_date.date()].copy()
        if target_weather.empty:
            raise WeatherForecastUnavailableError(
                f"Weather forecast data is unavailable for {target_date:%Y-%m-%d}."
            )

        weather_feature_cols = [
            "temp_10_13",
            "apparent_temp_10_13",
            "humidity_10_13",
            "wind_10_13",
            "precip_10_13",
        ]
        if any(column not in target_weather.columns for column in weather_feature_cols):
            raise WeatherForecastUnavailableError(
                f"Weather forecast data is unavailable for {target_date:%Y-%m-%d}."
            )
        if target_weather[weather_feature_cols].isna().any(axis=None):
            raise WeatherForecastUnavailableError(
                f"Weather forecast data is unavailable for {target_date:%Y-%m-%d}."
            )

        need_weather = any(c in self.feature_cols for c in weather_feature_cols)
        if need_weather:
            one = merge_weather_features(one, target_weather)

        one = one.replace([np.inf, -np.inf], np.nan)
        for c in self.feature_cols:
            if c not in one.columns:
                one[c] = 0.0
            one[c] = one[c].fillna(self.history_df[c].median() if c in self.history_df.columns else 0.0)

        return one[self.feature_cols]

    def predict_next(self, target_date: str | None = None, meal_buffer_pct: float | None = None) -> PredictionOutput:
        if target_date:
            requested_date = target_date
        else:
            requested_date = infer_next_service_date(self.history_df[DATE_COL])

        dt = pd.Timestamp(validate_forecast_target_date(requested_date, timezone=self.weather_timezone))
        today = pd.Timestamp(config.forecast_today(self.weather_timezone)).normalize()
        forecast_origin = min(today, dt.normalize() - pd.Timedelta(days=1))
        calendar_days_ahead = int((dt.normalize() - forecast_origin).days)
        service_horizon = service_horizon_between(forecast_origin, dt)
        segment = self._segment_for_date(dt)
        if segment not in self.models:
            raise ValueError(f"No trained model for segment: {segment}")

        x = self._prepare_one_row(dt)
        if getattr(self, "uses_locked_f6", False):
            x = self.preprocessors[segment].transform(x)
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                raise ValueError(
                    f"F6 {segment} preprocessor produced non-finite transformed features"
                )
        pred_point = float(self.models[segment].predict(x)[0])
        pred_q = float(self.quantile_models[segment].predict(x)[0]) if segment in self.quantile_models else pred_point
        if not np.isfinite(pred_point) or not np.isfinite(pred_q):
            raise ValueError("Model prediction produced a non-finite value")

        if getattr(
            self,
            "recommendation_policy_id",
            "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS",
        ) == RECOMMENDATION_POLICY_ID:
            buffer_pct = 0.0
            residual_buf = 0.0
            suggested = pred_q
        else:
            buffer_pct = self.default_meal_buffer_pct if meal_buffer_pct is None else meal_buffer_pct
            residual_buf = float(self.residual_buffer_by_day.get(segment, 0.0))
            suggested = max(
                pred_point * (1 + buffer_pct),
                pred_q,
                pred_point + residual_buf,
            )

        return PredictionOutput(
            service_date=dt,
            predicted_visitors=pred_point,
            predicted_quantile=pred_q,
            residual_buffer=residual_buf,
            suggested_meals=int(np.ceil(suggested)),
            meal_buffer_pct=buffer_pct,
            model_segment=segment,
            package_id=getattr(self, "package_id", None),
            model_package_schema_version=getattr(
                self, "model_package_schema_version", None
            ),
            feature_set_id=(
                self.feature_contract.get("feature_set_id")
                if isinstance(getattr(self, "feature_contract", None), dict)
                else None
            ),
            feature_order_sha256=(
                self.feature_contract.get("feature_order_sha256")
                if isinstance(getattr(self, "feature_contract", None), dict)
                else None
            ),
            recommendation_policy_id=getattr(
                self,
                "recommendation_policy_id",
                "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS",
            ),
            forecast_origin=forecast_origin,
            calendar_days_ahead=calendar_days_ahead,
            service_horizon=service_horizon,
        )
