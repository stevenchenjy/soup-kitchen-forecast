from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.config import DATE_COL
from src.data_processing import infer_next_service_date
from src.features import add_basic_calendar_features, add_lag_features, merge_weather_features
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


class VisitorPredictor:
    def __init__(self, model_path: str):
        pack = joblib.load(model_path)
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

    @staticmethod
    def _segment_for_date(target_date: pd.Timestamp) -> str:
        wd = pd.to_datetime(target_date).weekday()
        if wd == 5:
            return "sat"
        if wd == 6:
            return "sun"
        raise ValueError("target_date must be Saturday or Sunday")

    def _prepare_one_row(self, target_date: pd.Timestamp) -> pd.DataFrame:
        history = self.history_df.sort_values(DATE_COL).copy().reset_index(drop=True)
        row = {DATE_COL: pd.to_datetime(target_date), "visitors": np.nan}
        tmp = pd.concat([history[[DATE_COL, "visitors"]], pd.DataFrame([row])], ignore_index=True)

        tmp = add_basic_calendar_features(tmp)
        tmp = add_lag_features(tmp)

        one = tmp.iloc[[-1]].copy()

        need_weather = any(c in self.feature_cols for c in ["temp_10_13", "apparent_temp_10_13", "humidity_10_13", "wind_10_13", "precip_10_13"])
        if need_weather:
            client = WeatherClient(
                zip_code=self.weather_zip_code,
                country=self.weather_country_code,
                timezone=self.weather_timezone,
            )
            try:
                weather = client.fetch_forecast_daily(target_date.date())
            except Exception:
                weather = pd.DataFrame()
            if weather.empty:
                weather = pd.DataFrame(
                    [
                        {
                            "date": target_date.date(),
                            "temp_10_13": history.get("temp_10_13", pd.Series([0])).median() if "temp_10_13" in history.columns else 0,
                            "apparent_temp_10_13": history.get("apparent_temp_10_13", pd.Series([0])).median() if "apparent_temp_10_13" in history.columns else 0,
                            "humidity_10_13": history.get("humidity_10_13", pd.Series([0])).median() if "humidity_10_13" in history.columns else 0,
                            "wind_10_13": history.get("wind_10_13", pd.Series([0])).median() if "wind_10_13" in history.columns else 0,
                            "precip_10_13": 0,
                        }
                    ]
                )
            one = merge_weather_features(one, weather)

        one = one.replace([np.inf, -np.inf], np.nan)
        for c in self.feature_cols:
            if c not in one.columns:
                one[c] = 0.0
            one[c] = one[c].fillna(self.history_df[c].median() if c in self.history_df.columns else 0.0)

        return one[self.feature_cols]

    def predict_next(self, target_date: str | None = None, meal_buffer_pct: float | None = None) -> PredictionOutput:
        if target_date:
            dt = pd.to_datetime(target_date)
        else:
            dt = infer_next_service_date(self.history_df[DATE_COL])

        segment = self._segment_for_date(dt)
        if segment not in self.models:
            raise ValueError(f"No trained model for segment: {segment}")

        x = self._prepare_one_row(dt)
        pred_point = float(self.models[segment].predict(x)[0])
        pred_q = float(self.quantile_models[segment].predict(x)[0]) if segment in self.quantile_models else pred_point

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
        )
