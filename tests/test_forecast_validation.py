from __future__ import annotations

from datetime import date, timedelta
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import ForecastTargetDateError, validate_forecast_target_date
from src.prediction_logs import save_prediction_log
from src.predictor import PredictionOutput, VisitorPredictor, WeatherForecastUnavailableError


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.value])


class ForecastValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 6, 19)  # Friday; day 15 is Saturday and day 16 is Sunday.
        self.today_patch = patch("src.config.forecast_today", return_value=self.today)
        self.today_patch.start()

    def tearDown(self) -> None:
        self.today_patch.stop()

    def _predictor(self) -> VisitorPredictor:
        predictor = VisitorPredictor.__new__(VisitorPredictor)
        predictor.models = {"sat": _ConstantModel(100), "sun": _ConstantModel(110)}
        predictor.quantile_models = {"sat": _ConstantModel(120), "sun": _ConstantModel(125)}
        predictor.feature_cols = []
        predictor.history_df = pd.DataFrame(
            {
                "service_date": pd.to_datetime(["2026-06-13", "2026-06-14"]),
                "visitors": [100, 110],
            }
        )
        predictor.default_meal_buffer_pct = 0.08
        predictor.residual_buffer_by_day = {"sat": 5.0, "sun": 5.0}
        predictor.weather_zip_code = "12550"
        predictor.weather_country_code = "US"
        predictor.weather_timezone = "America/New_York"
        return predictor

    @staticmethod
    def _weather_for(target_date: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [target_date],
                "temp_10_13": [72.0],
                "apparent_temp_10_13": [73.0],
                "humidity_10_13": [55.0],
                "wind_10_13": [8.0],
                "precip_10_13": [0.0],
            }
        )

    def test_saturday_and_sunday_within_window_succeed(self) -> None:
        predictor = self._predictor()
        for target_date in [self.today + timedelta(days=1), self.today + timedelta(days=2)]:
            with self.subTest(target_date=target_date), patch(
                "src.predictor.WeatherClient.fetch_forecast_daily",
                return_value=self._weather_for(target_date),
            ):
                prediction = predictor.predict_next(target_date.isoformat())
                self.assertEqual(prediction.service_date.date(), target_date)

    def test_weekday_is_rejected(self) -> None:
        with self.assertRaises(ForecastTargetDateError):
            validate_forecast_target_date(self.today + timedelta(days=3))

    def test_past_date_is_rejected(self) -> None:
        with self.assertRaises(ForecastTargetDateError):
            validate_forecast_target_date(date(2026, 6, 14))

    def test_day_15_is_allowed(self) -> None:
        self.assertEqual(
            validate_forecast_target_date(self.today + timedelta(days=15)),
            self.today + timedelta(days=15),
        )

    def test_day_16_is_rejected(self) -> None:
        with self.assertRaises(ForecastTargetDateError):
            validate_forecast_target_date(self.today + timedelta(days=16))

    def test_missing_weather_is_rejected(self) -> None:
        predictor = self._predictor()
        with patch("src.predictor.WeatherClient.fetch_forecast_daily", return_value=pd.DataFrame()):
            with self.assertRaises(WeatherForecastUnavailableError):
                predictor.predict_next((self.today + timedelta(days=1)).isoformat())

    def test_weather_request_failure_is_rejected(self) -> None:
        predictor = self._predictor()
        with patch(
            "src.predictor.WeatherClient.fetch_forecast_daily",
            side_effect=RuntimeError("weather service unavailable"),
        ):
            with self.assertRaises(WeatherForecastUnavailableError):
                predictor.predict_next((self.today + timedelta(days=1)).isoformat())

    def test_weather_for_another_date_is_rejected(self) -> None:
        predictor = self._predictor()
        target_date = self.today + timedelta(days=1)
        with patch(
            "src.predictor.WeatherClient.fetch_forecast_daily",
            return_value=self._weather_for(target_date + timedelta(days=1)),
        ):
            with self.assertRaises(WeatherForecastUnavailableError):
                predictor.predict_next(target_date.isoformat())

    def test_invalid_date_is_not_saved_to_prediction_logs(self) -> None:
        prediction = PredictionOutput(
            service_date=pd.Timestamp(self.today + timedelta(days=16)),
            predicted_visitors=100,
            predicted_quantile=110,
            residual_buffer=5,
            suggested_meals=115,
            meal_buffer_pct=0.08,
            model_segment="sun",
        )
        with patch("src.prediction_logs._connect") as connect:
            with self.assertRaises(ForecastTargetDateError):
                save_prediction_log("ny_12550", prediction)
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
