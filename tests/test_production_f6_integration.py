from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from scripts import train_backtest
from src.config import DATE_COL, TARGET_COL
from src.predictor import VisitorPredictor
from src.production_features import (
    LOCKED_F6_FEATURES,
    LOCKED_F6_FEATURE_ORDER_SHA256,
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
    build_locked_f6_feature_row,
    build_locked_f6_training_frame,
    feature_order_sha256,
    locked_feature_contract_metadata,
    service_horizon_between,
    validate_lock_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, values) -> np.ndarray:
        assert np.asarray(values).shape[1] == len(LOCKED_F6_FEATURES)
        return np.array([self.value])


def attendance_history() -> pd.DataFrame:
    dates = pd.date_range("2025-01-04", "2026-06-14", freq="D")
    dates = dates[dates.weekday.isin([5, 6])]
    return pd.DataFrame(
        {
            DATE_COL: dates,
            TARGET_COL: 90.0 + np.arange(len(dates), dtype=float) % 50,
        }
    )


def f6_package() -> dict:
    history = attendance_history()
    training = build_locked_f6_training_frame(history)
    preprocessors = {}
    for segment, is_sun in (("sat", 0), ("sun", 1)):
        part = training.df[training.df["is_sun"] == is_sun]
        preprocessors[segment] = SimpleImputer(
            strategy="median", keep_empty_features=True
        ).fit(part[training.feature_cols])
    return {
        "model_package_schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
        "models": {"sat": ConstantModel(100), "sun": ConstantModel(110)},
        "quantile_models": {"sat": ConstantModel(120.2), "sun": ConstantModel(125.2)},
        "preprocessors": preprocessors,
        "feature_cols": list(LOCKED_F6_FEATURES),
        "feature_contract": locked_feature_contract_metadata(),
        "history_df": history,
        "recommendation_policy_id": RECOMMENDATION_POLICY_ID,
        "default_meal_buffer_pct": 0.0,
        "residual_buffer_by_day": {"sat": 0.0, "sun": 0.0},
        "weather_context": {
            "zip_code": "12550",
            "country_code": "US",
            "timezone": "America/New_York",
        },
    }


def test_code_owned_f6_contract_matches_locked_artifact_exactly() -> None:
    artifact = validate_lock_artifact()
    assert tuple(artifact["ordered_feature_list"]) == LOCKED_F6_FEATURES
    assert len(LOCKED_F6_FEATURES) == 33
    assert feature_order_sha256(LOCKED_F6_FEATURES) == LOCKED_F6_FEATURE_ORDER_SHA256


def test_training_and_direct_prediction_use_the_same_feature_builder() -> None:
    history = attendance_history()
    bundle = build_locked_f6_training_frame(history)
    assert bundle.feature_cols == list(LOCKED_F6_FEATURES)
    assert bundle.df["training_origin"].lt(bundle.df[DATE_COL]).all()
    assert bundle.history_df[DATE_COL].dt.weekday.isin([5, 6]).all()
    sample = bundle.df.iloc[40]
    direct = build_locked_f6_feature_row(
        history,
        sample[DATE_COL],
        sample["training_origin"],
        service_horizon=1,
    )
    pd.testing.assert_series_equal(
        direct.iloc[0],
        sample[list(LOCKED_F6_FEATURES)].astype(float),
        check_names=False,
    )


def test_live_service_horizon_counts_only_t1_weekend_services() -> None:
    assert service_horizon_between("2026-06-19", "2026-06-20") == 1
    assert service_horizon_between("2026-06-19", "2026-06-21") == 2
    assert service_horizon_between("2026-06-19", "2026-07-04") == 5


def test_schema_v2_predictor_uses_f6_w0_and_locked_c0(tmp_path: Path) -> None:
    path = tmp_path / "f6.joblib"
    joblib.dump(f6_package(), path)
    predictor = VisitorPredictor(str(path))
    with patch("src.config.forecast_today", return_value=date(2026, 6, 19)), patch(
        "src.predictor.WeatherClient.fetch_forecast_daily"
    ) as fetch_weather:
        prediction = predictor.predict_next("2026-06-20", meal_buffer_pct=0.30)
    fetch_weather.assert_not_called()
    assert predictor.uses_locked_f6 is True
    assert prediction.predicted_visitors == 100
    assert prediction.predicted_quantile == 120.2
    assert prediction.suggested_meals == 121
    assert prediction.meal_buffer_pct == 0.0
    assert prediction.residual_buffer == 0.0


def test_schema_v2_rejects_feature_order_drift(tmp_path: Path) -> None:
    package = f6_package()
    package["feature_cols"] = list(reversed(package["feature_cols"]))
    path = tmp_path / "invalid.joblib"
    joblib.dump(package, path)
    with pytest.raises(ValueError, match="locked ordered F6"):
        VisitorPredictor(str(path))


def test_existing_schema_v1_package_remains_loadable_and_legacy() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = VisitorPredictor(str(ROOT / "models/visitor_model_ny_12550.joblib"))
    assert predictor.model_package_schema_version == 1
    assert predictor.uses_locked_f6 is False
    assert predictor.recommendation_policy_id == "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS"


def test_training_assembles_f6_package_without_weather_or_real_fits() -> None:
    history = attendance_history()
    fake_outputs = {
        key: {"metrics": {"BacktestRows": 0}, "predictions": pd.DataFrame()}
        for key in ("overall", "sat", "sun")
    }
    dumped: list[dict] = []

    def fake_final_models(frame, feature_cols, quantile, *, return_preprocessors):
        assert feature_cols == list(LOCKED_F6_FEATURES)
        assert return_preprocessors is True
        return (
            {"sat": ConstantModel(100), "sun": ConstantModel(110)},
            {"sat": ConstantModel(120), "sun": ConstantModel(125)},
            {"sat": object(), "sun": object()},
        )

    with patch.object(train_backtest, "bootstrap_location_from_csv"), patch.object(
        train_backtest, "load_clean_data", return_value=history
    ), patch(
        "src.weather.build_weather_dataset"
    ) as weather_loader, patch.object(
        train_backtest, "rolling_backtest_by_daytype", return_value=fake_outputs
    ), patch.object(
        train_backtest, "fit_final_models_by_daytype", side_effect=fake_final_models
    ), patch.object(
        train_backtest, "_write_predictions", return_value=pd.DataFrame()
    ), patch.object(
        train_backtest, "_write_metrics", return_value={}
    ), patch.object(
        train_backtest, "_write_plots"
    ), patch.object(
        train_backtest.joblib, "dump", side_effect=lambda package, path: dumped.append(package)
    ):
        train_backtest.train_location("ny_12550")

    weather_loader.assert_not_called()
    assert len(dumped) == 1
    package = dumped[0]
    assert package["model_package_schema_version"] == 2
    assert package["feature_cols"] == list(LOCKED_F6_FEATURES)
    assert package["feature_contract"] == locked_feature_contract_metadata()
    assert package["recommendation_policy_id"] == RECOMMENDATION_POLICY_ID
    assert package["default_meal_buffer_pct"] == 0.0
    assert package["residual_buffer_by_day"] == {"sat": 0.0, "sun": 0.0}


def test_saved_models_supabase_export_and_local_attendance_are_unchanged() -> None:
    expected = {
        "models/visitor_model_ny_12550.joblib": "061be9292fb85cecbd4b5ab2a213bba9d4a692305d23234ff3f81c49affc94f2",
        "models/visitor_model.joblib": "cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920",
        "data/locations/ny_12550/Updated/2026-07-15T05-23_export.csv": "e3f84ac47245fa7eb5496413dbd04c5c0d0fead2ed553e257da57c3278ffdef8",
        "data/locations/ny_12550/attendance.db": "d4b0df65bebac69fe3069199cc71d062c2eea956102aafaf66425c1ce8a30d9d",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
