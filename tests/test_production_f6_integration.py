from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from scripts import nightly_retrain, train_backtest, train_f6_candidate
from src.config import DATE_COL, TARGET_COL, model_file_for_location
from src.predictor import VisitorPredictor
from src.production_features import (
    LOCKED_F6_FEATURES,
    LOCKED_F6_FEATURE_ORDER_SHA256,
    MODEL_PACKAGE_SCHEMA_VERSION,
    OPTIONAL_RESEARCH_LOCK_ARTIFACT,
    RECOMMENDATION_POLICY_ID,
    TRACKED_F6_CONTRACT,
    build_locked_f6_feature_row,
    build_locked_f6_training_frame,
    feature_order_sha256,
    load_tracked_f6_contract,
    locked_feature_contract_metadata,
    service_horizon_between,
    validate_research_lock_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_SHA256 = "ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0"
FALLBACK_MODEL_SHA256 = "cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920"


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, values) -> np.ndarray:
        assert np.asarray(values).shape[1] == len(LOCKED_F6_FEATURES)
        return np.array([self.value])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "package_id": "f6-synthetic-v1",
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


def test_tracked_f6_contract_loads_with_exact_order_and_hash() -> None:
    contract = load_tracked_f6_contract()
    assert TRACKED_F6_CONTRACT.is_file()
    assert TRACKED_F6_CONTRACT.relative_to(ROOT).parts[:2] == ("config", "model_contracts")
    assert contract["feature_set_id"] == "F6_COMPACT_SELECTED"
    assert contract["feature_contract_version"] == "f6_v1"
    assert tuple(contract["ordered_feature_list"]) == LOCKED_F6_FEATURES
    assert len(LOCKED_F6_FEATURES) == 33
    assert feature_order_sha256(LOCKED_F6_FEATURES) == LOCKED_F6_FEATURE_ORDER_SHA256
    assert contract["feature_order_sha256"] == LOCKED_F6_FEATURE_ORDER_SHA256


def test_stage1_builders_need_no_ignored_research_or_supabase_inputs(tmp_path: Path) -> None:
    missing_research = tmp_path / "artifacts/ny_12550/model_optimization"
    missing_export = tmp_path / "data/locations/ny_12550/Updated"
    assert not missing_research.exists()
    assert not missing_export.exists()

    contract = load_tracked_f6_contract()
    bundle = build_locked_f6_training_frame(attendance_history())
    assert contract == locked_feature_contract_metadata()
    assert bundle.feature_cols == list(LOCKED_F6_FEATURES)
    assert not bundle.df.empty


@pytest.mark.skipif(
    not OPTIONAL_RESEARCH_LOCK_ARTIFACT.is_file(),
    reason="Optional ignored Phase 2A.5 lock artifact is unavailable",
)
def test_optional_local_research_lock_matches_tracked_contract() -> None:
    artifact = validate_research_lock_artifact()
    assert artifact["ordered_feature_list"] == list(LOCKED_F6_FEATURES)


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


def test_schema_v2_synthetic_package_uses_f6_w0_and_locked_c0(tmp_path: Path) -> None:
    path = tmp_path / "f6.joblib"
    joblib.dump(f6_package(), path)
    predictor = VisitorPredictor(str(path))
    with patch("src.config.forecast_today", return_value=date(2026, 6, 19)), patch(
        "src.predictor.WeatherClient.fetch_forecast_daily"
    ) as fetch_weather:
        prediction = predictor.predict_next("2026-06-20", meal_buffer_pct=0.30)
    fetch_weather.assert_not_called()
    assert predictor.model_package_schema_version == 2
    assert predictor.package_id == "f6-synthetic-v1"
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


def test_unknown_schema_version_fails_clearly(tmp_path: Path) -> None:
    package = f6_package()
    package["model_package_schema_version"] = 3
    path = tmp_path / "unknown.joblib"
    joblib.dump(package, path)
    with pytest.raises(ValueError, match="Unsupported model-package schema version: 3"):
        VisitorPredictor(str(path))


def test_existing_schema_v1_package_remains_loadable_and_legacy() -> None:
    active_path = ROOT / "models/visitor_model_ny_12550.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = VisitorPredictor(str(active_path))
    assert predictor.model_package_schema_version == 1
    assert predictor.package_id == active_path.name
    assert predictor.uses_locked_f6 is False
    assert predictor.recommendation_policy_id == "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS"


def test_candidate_destination_refuses_active_models_directory(tmp_path: Path) -> None:
    active_model = model_file_for_location("ny_12550")
    with pytest.raises(ValueError, match="outside the active models directory"):
        train_f6_candidate.candidate_package_dir(
            "ny_12550", active_model, "f6-candidate-v1"
        )
    with pytest.raises(ValueError, match="outside the active models directory"):
        train_f6_candidate.candidate_package_dir(
            "ny_12550", ROOT / "models", "f6-candidate-v1"
        )


def test_candidate_destination_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "candidates" / "f6-candidate-v1"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        train_f6_candidate.candidate_package_dir(
            "ny_12550", tmp_path / "candidates", "f6-candidate-v1"
        )


def test_candidate_package_has_complete_metadata_checksums_and_no_activation(
    tmp_path: Path,
) -> None:
    attendance_path = tmp_path / "attendance.csv"
    attendance_history().to_csv(attendance_path, index=False)
    fake_outputs = {
        key: {"metrics": {"BacktestRows": 1}, "predictions": pd.DataFrame()}
        for key in ("overall", "sat", "sun")
    }
    fake_models = ({"sat": object(), "sun": object()},) * 3
    active_path = model_file_for_location("ny_12550")
    active_before = sha256(active_path)

    with patch.object(
        train_f6_candidate, "rolling_backtest_by_daytype", return_value=fake_outputs
    ), patch.object(
        train_f6_candidate, "fit_final_models_by_daytype", return_value=fake_models
    ):
        destination = train_f6_candidate.train_f6_candidate(
            location_id="ny_12550",
            attendance_csv=attendance_path,
            output_dir=tmp_path / "candidates",
            package_id="f6-candidate-v1",
        )

    metadata = json.loads((destination / train_f6_candidate.METADATA_NAME).read_text())
    checksums = json.loads((destination / train_f6_candidate.CHECKSUMS_NAME).read_text())
    package = joblib.load(destination / train_f6_candidate.MODEL_PACKAGE_NAME)
    assert set(destination.iterdir()) == {
        destination / train_f6_candidate.MODEL_PACKAGE_NAME,
        destination / train_f6_candidate.METADATA_NAME,
        destination / train_f6_candidate.CHECKSUMS_NAME,
    }
    assert metadata["package_id"] == "f6-candidate-v1"
    assert metadata["package_status"] == "candidate_not_active"
    assert metadata["location_id"] == "ny_12550"
    assert metadata["model_package_schema_version"] == 2
    assert metadata["feature_contract"] == locked_feature_contract_metadata()
    assert metadata["recommendation_policy_id"] == RECOMMENDATION_POLICY_ID
    assert metadata["activation"] == {
        "active_model_changed": False,
        "requires_separate_activation_stage": True,
    }
    assert metadata["attendance_input"]["sha256"] == sha256(attendance_path)
    assert checksums["algorithm"] == "sha256"
    for filename, digest in checksums["files"].items():
        assert sha256(destination / filename) == digest
    assert package["package_status"] == "candidate_not_active"
    assert package["model_package_schema_version"] == 2
    assert sha256(active_path) == active_before == ACTIVE_MODEL_SHA256


def test_nightly_active_training_remains_legacy_schema_v1() -> None:
    history = attendance_history()
    fake_outputs = {
        key: {"metrics": {"BacktestRows": 0}, "predictions": pd.DataFrame()}
        for key in ("overall", "sat", "sun")
    }
    dumped: list[dict] = []
    assert nightly_retrain.train_location is train_backtest.train_location

    with patch.object(train_backtest, "bootstrap_location_from_csv"), patch.object(
        train_backtest, "load_clean_data", return_value=history
    ), patch.object(
        train_backtest, "_load_or_build_weather", return_value=None
    ), patch.object(
        train_backtest, "rolling_backtest_by_daytype", return_value=fake_outputs
    ), patch.object(
        train_backtest,
        "fit_final_models_by_daytype",
        return_value=(
            {"sat": object(), "sun": object()},
            {"sat": object(), "sun": object()},
        ),
    ), patch.object(
        train_backtest, "_write_predictions", return_value=pd.DataFrame()
    ), patch.object(
        train_backtest, "_write_metrics", return_value={}
    ), patch.object(
        train_backtest, "_write_plots"
    ), patch.object(
        train_backtest.joblib,
        "dump",
        side_effect=lambda package, path: dumped.append(package),
    ):
        train_backtest.train_location("ny_12550")

    assert len(dumped) == 1
    package = dumped[0]
    assert "model_package_schema_version" not in package
    assert "feature_contract" not in package
    assert "preprocessors" not in package
    assert package["default_meal_buffer_pct"] == 0.08
    assert set(package["residual_buffer_by_day"]) == {"sat", "sun"}


def test_tracked_active_models_and_source_data_remain_unchanged() -> None:
    assert sha256(ROOT / "models/visitor_model_ny_12550.joblib") == ACTIVE_MODEL_SHA256
    assert sha256(ROOT / "models/visitor_model.joblib") == FALLBACK_MODEL_SHA256
    assert sha256(ROOT / "data/locations/ny_12550/attendance.db") == (
        "d4b0df65bebac69fe3069199cc71d062c2eea956102aafaf66425c1ce8a30d9d"
    )
    assert "data/locations/*/Updated/*.csv" in (ROOT / ".gitignore").read_text()
