from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.predictor import VisitorPredictor
from src.production_features import feature_order_sha256, load_tracked_f6_contract


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = (
    ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1"
)
CANDIDATE_PACKAGE = CANDIDATE_DIR / "model_package.joblib"
EXPECTED_CANDIDATE_SHA256 = (
    "9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397"
)
EXPECTED_ACTIVE_SHA256 = (
    "9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397"
)
EXPECTED_FALLBACK_SHA256 = (
    "cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920"
)
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_versioned_f6_candidate_files_and_checksums_are_complete() -> None:
    assert sorted(path.name for path in CANDIDATE_DIR.iterdir() if path.is_file()) == [
        "checksums.json",
        "metadata.json",
        "model_package.joblib",
    ]
    assert sha256(CANDIDATE_PACKAGE) == EXPECTED_CANDIDATE_SHA256
    checksums = json.loads((CANDIDATE_DIR / "checksums.json").read_text())
    assert checksums["algorithm"] == "sha256"
    for filename, expected in checksums["files"].items():
        assert sha256(CANDIDATE_DIR / filename) == expected


def test_versioned_f6_candidate_metadata_records_validated_source() -> None:
    metadata = json.loads((CANDIDATE_DIR / "metadata.json").read_text())
    attendance = metadata["attendance_input"]
    assert metadata["package_id"] == "ny_12550_f6_2026-07-12_v1"
    assert metadata["package_status"] == "candidate_not_active"
    assert metadata["model_package_schema_version"] == 2
    assert metadata["recommendation_policy_id"] == "C0_EXISTING_RAW_QUANTILE"
    assert attendance["path"] == "data/updated/attendance_rows.csv"
    assert attendance["sha256"] == (
        "7bd8c4bf341b516a3370ffbe37eb517698ea68192656a03473a120cd50d603db"
    )
    assert attendance["byte_size"] == 30620
    assert attendance["encoding"] == "us-ascii"
    assert attendance["delimiter"] == ","
    assert attendance["column_names"] == [
        "location_id",
        "service_date",
        "visitors",
        "created_at",
        "updated_at",
    ]
    assert attendance["row_count"] == 360
    assert attendance["minimum_service_date"] == "2023-01-01"
    assert attendance["maximum_service_date"] == "2026-07-12"
    assert attendance["duplicate_service_date_rows"] == 0
    assert attendance["missing_attendance_values"] == 0
    assert attendance["invalid_attendance_values"] == 0
    assert attendance["non_weekend_service_dates"] == ["2026-04-14"]
    assert attendance["training_eligible_weekend_rows"] == 359
    assert metadata["activation"] == {
        "active_model_changed": False,
        "requires_separate_activation_stage": True,
    }


def test_versioned_f6_candidate_loads_with_locked_contract_and_models() -> None:
    package = joblib.load(CANDIDATE_PACKAGE)
    contract = load_tracked_f6_contract()
    assert package["model_package_schema_version"] == 2
    assert package["package_id"] == "ny_12550_f6_2026-07-12_v1"
    assert package["package_status"] == "candidate_not_active"
    assert package["feature_contract"] == contract
    assert package["feature_cols"] == contract["ordered_feature_list"]
    assert len(package["feature_cols"]) == 33
    assert feature_order_sha256(package["feature_cols"]) == (
        "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
    )
    assert package["recommendation_policy_id"] == "C0_EXISTING_RAW_QUANTILE"
    assert set(package["models"]) == {"sat", "sun"}
    assert set(package["quantile_models"]) == {"sat", "sun"}
    assert set(package["preprocessors"]) == {"sat", "sun"}
    assert package["default_meal_buffer_pct"] == 0.0
    assert package["residual_buffer_by_day"] == {"sat": 0.0, "sun": 0.0}

    for segment in ("sat", "sun"):
        point = package["models"][segment]
        point_params = point.get_params()
        assert type(point).__name__ == "RandomForestRegressor"
        assert point_params["n_estimators"] == 400
        assert point_params["max_depth"] == 8
        assert point_params["min_samples_leaf"] == 2
        assert point_params["random_state"] == 42

        quantile = package["quantile_models"][segment]
        quantile_params = quantile.get_params()
        assert type(quantile).__name__ == "HistGradientBoostingRegressor"
        assert quantile_params["loss"] == "quantile"
        assert quantile_params["quantile"] == 0.8
        assert quantile_params["learning_rate"] == 0.05
        assert quantile_params["max_depth"] == 4
        assert quantile_params["max_iter"] == 500
        assert quantile_params["random_state"] == 42

        preprocessor = package["preprocessors"][segment]
        assert type(preprocessor).__name__ == "SimpleImputer"
        assert preprocessor.strategy == "median"
        assert preprocessor.keep_empty_features is True

    history = package["history_df"].copy()
    history["service_date"] = pd.to_datetime(history["service_date"])
    assert len(history) == 359
    assert history["service_date"].min().date() == date(2023, 1, 1)
    assert history["service_date"].max().date() == date(2026, 7, 12)
    assert history["service_date"].dt.weekday.isin([5, 6]).all()

    predictor = VisitorPredictor(str(CANDIDATE_PACKAGE))
    assert predictor.model_package_schema_version == 2
    assert predictor.package_id == "ny_12550_f6_2026-07-12_v1"
    assert predictor.uses_locked_f6 is True


def test_versioned_candidate_is_the_exact_active_model_and_fallback_is_unchanged() -> None:
    assert sha256(ROOT / "models/visitor_model_ny_12550.joblib") == (
        EXPECTED_ACTIVE_SHA256
    )
    assert sha256(ROOT / "models/visitor_model.joblib") == EXPECTED_FALLBACK_SHA256
