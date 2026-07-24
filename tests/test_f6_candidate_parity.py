from __future__ import annotations

from copy import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import DATE_COL, TARGET_COL
from src.f6_readiness import (
    EXPECTED_CANDIDATE_PACKAGE_SHA256,
    evaluate_parity_case,
    parity_case_registry,
    sunday_leakage_verification,
    verify_candidate_directory,
)
from src.predictor import VisitorPredictor
from src.production_features import (
    LOCKED_F6_FEATURE_ORDER_SHA256,
    build_locked_f6_feature_result,
    build_locked_f6_feature_row,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1"
CANDIDATE_PACKAGE = CANDIDATE_DIR / "model_package.joblib"
ACTIVE_MODEL = ROOT / "models/visitor_model_ny_12550.joblib"
VALID_CASES = tuple(case for case in parity_case_registry() if case.valid)
CASE_BY_ID = {case.case_id: case for case in parity_case_registry()}

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Could not find the number of physical cores:UserWarning"
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def candidate_predictor() -> VisitorPredictor:
    predictor, _ = verify_candidate_directory(CANDIDATE_DIR)
    return predictor


@pytest.fixture(scope="module")
def parity_results(candidate_predictor: VisitorPredictor) -> dict[str, object]:
    return {
        case.case_id: evaluate_parity_case(candidate_predictor, case)
        for case in VALID_CASES
    }


def test_candidate_checksum_and_integrity_verification(
    candidate_predictor: VisitorPredictor,
) -> None:
    _, result = verify_candidate_directory(CANDIDATE_DIR)
    assert result["checksums_valid"] is True
    assert result["package_sha256"] == EXPECTED_CANDIDATE_PACKAGE_SHA256
    assert result["package_id"] == "ny_12550_f6_2026-07-12_v1"
    assert result["schema_version"] == 2
    assert result["feature_count"] == 33
    assert result["model_segments"] == ["sat", "sun"]
    assert result["quantile_segments"] == ["sat", "sun"]
    assert result["preprocessor_segments"] == ["sat", "sun"]
    assert result["history_maximum_date"] == "2026-07-12"
    assert candidate_predictor.package_id == result["package_id"]


def test_candidate_fresh_subprocess_load_avoids_local_only_inputs() -> None:
    code = """
import json
from datetime import date
from pathlib import Path
import sys
from unittest.mock import patch

root = Path.cwd().resolve()
blocked = [
    (root / "data/updated").resolve(),
    (root / "data/locations/ny_12550/Updated").resolve(),
    (root / "artifacts/ny_12550/model_optimization").resolve(),
]

def audit(event, args):
    if event != "open" or not args:
        return
    try:
        path = Path(args[0]).resolve()
    except (TypeError, OSError):
        return
    if any(path == item or item in path.parents for item in blocked):
        raise AssertionError(f"local-only input accessed: {path}")

sys.addaudithook(audit)
from src.predictor import VisitorPredictor
path = root / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
predictor = VisitorPredictor(str(path))
with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
    prediction = predictor.predict_next("2026-07-18")
print(json.dumps({
    "package_id": predictor.package_id,
    "schema": predictor.model_package_schema_version,
    "segment": prediction.model_segment,
    "meals": prediction.suggested_meals,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "package_id": "ny_12550_f6_2026-07-12_v1",
        "schema": 2,
        "segment": "sat",
        "meals": 116,
    }


@pytest.mark.parametrize("case_id", ["P1", "P2", "P3", "P4"], ids=str)
def test_primary_raw_feature_parity(
    parity_results: dict[str, object],
    case_id: str,
) -> None:
    result = parity_results[case_id]
    rows = result.raw_feature_rows
    assert len(rows) == 33
    assert all(row["within_tolerance"] for row in rows)
    assert max(row["absolute_difference"] for row in rows) <= 1e-10
    values = {row["feature"]: row["direct_value"] for row in rows}
    case = CASE_BY_ID[case_id]
    assert values["calendar_days_ahead"] == case.calendar_days_ahead
    assert values["service_horizon"] == case.service_horizon
    assert values["future_eligible_services_between"] == case.service_horizon - 1


def test_missingness_raw_feature_parity(
    parity_results: dict[str, object],
) -> None:
    result = parity_results["P5"]
    rows = result.raw_feature_rows
    missing = {row["feature"] for row in rows if row["direct_missing"]}
    assert {
        "last_observed_daytype_6",
        "daytype_median_last_6",
        "daytype_slot_last_observed",
        "daytype_slot_days_since_latest",
    }.issubset(missing)
    values = {row["feature"]: row["direct_value"] for row in rows}
    assert values["daytype_slot_history_missing"] == 1.0
    assert all(row["direct_missing"] == row["predictor_missing"] for row in rows)
    assert max(row["absolute_difference"] for row in rows) <= 1e-10


@pytest.mark.parametrize("case_id", ["P1", "P2", "P3", "P4", "P5"], ids=str)
def test_transformed_feature_parity(
    parity_results: dict[str, object],
    case_id: str,
) -> None:
    rows = parity_results[case_id].transformed_rows
    assert len(rows) == 33
    assert all(np.isfinite(row["direct_value"]) for row in rows)
    assert all(np.isfinite(row["predictor_value"]) for row in rows)
    assert max(row["absolute_difference"] for row in rows) <= 1e-10


@pytest.mark.parametrize(
    ("case_id", "segment", "expected_meals"),
    [
        ("P1", "sat", 116),
        ("P2", "sun", 169),
        ("P3", "sun", 178),
        ("P4", "sat", 114),
        ("P5", "sun", 149),
    ],
)
def test_prediction_and_c0_recommendation_parity(
    parity_results: dict[str, object],
    case_id: str,
    segment: str,
    expected_meals: int,
) -> None:
    row = parity_results[case_id].prediction_row
    assert row["segment"] == segment
    assert row["point_absolute_difference"] <= 1e-10
    assert row["quantile_absolute_difference"] <= 1e-10
    assert row["suggested_meals"] == row["expected_c0_meals"] == expected_meals
    assert row["meal_buffer_pct"] == 0.0
    assert row["residual_buffer"] == 0.0
    assert row["package_id"] == "ny_12550_f6_2026-07-12_v1"
    assert row["schema_version"] == 2
    assert row["recommendation_policy_id"] == "C0_EXISTING_RAW_QUANTILE"


def test_correct_segment_preprocessor_is_selected(
    candidate_predictor: VisitorPredictor,
) -> None:
    class CountingPreprocessor:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        def transform(self, values):
            self.calls += 1
            return self.wrapped.transform(values)

    predictor = copy(candidate_predictor)
    sat = CountingPreprocessor(candidate_predictor.preprocessors["sat"])
    sun = CountingPreprocessor(candidate_predictor.preprocessors["sun"])
    predictor.preprocessors = {"sat": sat, "sun": sun}
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        predictor.predict_next("2026-07-18")
    assert sat.calls == 1
    assert sun.calls == 0
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        predictor.predict_next("2026-07-19")
    assert sat.calls == 1
    assert sun.calls == 1
    assert candidate_predictor.preprocessors["sat"] is not (
        candidate_predictor.preprocessors["sun"]
    )
    assert not np.array_equal(
        candidate_predictor.preprocessors["sat"].statistics_,
        candidate_predictor.preprocessors["sun"].statistics_,
    )


def test_sunday_before_weekend_has_no_saturday_leakage(
    candidate_predictor: VisitorPredictor,
) -> None:
    result = sunday_leakage_verification(candidate_predictor, CASE_BY_ID["P2"])
    assert result["passed"] is True
    assert result["service_horizon"] == 2
    assert result["saturday_in_provenance"] is False
    assert result["post_origin_source_count"] == 0
    assert result["attendance_source_weekdays"] == ["Sunday"]
    assert result["maximum_raw_difference_after_saturday_append_or_mask"] <= 1e-10
    assert result["maximum_point_difference_after_saturday_append_or_mask"] <= 1e-10
    assert (
        result["maximum_quantile_difference_after_saturday_append_or_mask"] <= 1e-10
    )


def _package_fixture() -> dict:
    package = joblib.load(CANDIDATE_PACKAGE)
    package["models"] = dict(package["models"])
    package["quantile_models"] = dict(package["quantile_models"])
    package["preprocessors"] = dict(package["preprocessors"])
    package["feature_cols"] = list(package["feature_cols"])
    package["feature_contract"] = dict(package["feature_contract"])
    package["feature_contract"]["ordered_feature_list"] = list(
        package["feature_contract"]["ordered_feature_list"]
    )
    return package


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda package: package.update(model_package_schema_version=3),
            "Unsupported model-package schema version",
        ),
        (
            lambda package: package["models"].pop("sat"),
            "exactly Saturday and Sunday point models",
        ),
        (
            lambda package: package["models"].pop("sun"),
            "exactly Saturday and Sunday point models",
        ),
        (
            lambda package: package["quantile_models"].pop("sun"),
            "exactly Saturday and Sunday quantile models",
        ),
        (
            lambda package: package["preprocessors"].pop("sat"),
            "exactly Saturday and Sunday preprocessors",
        ),
        (
            lambda package: package.update(history_df=package["history_df"].iloc[0:0]),
            "attendance history is empty",
        ),
    ],
)
def test_candidate_package_structure_failures_stop_at_load(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    package = _package_fixture()
    mutation(package)
    monkeypatch.setattr("src.predictor.joblib.load", lambda path: package)
    with pytest.raises(ValueError, match=message):
        VisitorPredictor("synthetic-corrupt-package.joblib")


@pytest.mark.parametrize("corruption", ["order", "hash"])
def test_candidate_feature_contract_corruption_fails_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    package = _package_fixture()
    if corruption == "order":
        package["feature_cols"] = list(reversed(package["feature_cols"]))
    else:
        package["feature_contract"]["feature_order_sha256"] = "0" * 64
    monkeypatch.setattr("src.predictor.joblib.load", lambda path: package)
    with pytest.raises(ValueError, match="feature-order|locked ordered F6"):
        VisitorPredictor("synthetic-corrupt-contract.joblib")


def _temporary_candidate_directory(tmp_path: Path) -> Path:
    directory = tmp_path / CANDIDATE_DIR.name
    directory.mkdir(parents=True)
    shutil.copy2(CANDIDATE_PACKAGE, directory / "model_package.joblib")
    shutil.copy2(CANDIDATE_DIR / "metadata.json", directory / "metadata.json")
    shutil.copy2(CANDIDATE_DIR / "checksums.json", directory / "checksums.json")
    return directory


def test_candidate_directory_controlled_failures(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="directory is missing"):
        verify_candidate_directory(tmp_path / "missing")

    checksum_directory = _temporary_candidate_directory(tmp_path / "checksum")
    checksums = json.loads((checksum_directory / "checksums.json").read_text())
    checksums["files"]["model_package.joblib"] = "0" * 64
    (checksum_directory / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_candidate_directory(checksum_directory)

    metadata_directory = _temporary_candidate_directory(tmp_path / "metadata")
    metadata_path = metadata_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["package_id"] = "wrong-package-v1"
    metadata_path.write_text(json.dumps(metadata))
    checksums = json.loads((metadata_directory / "checksums.json").read_text())
    checksums["files"]["metadata.json"] = sha256(metadata_path)
    (metadata_directory / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="package_id does not match"):
        verify_candidate_directory(metadata_directory)


def test_candidate_json_containers_must_be_mappings(tmp_path: Path) -> None:
    checksums_directory = _temporary_candidate_directory(tmp_path / "checksums-list")
    (checksums_directory / "checksums.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum manifest must be a JSON mapping"):
        verify_candidate_directory(checksums_directory)

    metadata_directory = _temporary_candidate_directory(tmp_path / "metadata-list")
    metadata_path = metadata_directory / "metadata.json"
    metadata_path.write_text("[]")
    checksums_path = metadata_directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["files"]["metadata.json"] = sha256(metadata_path)
    checksums_path.write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="metadata must be a JSON mapping"):
        verify_candidate_directory(metadata_directory)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["feature_contract"].update(
                feature_order_sha256="0" * 64
            ),
            "metadata feature contract differs",
        ),
        (
            lambda metadata: metadata.update(recommendation_policy_id="wrong-policy"),
            "metadata recommendation policy differs",
        ),
        (
            lambda metadata: metadata["training"].update(
                training_window_id="TW_WRONG"
            ),
            "training training_window_id differs",
        ),
        (
            lambda metadata: metadata["attendance_input"].update(
                maximum_service_date="2026-07-11"
            ),
            "metadata attendance history does not end",
        ),
        (
            lambda metadata: metadata["activation"].update(
                active_model_changed=True
            ),
            "inactive activation state",
        ),
        (
            lambda metadata: metadata["tracked_contract"].update(sha256="0" * 64),
            "tracked-contract hash is incorrect",
        ),
    ],
)
def test_candidate_semantic_metadata_corruption_is_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    directory = _temporary_candidate_directory(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata))
    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["files"]["metadata.json"] = sha256(metadata_path)
    checksums_path.write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match=message):
        verify_candidate_directory(directory)


def test_invalid_target_horizon_origin_and_empty_history_fail_clearly(
    candidate_predictor: VisitorPredictor,
) -> None:
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        with pytest.raises(ValueError, match="Saturday or Sunday"):
            candidate_predictor.predict_next("2026-07-20")
    with pytest.raises(ValueError, match="service_horizon must be positive"):
        build_locked_f6_feature_row(
            candidate_predictor.history_df,
            "2026-07-18",
            "2026-07-17",
            service_horizon=0,
        )
    with pytest.raises(ValueError, match="forecast_origin must be earlier"):
        build_locked_f6_feature_result(
            candidate_predictor.history_df,
            "2026-07-18",
            "2026-07-18",
        )
    with pytest.raises(ValueError, match="Attendance history is empty"):
        build_locked_f6_feature_result(
            pd.DataFrame(columns=[DATE_COL, TARGET_COL]),
            "2026-07-18",
            "2026-07-17",
        )


def test_non_finite_transformed_features_fail_before_model_prediction(
    candidate_predictor: VisitorPredictor,
) -> None:
    class NonFinitePreprocessor:
        def transform(self, values):
            return np.full((1, 33), np.inf)

    class PredictionTripwire:
        def predict(self, values):
            raise AssertionError("model prediction must not be reached")

    predictor = copy(candidate_predictor)
    predictor.preprocessors = dict(candidate_predictor.preprocessors)
    predictor.preprocessors["sat"] = NonFinitePreprocessor()
    predictor.models = dict(candidate_predictor.models)
    predictor.quantile_models = dict(candidate_predictor.quantile_models)
    predictor.models["sat"] = PredictionTripwire()
    predictor.quantile_models["sat"] = PredictionTripwire()
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        with pytest.raises(ValueError, match="non-finite transformed features"):
            predictor.predict_next("2026-07-18")


def test_candidate_artifact_metadata_is_unchanged_and_active_remains_locked_f6() -> None:
    metadata = json.loads((CANDIDATE_DIR / "metadata.json").read_text())
    assert metadata["package_status"] == "candidate_not_active"
    assert metadata["activation"] == {
        "active_model_changed": False,
        "requires_separate_activation_stage": True,
    }
    active = VisitorPredictor(str(ACTIVE_MODEL))
    assert active.model_package_schema_version == 2
    assert active.uses_locked_f6 is True
    assert (
        active.feature_contract["feature_order_sha256"]
        == LOCKED_F6_FEATURE_ORDER_SHA256
    )
