"""Deterministic F6 candidate integrity and production-parity verification."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL
from src.predictor import PredictionOutput, VisitorPredictor
from src.production_features import (
    LOCKED_F6_FEATURE_ORDER_SHA256,
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
    TRACKED_F6_CONTRACT,
    build_locked_f6_feature_result,
    feature_order_sha256,
    load_tracked_f6_contract,
)


EXPECTED_CANDIDATE_PACKAGE_SHA256 = (
    "9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397"
)
RAW_TOLERANCE = 1e-10
TRANSFORMED_TOLERANCE = 1e-10
PREDICTION_TOLERANCE = 1e-10
OPERATIONAL_ATTENDANCE_MAXIMUM = 10_000.0


@dataclass(frozen=True)
class ParityCase:
    case_id: str
    target_date: str
    forecast_origin: str
    service_horizon: int
    segment: str
    expected_hidden_dates: tuple[str, ...]
    purpose: str
    valid: bool = True

    @property
    def target(self) -> pd.Timestamp:
        return pd.Timestamp(self.target_date).normalize()

    @property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.forecast_origin).normalize()

    @property
    def calendar_days_ahead(self) -> int:
        return int((self.target - self.origin).days)


@dataclass(frozen=True)
class ParityEvaluation:
    registry_row: dict[str, Any]
    raw_feature_rows: tuple[dict[str, Any], ...]
    transformed_rows: tuple[dict[str, Any], ...]
    prediction_row: dict[str, Any]
    provenance: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parity_case_registry() -> tuple[ParityCase, ...]:
    return (
        ParityCase(
            "P1",
            "2026-07-18",
            "2026-07-17",
            1,
            "sat",
            ("2026-07-18",),
            "Saturday one-service-ahead from the immediately preceding Friday",
        ),
        ParityCase(
            "P2",
            "2026-07-19",
            "2026-07-17",
            2,
            "sun",
            ("2026-07-18", "2026-07-19"),
            "Sunday from Friday, before Saturday attendance is available",
        ),
        ParityCase(
            "P3",
            "2026-07-26",
            "2026-07-20",
            2,
            "sun",
            ("2026-07-25", "2026-07-26"),
            "Two-service-ahead Sunday forecast",
        ),
        ParityCase(
            "P4",
            "2026-08-01",
            "2026-07-17",
            5,
            "sat",
            (
                "2026-07-18",
                "2026-07-19",
                "2026-07-25",
                "2026-07-26",
                "2026-08-01",
            ),
            "Five-service-ahead Saturday at the supported 15-day limit",
        ),
        ParityCase(
            "P5",
            "2023-01-29",
            "2023-01-27",
            2,
            "sun",
            ("2023-01-28", "2023-01-29"),
            "Natural early-history case with missing matching-slot features",
        ),
        ParityCase(
            "P6",
            "2026-07-18",
            "2026-07-17",
            1,
            "sat",
            ("2026-07-18",),
            "Deliberately corrupted feature-contract fixture",
            valid=False,
        ),
    )


def _verify_checksum_manifest(candidate_dir: Path) -> dict[str, Any]:
    manifest_path = candidate_dir / "checksums.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Candidate checksum manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Candidate checksum manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Candidate checksum manifest must be a JSON mapping")
    if manifest.get("algorithm") != "sha256":
        raise ValueError("Candidate checksum manifest must use sha256")
    expected_files = {"model_package.joblib", "metadata.json"}
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError(
            "Candidate checksum manifest must cover model_package.joblib and metadata.json"
        )
    for filename, expected in files.items():
        path = candidate_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Candidate manifest file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Candidate checksum mismatch for {filename}: {actual} != {expected}"
            )
    return manifest


def verify_candidate_directory(
    candidate_dir: str | Path,
    *,
    expected_package_sha256: str = EXPECTED_CANDIDATE_PACKAGE_SHA256,
) -> tuple[VisitorPredictor, dict[str, Any]]:
    directory = Path(candidate_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Candidate package directory is missing: {directory}")
    manifest = _verify_checksum_manifest(directory)
    metadata_path = directory / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Candidate metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Candidate metadata must be a JSON mapping")
    package_id = metadata.get("package_id")
    if package_id != directory.name:
        raise ValueError(
            "Candidate metadata package_id does not match its versioned directory"
        )
    if metadata.get("package_status") != "candidate_not_active":
        raise ValueError("Candidate metadata does not mark the package inactive")
    if metadata.get("model_package_schema_version") != MODEL_PACKAGE_SCHEMA_VERSION:
        raise ValueError("Candidate metadata schema version is not 2")
    contract = load_tracked_f6_contract()
    if metadata.get("feature_contract") != contract:
        raise ValueError("Candidate metadata feature contract differs from tracked F6")
    if metadata.get("recommendation_policy_id") != contract["recommendation_policy_id"]:
        raise ValueError("Candidate metadata recommendation policy differs from tracked F6")
    training = metadata.get("training")
    if not isinstance(training, dict):
        raise ValueError("Candidate metadata training section is missing")
    for field in ("training_window_id", "sample_weight_id", "segmentation"):
        if training.get(field) != contract[field]:
            raise ValueError(f"Candidate metadata training {field} differs from tracked F6")
    attendance = metadata.get("attendance_input")
    if not isinstance(attendance, dict):
        raise ValueError("Candidate metadata attendance input is missing")
    if attendance.get("maximum_service_date") != "2026-07-12":
        raise ValueError("Candidate metadata attendance history does not end on 2026-07-12")
    if metadata.get("activation") != {
        "active_model_changed": False,
        "requires_separate_activation_stage": True,
    }:
        raise ValueError("Candidate metadata does not preserve inactive activation state")
    tracked_contract = metadata.get("tracked_contract")
    if not isinstance(tracked_contract, dict):
        raise ValueError("Candidate metadata tracked-contract section is missing")
    if tracked_contract.get("path") != "config/model_contracts/f6_v1.json":
        raise ValueError("Candidate metadata tracked-contract path is incorrect")
    if tracked_contract.get("sha256") != sha256_file(TRACKED_F6_CONTRACT):
        raise ValueError("Candidate metadata tracked-contract hash is incorrect")
    package_path = directory / "model_package.joblib"
    package_hash = sha256_file(package_path)
    if package_hash != expected_package_sha256:
        raise ValueError(
            f"Candidate package SHA-256 mismatch: {package_hash} != {expected_package_sha256}"
        )
    if manifest["files"]["model_package.joblib"] != expected_package_sha256:
        raise ValueError("Candidate checksum manifest does not contain the locked package hash")

    predictor = VisitorPredictor(str(package_path))
    if predictor.package_id != package_id:
        raise ValueError("Loaded package ID differs from candidate metadata")
    if predictor.model_package_schema_version != MODEL_PACKAGE_SCHEMA_VERSION:
        raise ValueError("Loaded candidate schema version is not 2")
    if predictor.feature_contract != contract:
        raise ValueError("Loaded candidate contract differs from the tracked F6 contract")
    if len(predictor.feature_cols) != 33:
        raise ValueError("Loaded candidate does not contain exactly 33 features")
    if feature_order_sha256(predictor.feature_cols) != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise ValueError("Loaded candidate feature-order hash is not the locked F6 hash")
    if predictor.recommendation_policy_id != RECOMMENDATION_POLICY_ID:
        raise ValueError("Loaded candidate recommendation policy is not locked C0")
    history_dates = pd.to_datetime(predictor.history_df[DATE_COL], errors="raise")
    if history_dates.max().date().isoformat() != "2026-07-12":
        raise ValueError("Candidate attendance history does not end on 2026-07-12")

    return predictor, {
        "candidate_directory": str(directory),
        "package_id": package_id,
        "package_status": metadata["package_status"],
        "schema_version": predictor.model_package_schema_version,
        "package_sha256": package_hash,
        "metadata_sha256": sha256_file(metadata_path),
        "checksums_sha256": sha256_file(directory / "checksums.json"),
        "checksums_valid": True,
        "feature_set_id": contract["feature_set_id"],
        "feature_count": len(predictor.feature_cols),
        "feature_order_sha256": feature_order_sha256(predictor.feature_cols),
        "training_window_id": contract["training_window_id"],
        "sample_weight_id": contract["sample_weight_id"],
        "weekday_policy": contract["weekday_policy"],
        "weather_policy": contract["weather_policy"],
        "recommendation_policy_id": predictor.recommendation_policy_id,
        "model_segments": sorted(predictor.models),
        "quantile_segments": sorted(predictor.quantile_models),
        "preprocessor_segments": sorted(predictor.preprocessors),
        "history_row_count": int(len(predictor.history_df)),
        "history_minimum_date": history_dates.min().date().isoformat(),
        "history_maximum_date": history_dates.max().date().isoformat(),
        "source_csv_required_for_load": False,
        "research_artifacts_required_for_load": False,
    }


def _raw_comparison(
    case: ParityCase,
    direct: pd.DataFrame,
    production: pd.DataFrame,
) -> tuple[tuple[dict[str, Any], ...], float]:
    if list(direct.columns) != list(production.columns):
        raise AssertionError(f"{case.case_id}: raw feature names or order differ")
    if [str(dtype) for dtype in direct.dtypes] != [
        str(dtype) for dtype in production.dtypes
    ]:
        raise AssertionError(f"{case.case_id}: raw feature dtypes differ")
    direct_values = direct.to_numpy(dtype=float)
    production_values = production.to_numpy(dtype=float)
    direct_missing = np.isnan(direct_values)
    production_missing = np.isnan(production_values)
    if not np.array_equal(direct_missing, production_missing):
        raise AssertionError(f"{case.case_id}: raw feature missingness masks differ")
    if np.isinf(direct_values).any() or np.isinf(production_values).any():
        raise AssertionError(f"{case.case_id}: raw features contain infinity")
    finite = ~direct_missing
    differences = np.zeros_like(direct_values, dtype=float)
    differences[finite] = np.abs(direct_values[finite] - production_values[finite])
    maximum = float(differences.max(initial=0.0))
    if maximum > RAW_TOLERANCE:
        raise AssertionError(
            f"{case.case_id}: maximum raw feature difference {maximum} exceeds "
            f"{RAW_TOLERANCE}"
        )
    rows = []
    for index, feature in enumerate(direct.columns):
        direct_value = float(direct_values[0, index])
        production_value = float(production_values[0, index])
        missing = bool(direct_missing[0, index])
        rows.append(
            {
                "case_id": case.case_id,
                "feature_index": index,
                "feature": feature,
                "direct_dtype": str(direct.dtypes.iloc[index]),
                "predictor_dtype": str(production.dtypes.iloc[index]),
                "direct_value": None if missing else direct_value,
                "predictor_value": None if missing else production_value,
                "direct_missing": missing,
                "predictor_missing": bool(production_missing[0, index]),
                "absolute_difference": float(differences[0, index]),
                "within_tolerance": bool(differences[0, index] <= RAW_TOLERANCE),
            }
        )
    return tuple(rows), maximum


def _transformed_comparison(
    case: ParityCase,
    direct: np.ndarray,
    production: np.ndarray,
    feature_names: list[str],
) -> tuple[tuple[dict[str, Any], ...], float]:
    direct = np.asarray(direct, dtype=float)
    production = np.asarray(production, dtype=float)
    if direct.shape != production.shape:
        raise AssertionError(f"{case.case_id}: transformed shapes differ")
    if direct.shape != (1, 33):
        raise AssertionError(
            f"{case.case_id}: transformed shape is {direct.shape}, expected (1, 33)"
        )
    if not np.isfinite(direct).all() or not np.isfinite(production).all():
        raise AssertionError(f"{case.case_id}: transformed features are non-finite")
    differences = np.abs(direct - production)
    maximum = float(differences.max(initial=0.0))
    if maximum > TRANSFORMED_TOLERANCE:
        raise AssertionError(
            f"{case.case_id}: maximum transformed difference {maximum} exceeds "
            f"{TRANSFORMED_TOLERANCE}"
        )
    rows = tuple(
        {
            "case_id": case.case_id,
            "transformed_index": index,
            "feature": feature_names[index],
            "direct_value": float(direct[0, index]),
            "predictor_value": float(production[0, index]),
            "absolute_difference": float(differences[0, index]),
            "within_tolerance": bool(
                differences[0, index] <= TRANSFORMED_TOLERANCE
            ),
        }
        for index in range(direct.shape[1])
    )
    return rows, maximum


def _available_history_maximum(
    history: pd.DataFrame,
    origin: pd.Timestamp,
) -> str:
    dates = pd.to_datetime(history[DATE_COL], errors="raise").dt.normalize()
    available = dates[dates <= origin]
    return available.max().date().isoformat() if not available.empty else ""


def evaluate_parity_case(
    predictor: VisitorPredictor,
    case: ParityCase,
) -> ParityEvaluation:
    if not case.valid:
        raise ValueError(f"{case.case_id} is an invalid-fixture case")
    direct_result = build_locked_f6_feature_result(
        predictor.history_df,
        case.target,
        case.origin,
        service_horizon=case.service_horizon,
    )
    if len(direct_result.provenance) != len(predictor.feature_cols):
        raise AssertionError(f"{case.case_id}: feature provenance is incomplete")
    hidden_dates = set(case.expected_hidden_dates)
    used_attendance_dates: set[str] = set()
    for item in direct_result.provenance:
        if item.get("origin_valid") is not True:
            raise AssertionError(
                f"{case.case_id}: {item.get('feature')} has invalid origin provenance"
            )
        if item.get("forecast_origin") != case.forecast_origin:
            raise AssertionError(
                f"{case.case_id}: {item.get('feature')} records the wrong origin"
            )
        if item.get("target_date") != case.target_date:
            raise AssertionError(
                f"{case.case_id}: {item.get('feature')} records the wrong target"
            )
        if item.get("source_type") != "attendance":
            continue
        source_dates = {str(value) for value in item.get("source_dates", [])}
        used_attendance_dates.update(source_dates)
        if source_dates.intersection(hidden_dates):
            raise AssertionError(
                f"{case.case_id}: hidden attendance appears in feature provenance"
            )
        if any(pd.Timestamp(value) > case.origin for value in source_dates):
            raise AssertionError(
                f"{case.case_id}: post-origin attendance appears in provenance"
            )
    direct_raw = direct_result.features.to_frame().T
    with patch("src.config.forecast_today", return_value=case.origin.date()):
        production_raw = predictor._prepare_one_row(case.target)
        prediction = predictor.predict_next(case.target_date, meal_buffer_pct=0.30)
    raw_rows, maximum_raw = _raw_comparison(case, direct_raw, production_raw)

    segment = predictor._segment_for_date(case.target)
    if segment != case.segment:
        raise AssertionError(
            f"{case.case_id}: selected segment {segment} differs from {case.segment}"
        )
    preprocessor = predictor.preprocessors[segment]
    direct_transformed = preprocessor.transform(direct_raw)
    production_transformed = preprocessor.transform(production_raw)
    transformed_rows, maximum_transformed = _transformed_comparison(
        case,
        direct_transformed,
        production_transformed,
        list(predictor.feature_cols),
    )
    direct_point = float(
        predictor.models[segment].predict(direct_transformed)[0]
    )
    direct_quantile = float(
        predictor.quantile_models[segment].predict(direct_transformed)[0]
    )
    if not np.isfinite(direct_point) or not np.isfinite(direct_quantile):
        raise AssertionError(f"{case.case_id}: direct predictions are non-finite")
    if direct_point < 0 or direct_quantile < 0:
        raise AssertionError(f"{case.case_id}: direct predictions are negative")
    point_difference = abs(direct_point - prediction.predicted_visitors)
    quantile_difference = abs(direct_quantile - prediction.predicted_quantile)
    if point_difference > PREDICTION_TOLERANCE:
        raise AssertionError(f"{case.case_id}: point prediction parity failed")
    if quantile_difference > PREDICTION_TOLERANCE:
        raise AssertionError(f"{case.case_id}: quantile prediction parity failed")
    if prediction.model_segment != segment:
        raise AssertionError(f"{case.case_id}: prediction segment differs")
    expected_meals = math.ceil(direct_quantile)
    if prediction.suggested_meals != expected_meals:
        raise AssertionError(f"{case.case_id}: C0 recommendation is not ceil(q80)")
    if prediction.meal_buffer_pct != 0.0 or prediction.residual_buffer != 0.0:
        raise AssertionError(f"{case.case_id}: F6 buffers are not disabled")
    history_dates = pd.to_datetime(
        predictor.history_df[DATE_COL], errors="raise"
    ).dt.normalize()
    attendance_values = pd.to_numeric(
        predictor.history_df.loc[history_dates <= case.origin, TARGET_COL],
        errors="coerce",
    ).dropna()
    if attendance_values.empty:
        raise AssertionError(f"{case.case_id}: no attendance is available at origin")
    history_minimum = float(attendance_values.min())
    history_maximum = float(attendance_values.max())
    point_operationally_plausible = (
        0.0 <= direct_point <= OPERATIONAL_ATTENDANCE_MAXIMUM
    )
    quantile_operationally_plausible = (
        0.0 <= direct_quantile <= OPERATIONAL_ATTENDANCE_MAXIMUM
    )
    if not point_operationally_plausible or not quantile_operationally_plausible:
        raise AssertionError(
            f"{case.case_id}: prediction falls outside the operational attendance range"
        )

    missing_features = [
        row["feature"] for row in raw_rows if row["direct_missing"]
    ]
    registry_row = {
        **asdict(case),
        "expected_hidden_dates": "|".join(case.expected_hidden_dates),
        "calendar_days_ahead": case.calendar_days_ahead,
        "available_history_maximum_date": _available_history_maximum(
            predictor.history_df, case.origin
        ),
        "package_id": predictor.package_id,
        "schema_version": predictor.model_package_schema_version,
        "missing_feature_count": len(missing_features),
        "missing_features": "|".join(missing_features),
        "all_provenance_origin_valid": True,
        "hidden_dates_absent_from_provenance": True,
        "attendance_provenance_maximum_date": (
            max(used_attendance_dates) if used_attendance_dates else ""
        ),
    }
    prediction_row = {
        "case_id": case.case_id,
        "target_date": case.target_date,
        "forecast_origin": case.forecast_origin,
        "calendar_days_ahead": case.calendar_days_ahead,
        "service_horizon": case.service_horizon,
        "segment": segment,
        "direct_point_prediction": direct_point,
        "predictor_point_prediction": prediction.predicted_visitors,
        "point_absolute_difference": point_difference,
        "direct_quantile_prediction": direct_quantile,
        "predictor_quantile_prediction": prediction.predicted_quantile,
        "quantile_absolute_difference": quantile_difference,
        "suggested_meals": prediction.suggested_meals,
        "expected_c0_meals": expected_meals,
        "meal_buffer_pct": prediction.meal_buffer_pct,
        "residual_buffer": prediction.residual_buffer,
        "package_id": predictor.package_id,
        "schema_version": predictor.model_package_schema_version,
        "recommendation_policy_id": predictor.recommendation_policy_id,
        "origin_available_history_minimum_attendance": history_minimum,
        "origin_available_history_maximum_attendance": history_maximum,
        "operational_attendance_maximum": OPERATIONAL_ATTENDANCE_MAXIMUM,
        "point_operationally_plausible": point_operationally_plausible,
        "quantile_operationally_plausible": quantile_operationally_plausible,
        "maximum_raw_feature_difference": maximum_raw,
        "maximum_transformed_difference": maximum_transformed,
        "passed": True,
    }
    return ParityEvaluation(
        registry_row=registry_row,
        raw_feature_rows=raw_rows,
        transformed_rows=transformed_rows,
        prediction_row=prediction_row,
        provenance=direct_result.provenance,
    )


def sunday_leakage_verification(
    predictor: VisitorPredictor,
    case: ParityCase,
) -> dict[str, Any]:
    if case.case_id != "P2":
        raise ValueError("Sunday leakage verification requires parity case P2")
    base_history = predictor.history_df.copy()
    sentinel_date = pd.Timestamp("2026-07-18")
    sentinel = pd.DataFrame(
        [{DATE_COL: sentinel_date, TARGET_COL: 9999.0}]
    )
    augmented = pd.concat([base_history, sentinel], ignore_index=True)
    masked = augmented.copy()
    masked.loc[masked[DATE_COL] == sentinel_date, TARGET_COL] = np.nan

    evaluations: dict[str, ParityEvaluation] = {}
    for label, history in (
        ("base", base_history),
        ("saturday_sentinel", augmented),
        ("saturday_masked", masked),
    ):
        variant = copy(predictor)
        variant.history_df = history
        evaluations[label] = evaluate_parity_case(variant, case)

    base_raw = pd.DataFrame(evaluations["base"].raw_feature_rows)
    base_prediction = evaluations["base"].prediction_row
    maximum_raw_difference = 0.0
    maximum_point_difference = 0.0
    maximum_quantile_difference = 0.0
    for label in ("saturday_sentinel", "saturday_masked"):
        current_raw = pd.DataFrame(evaluations[label].raw_feature_rows)
        direct_left = pd.to_numeric(base_raw["direct_value"], errors="coerce")
        direct_right = pd.to_numeric(current_raw["direct_value"], errors="coerce")
        if not np.array_equal(direct_left.isna(), direct_right.isna()):
            raise AssertionError(f"P2 leakage: missingness changed for {label}")
        finite = direct_left.notna()
        difference = np.abs(
            direct_left[finite].to_numpy(float)
            - direct_right[finite].to_numpy(float)
        )
        maximum_raw_difference = max(
            maximum_raw_difference,
            float(difference.max(initial=0.0)),
        )
        row = evaluations[label].prediction_row
        maximum_point_difference = max(
            maximum_point_difference,
            abs(
                base_prediction["predictor_point_prediction"]
                - row["predictor_point_prediction"]
            ),
        )
        maximum_quantile_difference = max(
            maximum_quantile_difference,
            abs(
                base_prediction["predictor_quantile_prediction"]
                - row["predictor_quantile_prediction"]
            ),
        )
        if row["suggested_meals"] != base_prediction["suggested_meals"]:
            raise AssertionError(f"P2 leakage: recommendation changed for {label}")

    attendance_provenance = [
        item
        for item in evaluations["saturday_sentinel"].provenance
        if item["source_type"] == "attendance"
    ]
    all_source_dates = sorted(
        {
            source_date
            for item in attendance_provenance
            for source_date in item["source_dates"]
        }
    )
    if sentinel_date.date().isoformat() in all_source_dates:
        raise AssertionError("P2 leakage: Saturday sentinel appears in provenance")
    if any(pd.Timestamp(value) > case.origin for value in all_source_dates):
        raise AssertionError("P2 leakage: post-origin attendance appears in provenance")
    source_weekdays = sorted({pd.Timestamp(value).day_name() for value in all_source_dates})
    if source_weekdays and source_weekdays != ["Sunday"]:
        raise AssertionError("P2 leakage: non-Sunday attendance sources were used")
    if maximum_raw_difference > RAW_TOLERANCE:
        raise AssertionError("P2 leakage: raw features changed")
    if maximum_point_difference > PREDICTION_TOLERANCE:
        raise AssertionError("P2 leakage: point prediction changed")
    if maximum_quantile_difference > PREDICTION_TOLERANCE:
        raise AssertionError("P2 leakage: quantile prediction changed")
    if base_prediction["service_horizon"] != 2:
        raise AssertionError("P2 leakage: service horizon is not 2")

    return {
        "case_id": case.case_id,
        "forecast_origin": case.forecast_origin,
        "target_date": case.target_date,
        "intervening_saturday": sentinel_date.date().isoformat(),
        "sentinel_value": 9999.0,
        "service_horizon": base_prediction["service_horizon"],
        "all_attendance_source_dates": all_source_dates,
        "attendance_source_weekdays": source_weekdays,
        "saturday_in_provenance": False,
        "post_origin_source_count": 0,
        "maximum_raw_difference_after_saturday_append_or_mask": maximum_raw_difference,
        "maximum_point_difference_after_saturday_append_or_mask": maximum_point_difference,
        "maximum_quantile_difference_after_saturday_append_or_mask": maximum_quantile_difference,
        "recommendation_stable": True,
        "passed": True,
    }


def prediction_signature(prediction: PredictionOutput) -> dict[str, Any]:
    return {
        "service_date": prediction.service_date.date().isoformat(),
        "predicted_visitors": prediction.predicted_visitors,
        "predicted_quantile": prediction.predicted_quantile,
        "residual_buffer": prediction.residual_buffer,
        "suggested_meals": prediction.suggested_meals,
        "meal_buffer_pct": prediction.meal_buffer_pct,
        "model_segment": prediction.model_segment,
    }
