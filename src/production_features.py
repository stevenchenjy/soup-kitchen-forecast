"""Shared locked-F6 feature contract for production training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATE_COL, PROJECT_ROOT, TARGET_COL
from src.feature_sets import F6
from src.origin_features import (
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
    build_repaired_features_as_of,
)


LOCKED_F6_FEATURES: tuple[str, ...] = (
    "year_num",
    "month_num",
    "day_num",
    "weekday_num",
    "weekofyear",
    "is_weekend",
    "month_sin",
    "month_cos",
    "slot_num",
    "is_sun",
    "last_observed_daytype_1",
    "last_observed_daytype_2",
    "last_observed_daytype_4",
    "last_observed_daytype_6",
    "daytype_mean_last_2",
    "daytype_median_last_4",
    "daytype_median_last_6",
    "daytype_std_last_4",
    "daytype_recent_vs_previous_3",
    "daytype_mean2_minus_previous2",
    "daytype_slot_last_observed",
    "daytype_slot_match_count",
    "daytype_slot_days_since_latest",
    "calendar_days_ahead",
    "service_horizon",
    "observed_daytype_count",
    "future_eligible_services_between",
    "days_since_last_observed_daytype",
    "missing_last_observed_daytype_1",
    "missing_last_observed_daytype_2",
    "missing_last_observed_daytype_4",
    "missing_last_observed_daytype_6",
    "daytype_slot_history_missing",
)
LOCKED_F6_FEATURE_ORDER_SHA256 = (
    "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
)
LOCKED_FEATURE_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation"
    / "07_locked_feature_set.json"
)
MODEL_PACKAGE_SCHEMA_VERSION = 2
TRAINING_WINDOW_ID = "TW_EXPANDING"
SAMPLE_WEIGHT_ID = "SW_UNIFORM"
RECOMMENDATION_POLICY_ID = "C0_EXISTING_RAW_QUANTILE"
FEATURE_BUILDER_ID = "origin_features.build_repaired_features_as_of:v1"


@dataclass(frozen=True)
class ProductionFeatureBundle:
    df: pd.DataFrame
    feature_cols: list[str]
    history_df: pd.DataFrame


def feature_order_sha256(feature_cols: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(list(feature_cols), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_locked_f6_feature_order(
    feature_cols: list[str] | tuple[str, ...],
) -> list[str]:
    ordered = list(feature_cols)
    if ordered != list(LOCKED_F6_FEATURES):
        raise ValueError("Feature columns do not match the locked ordered F6 contract")
    actual_hash = feature_order_sha256(ordered)
    if actual_hash != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise ValueError(f"Locked F6 feature-order hash mismatch: {actual_hash}")
    return ordered


def validate_lock_artifact(path: str | Path = LOCKED_FEATURE_ARTIFACT) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("selected_feature_set_id") != F6:
        raise ValueError("Lock artifact does not select F6_COMPACT_SELECTED")
    validate_locked_f6_feature_order(payload.get("ordered_feature_list", []))
    if payload.get("feature_list_sha256") != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise ValueError("Lock artifact feature-order hash is not the production F6 hash")
    return payload


def locked_feature_contract_metadata() -> dict[str, Any]:
    validate_locked_f6_feature_order(LOCKED_F6_FEATURES)
    return {
        "feature_set_id": F6,
        "feature_order_sha256": LOCKED_F6_FEATURE_ORDER_SHA256,
        "ordered_feature_list": list(LOCKED_F6_FEATURES),
        "feature_builder_id": FEATURE_BUILDER_ID,
        "training_window_id": TRAINING_WINDOW_ID,
        "sample_weight_id": SAMPLE_WEIGHT_ID,
        "weekday_policy": T1_VALID_WEEKENDS,
        "weather_policy": W0_NO_WEATHER,
        "segmentation": "separate_saturday_sunday",
    }


def validate_package_feature_contract(
    package_contract: dict[str, Any],
    feature_cols: list[str] | tuple[str, ...],
) -> None:
    expected = locked_feature_contract_metadata()
    if package_contract.get("feature_set_id") != F6:
        raise ValueError("Unsupported production feature contract")
    if package_contract.get("feature_order_sha256") != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise ValueError("Model package F6 feature-order hash mismatch")
    validate_locked_f6_feature_order(feature_cols)
    if package_contract.get("ordered_feature_list") != expected["ordered_feature_list"]:
        raise ValueError("Model package ordered features differ from locked F6")
    for key in (
        "feature_builder_id",
        "training_window_id",
        "sample_weight_id",
        "weekday_policy",
        "weather_policy",
        "segmentation",
    ):
        if package_contract.get(key) != expected[key]:
            raise ValueError(f"Model package feature contract mismatch for {key}")


def normalize_t1_history(attendance_history: pd.DataFrame) -> pd.DataFrame:
    missing = {DATE_COL, TARGET_COL}.difference(attendance_history.columns)
    if missing:
        raise ValueError(f"Attendance history is missing columns: {sorted(missing)}")
    history = attendance_history[[DATE_COL, TARGET_COL]].copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL], errors="raise").dt.normalize()
    history[TARGET_COL] = pd.to_numeric(history[TARGET_COL], errors="raise").astype(float)
    history = history.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    if history[DATE_COL].duplicated().any():
        raise ValueError("Attendance history contains duplicate service dates")
    return history[history[DATE_COL].dt.weekday.isin([5, 6])].reset_index(drop=True)


def service_horizon_between(forecast_origin: Any, target_date: Any) -> int:
    origin = pd.Timestamp(forecast_origin).normalize()
    target = pd.Timestamp(target_date).normalize()
    if origin >= target:
        raise ValueError("forecast_origin must be earlier than target_date")
    dates = pd.date_range(origin + pd.Timedelta(days=1), target, freq="D")
    return int(sum(value.weekday() in {5, 6} for value in dates))


def build_locked_f6_feature_row(
    attendance_history: pd.DataFrame,
    target_date: Any,
    forecast_origin: Any,
    *,
    service_horizon: int | None = None,
) -> pd.DataFrame:
    target = pd.Timestamp(target_date).normalize()
    origin = pd.Timestamp(forecast_origin).normalize()
    horizon = (
        service_horizon_between(origin, target)
        if service_horizon is None
        else int(service_horizon)
    )
    result = build_repaired_features_as_of(
        normalize_t1_history(attendance_history),
        target,
        origin,
        calendar_days_ahead=int((target - origin).days),
        service_horizon=horizon,
        weekday_policy=T1_VALID_WEEKENDS,
        feature_cols=list(LOCKED_F6_FEATURES),
    )
    row = result.features.to_frame().T
    row.columns = validate_locked_f6_feature_order(list(row.columns))
    return row


def build_locked_f6_training_frame(
    attendance_history: pd.DataFrame,
) -> ProductionFeatureBundle:
    history = normalize_t1_history(attendance_history)
    rows: list[dict[str, Any]] = []
    for index in range(1, len(history)):
        target = pd.Timestamp(history.loc[index, DATE_COL])
        origin = pd.Timestamp(history.loc[index - 1, DATE_COL])
        features = build_locked_f6_feature_row(
            history,
            target,
            origin,
            service_horizon=1,
        ).iloc[0]
        row = features.to_dict()
        row.update(
            {
                DATE_COL: target,
                TARGET_COL: float(history.loc[index, TARGET_COL]),
                "training_origin": origin,
            }
        )
        rows.append(row)
    columns = [DATE_COL, TARGET_COL, "training_origin", *LOCKED_F6_FEATURES]
    frame = pd.DataFrame(rows, columns=columns)
    return ProductionFeatureBundle(
        df=frame,
        feature_cols=list(LOCKED_F6_FEATURES),
        history_df=history,
    )
