"""Isolated weather challenger. Its historical bootstrap is not forecast validation."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from numbers import Real
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL
from src.location_config import get_location
from src.modeling import fit_final_models_by_daytype
from src.production_features import (
    LOCKED_F6_FEATURES,
    RECOMMENDATION_POLICY_ID,
    build_locked_f6_feature_row,
    build_locked_f6_training_frame,
    feature_order_sha256,
    locked_feature_contract_metadata,
    normalize_t1_history,
    validate_locked_f6_feature_order,
)

WEATHER_FEATURES = (
    "wx_apparent_min_c", "wx_apparent_max_c", "wx_precip_mm",
    "wx_gust_max_kmh", "wx_snowfall_cm", "wx_snowdepth_max_m",
)
WEATHER_CANDIDATE_FEATURES = (*LOCKED_F6_FEATURES, *WEATHER_FEATURES)
WEATHER_CANDIDATE_FEATURE_ORDER_SHA256 = feature_order_sha256(WEATHER_CANDIDATE_FEATURES)
PACKAGE_KIND = "weather_shadow_candidate"
PACKAGE_STATUS = "shadow_only_not_validated"
TRAINING_WEATHER = "realized_historical_bootstrap"
MODEL_PACKAGE_NAME = "weather_candidate.joblib"
MIN_TRAIN_SIZE_PER_SEGMENT = 18


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_sha256(history: pd.DataFrame) -> str:
    normalized = normalize_t1_history(history)
    records = [[pd.Timestamp(row[DATE_COL]).date().isoformat(), float(row[TARGET_COL])]
               for _, row in normalized.iterrows()]
    return hashlib.sha256(json.dumps(records, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def weather_candidate_contract() -> dict[str, Any]:
    return {
        "feature_set_id": "F6_WEATHER_SHADOW_V1",
        "feature_contract_version": "f6_weather_shadow_v1",
        "ordered_feature_list": list(WEATHER_CANDIDATE_FEATURES),
        "feature_order_sha256": WEATHER_CANDIDATE_FEATURE_ORDER_SHA256,
        "base_feature_contract": locked_feature_contract_metadata(),
        "weather_feature_list": list(WEATHER_FEATURES),
        "weather_window_local": "11:00-13:00",
        "weather_interval_semantics": {
            "apparent_temperature_and_snow_depth": "instantaneous timestamps 11:00,12:00,13:00",
            "precipitation_snowfall_gusts": "preceding-hour timestamps 12:00,13:00",
        },
    }


def _weather_values(features: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(features, Mapping):
        raise ValueError("weather_features must be a mapping")
    missing = set(WEATHER_FEATURES).difference(features)
    if missing:
        raise ValueError(f"Missing weather features: {sorted(missing)}")
    if any(isinstance(features[name], (bool, np.bool_)) or not isinstance(features[name], Real) for name in WEATHER_FEATURES):
        raise ValueError("Weather features must be numeric and finite")
    try:
        result = {name: float(features[name]) for name in WEATHER_FEATURES}
    except (TypeError, ValueError) as exc:
        raise ValueError("Weather features must be numeric and finite") from exc
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Weather features must be numeric and finite")
    if any(result[name] < 0 for name in WEATHER_FEATURES[2:]):
        raise ValueError("Precipitation, gusts, snowfall and snow depth cannot be negative")
    if result[WEATHER_FEATURES[0]] > result[WEATHER_FEATURES[1]]:
        raise ValueError("Minimum apparent temperature exceeds maximum")
    return result


def combine_weather_features(x_f6: pd.DataFrame, weather_features: Mapping[str, Any]) -> pd.DataFrame:
    """Append weather to the exact raw F6 row used for the paired baseline."""
    if not isinstance(x_f6, pd.DataFrame) or x_f6.shape != (1, len(LOCKED_F6_FEATURES)):
        raise ValueError("Expected exactly one raw F6 feature row with 33 columns")
    validate_locked_f6_feature_order(list(x_f6.columns))
    try:
        row = x_f6.astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("F6 features must be numeric") from exc
    if np.isinf(row.to_numpy()).any():
        raise ValueError("F6 features cannot contain infinity")
    for name, value in _weather_values(weather_features).items():
        row[name] = value
    return row.loc[:, list(WEATHER_CANDIDATE_FEATURES)]


def build_weather_candidate_feature_row(
    attendance_history: pd.DataFrame, target_date: Any, forecast_origin: Any,
    weather_features: Mapping[str, Any],
) -> pd.DataFrame:
    return combine_weather_features(
        build_locked_f6_feature_row(attendance_history, target_date, forecast_origin),
        weather_features,
    )


def build_weather_candidate_package(
    *, location_id: str, attendance: pd.DataFrame, weather_features: pd.DataFrame,
    package_id: str, baseline_source: Mapping[str, Any], weather_input: Mapping[str, Any],
    weather_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Train on the locked F6 rows, with complete realized weather for every row.

    No performance claim is inferred from this fit. Only subsequently captured
    forecasts and predictions can enter the prospective paired evaluation.
    """
    get_location(location_id)
    bundle = build_locked_f6_training_frame(attendance)
    required = {DATE_COL, *WEATHER_FEATURES}
    if not isinstance(weather_features, pd.DataFrame) or not required.issubset(weather_features.columns):
        raise ValueError("Historical weather frame must contain service_date and all six weather features")
    weather = weather_features.loc[:, [DATE_COL, *WEATHER_FEATURES]].copy()
    weather[DATE_COL] = pd.to_datetime(weather[DATE_COL], errors="raise").dt.normalize()
    if weather[DATE_COL].isna().any() or weather[DATE_COL].duplicated().any():
        raise ValueError("Historical weather dates must be present and unique")
    frame = bundle.df.merge(weather, on=DATE_COL, how="left", validate="one_to_one")
    if frame.empty:
        raise ValueError("No eligible F6 training rows")
    for _, row in frame.iterrows():
        try:
            _weather_values(row.to_dict())
        except ValueError as exc:
            raise ValueError(f"Invalid historical weather for {row[DATE_COL].date()}: {exc}") from exc
    counts = {segment: int((frame["is_sun"] == flag).sum()) for segment, flag in (("sat", 0), ("sun", 1))}
    if min(counts.values()) < MIN_TRAIN_SIZE_PER_SEGMENT:
        raise ValueError(f"Weather candidate needs at least {MIN_TRAIN_SIZE_PER_SEGMENT} rows per segment: {counts}")
    if np.isinf(frame.loc[:, list(WEATHER_CANDIDATE_FEATURES)].to_numpy(dtype=float)).any():
        raise ValueError("Training features cannot contain infinity")
    models, quantile_models, preprocessors = fit_final_models_by_daytype(
        frame, list(WEATHER_CANDIDATE_FEATURES), quantile=0.8, return_preprocessors=True,
    )
    # This timestamp is recorded after fitting; backdated availability is prohibited.
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "package_kind": PACKAGE_KIND,
        "candidate_schema_version": 1,
        "package_id": package_id,
        "package_status": PACKAGE_STATUS,
        "created_at_utc": created_at,
        "location_id": location_id,
        "baseline_model_sha256": baseline_source.get("sha256"),
        "training_end_date": bundle.history_df[DATE_COL].max().date().isoformat(),
        "feature_order_sha256": WEATHER_CANDIDATE_FEATURE_ORDER_SHA256,
        "feature_contract": weather_candidate_contract(),
        "training_weather": TRAINING_WEATHER,
        "baseline_source": dict(baseline_source),
        "weather_input": dict(weather_input),
        "weather_context": dict(weather_context),
        "history": {
            "sha256": history_sha256(bundle.history_df),
            "row_count": len(bundle.history_df),
            "minimum_service_date": bundle.history_df[DATE_COL].min().date().isoformat(),
            "maximum_service_date": bundle.history_df[DATE_COL].max().date().isoformat(),
        },
        "training": {
            "entrypoint": "scripts/train_weather_candidate.py",
            "segmentation": "separate_saturday_sunday",
            "training_window_id": "TW_EXPANDING",
            "sample_weight_id": "SW_UNIFORM",
            "row_count_by_segment": counts,
            "training_origin_policy": "locked_f6_previous_observed_service_date_service_horizon_1",
            "weather_availability_at_training_origin": "not_asserted_realized_historical_weather",
            "quantile": 0.8,
            "point_model_parameters": {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 2, "random_state": 42},
            "quantile_model_parameters": {"loss": "quantile", "quantile": 0.8, "learning_rate": 0.05, "max_depth": 4, "max_iter": 500, "random_state": 42},
            "preprocessing": "separate_segment_median_imputer_keep_empty_features",
        },
        "recommendation_policy_id": RECOMMENDATION_POLICY_ID,
        "activation": {"active_model_changed": False, "automatic_activation_allowed": False},
        "evaluation": {
            "status": "prospective_forecast_evaluation_required",
            "bootstrap_is_archived_forecast_validation": False,
            "prediction_must_be_saved_by_preparation_cutoff": True,
        },
    }
    package = {
        **metadata,
        # Production VisitorPredictor only accepts 1 or 2. Never impersonate F6.
        "model_package_schema_version": 0,
        "feature_cols": list(WEATHER_CANDIDATE_FEATURES),
        "models": models, "quantile_models": quantile_models, "preprocessors": preprocessors,
        "history_df": bundle.history_df.copy(), "metadata": metadata,
    }
    validate_weather_candidate(package, expected_location_id=location_id)
    return package


def validate_weather_candidate(package: Mapping[str, Any], *, expected_location_id: str | None = None) -> None:
    if not isinstance(package, Mapping):
        raise ValueError("Weather candidate package must be a mapping")
    for key, value in (("package_kind", PACKAGE_KIND), ("candidate_schema_version", 1),
                       ("package_status", PACKAGE_STATUS), ("training_weather", TRAINING_WEATHER),
                       ("model_package_schema_version", 0), ("recommendation_policy_id", RECOMMENDATION_POLICY_ID)):
        if package.get(key) != value:
            raise ValueError(f"Weather candidate {key} mismatch")
    package_id = package.get("package_id")
    if not isinstance(package_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*[-_]v[1-9][0-9]*", package_id):
        raise ValueError("Weather candidate package_id must be unique and explicitly versioned")
    if package.get("feature_cols") != list(WEATHER_CANDIDATE_FEATURES) or package.get("feature_contract") != weather_candidate_contract():
        raise ValueError("Weather candidate feature contract or ordered features mismatch")
    if expected_location_id is not None and package.get("location_id") != expected_location_id:
        raise ValueError("Weather candidate location does not match requested location")
    location = get_location(str(package.get("location_id")))
    context = package.get("weather_context")
    if not isinstance(context, Mapping) or any(context.get(key) != value for key, value in
            (("zip_code", location.zip_code), ("country_code", location.country_code), ("timezone", location.timezone))):
        raise ValueError("Weather candidate weather_context does not match location")
    try:
        latitude, longitude = float(context["latitude"]), float(context["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Weather candidate requires verified latitude and longitude") from exc
    if not np.isfinite([latitude, longitude]).all() or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Weather candidate coordinates must be finite and valid")
    created = pd.Timestamp(package.get("created_at_utc"))
    if pd.isna(created) or created.tzinfo is None:
        raise ValueError("Weather candidate created_at_utc must be timezone-aware")
    metadata = package.get("metadata")
    metadata_keys = set(package).difference({"model_package_schema_version", "feature_cols", "models", "quantile_models", "preprocessors", "history_df", "metadata"})
    if not isinstance(metadata, Mapping) or set(metadata) != metadata_keys or any(package.get(key) != value for key, value in metadata.items()):
        raise ValueError("Weather candidate metadata does not match package")
    for source in ("baseline_source", "weather_input", "history"):
        if not isinstance(package.get(source), Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(package[source].get("sha256"))):
            raise ValueError(f"Weather candidate {source} SHA-256 is missing or invalid")
    history = package.get("history_df")
    if not isinstance(history, pd.DataFrame) or history.empty or not {DATE_COL, TARGET_COL}.issubset(history.columns):
        raise ValueError("Weather candidate attendance history is empty or invalid")
    normalized_history = normalize_t1_history(history)
    if normalized_history.empty or len(normalized_history) != len(history):
        raise ValueError("Weather candidate history must contain only eligible service dates")
    if not np.isfinite(normalized_history[TARGET_COL]).all() or (normalized_history[TARGET_COL] < 0).any():
        raise ValueError("Weather candidate history attendance must be finite and nonnegative")
    expected_history = {"sha256": history_sha256(history), "row_count": len(normalized_history),
                        "minimum_service_date": normalized_history[DATE_COL].min().date().isoformat(),
                        "maximum_service_date": normalized_history[DATE_COL].max().date().isoformat()}
    if package.get("baseline_model_sha256") != package["baseline_source"]["sha256"] or package.get("feature_order_sha256") != WEATHER_CANDIDATE_FEATURE_ORDER_SHA256:
        raise ValueError("Weather candidate source or feature fingerprint mismatch")
    if package.get("training_end_date") != expected_history["maximum_service_date"]:
        raise ValueError("Weather candidate training_end_date mismatch")
    if package["history"] != expected_history:
        raise ValueError("Weather candidate history checksum or dates mismatch")
    if package.get("activation") != {"active_model_changed": False, "automatic_activation_allowed": False}:
        raise ValueError("Weather candidate must remain inactive")
    for key in ("models", "quantile_models", "preprocessors"):
        group = package.get(key)
        if not isinstance(group, Mapping) or set(group) != {"sat", "sun"}:
            raise ValueError(f"Weather candidate requires exactly Saturday and Sunday {key}")
        for obj in group.values():
            if getattr(obj, "n_features_in_", None) != len(WEATHER_CANDIDATE_FEATURES):
                raise ValueError(f"Weather candidate {key} feature dimension mismatch")
            method = "transform" if key == "preprocessors" else "predict"
            if not callable(getattr(obj, method, None)):
                raise ValueError(f"Weather candidate {key} lacks {method}")
            if key == "preprocessors":
                stats = np.asarray(getattr(obj, "statistics_", []), dtype=float)
                if stats.shape != (len(WEATHER_CANDIDATE_FEATURES),) or not np.isfinite(stats).all():
                    raise ValueError("Weather candidate preprocessor statistics must be finite with 39 entries")
                names = getattr(obj, "feature_names_in_", None)
                if names is not None and list(names) != list(WEATHER_CANDIDATE_FEATURES):
                    raise ValueError("Weather candidate preprocessor feature order mismatch")


def load_weather_candidate(path: str | Path, *, expected_location_id: str | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    directory = path if path.is_dir() else path.parent
    package_path = directory / MODEL_PACKAGE_NAME if path.is_dir() else path
    if package_path.name != MODEL_PACKAGE_NAME:
        raise ValueError(f"Expected separate weather candidate filename {MODEL_PACKAGE_NAME}")
    checksums = json.loads((directory / "checksums.json").read_text())
    if not isinstance(checksums, dict) or checksums.get("algorithm") != "sha256" or not isinstance(checksums.get("files"), dict):
        raise ValueError("Weather candidate checksum manifest must be a SHA-256 mapping")
    for name in (MODEL_PACKAGE_NAME, "metadata.json"):
        if checksums["files"].get(name) != sha256_file(directory / name):
            raise ValueError(f"Weather candidate checksum mismatch for {name}")
    # joblib artifacts are executable: only load locally trained, trusted packages.
    package = joblib.load(package_path)
    validate_weather_candidate(package, expected_location_id=expected_location_id)
    if package["package_id"] != directory.name:
        raise ValueError("Weather candidate package_id must match its directory")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata != package["metadata"]:
        raise ValueError("Weather candidate metadata file does not match package")
    return package


def predict_weather_candidate(
    package: Mapping[str, Any], x_f6: pd.DataFrame,
    weather_features: Mapping[str, Any], target_date: Any,
) -> dict[str, Any]:
    validate_weather_candidate(package)
    target = pd.Timestamp(target_date)
    if pd.isna(target) or target.weekday() not in {5, 6}:
        raise ValueError("target_date must be Saturday or Sunday")
    segment = "sun" if target.weekday() == 6 else "sat"
    row = combine_weather_features(x_f6, weather_features)
    if float(row["is_sun"].iloc[0]) != float(target.weekday() == 6):
        raise ValueError("Paired F6 row daytype does not match target_date")
    transformed = np.asarray(package["preprocessors"][segment].transform(row), dtype=float)
    if transformed.shape != (1, len(WEATHER_CANDIDATE_FEATURES)) or not np.isfinite(transformed).all():
        raise ValueError("Weather candidate has invalid or non-finite transformed features")
    values = []
    for key in ("models", "quantile_models"):
        predicted = np.asarray(package[key][segment].predict(transformed), dtype=float)
        if predicted.shape != (1,) or not np.isfinite(predicted).all():
            raise ValueError("Weather candidate predictions must be single finite values")
        values.append(float(predicted[0]))
    point, q80 = values
    return {
        "point": point, "q80": q80, "suggested_meals": int(np.ceil(q80)),
        "package_id": package["package_id"], "package_kind": PACKAGE_KIND,
        "package_status": PACKAGE_STATUS, "model_segment": segment,
        "feature_order_sha256": WEATHER_CANDIDATE_FEATURE_ORDER_SHA256,
        "recommendation_policy_id": RECOMMENDATION_POLICY_ID,
        "training_weather": TRAINING_WEATHER,
    }
