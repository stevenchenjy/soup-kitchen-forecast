from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATE_COL, PROJECT_ROOT, TARGET_COL, model_file_for_location
from src.location_config import get_location
from src.modeling import fit_final_models_by_daytype, rolling_backtest_by_daytype
from src.production_features import (
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
    TRACKED_F6_CONTRACT,
    build_locked_f6_training_frame,
    locked_feature_contract_metadata,
)


MODEL_PACKAGE_NAME = "model_package.joblib"
METADATA_NAME = "metadata.json"
CHECKSUMS_NAME = "checksums.json"
LOCKED_MIN_TRAIN_SIZE = 18
LOCKED_QUANTILE = 0.8
_PACKAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_package_dir(
    location_id: str,
    output_dir: str | Path,
    package_id: str,
) -> Path:
    """Resolve and validate a candidate destination without creating it."""

    get_location(location_id)
    if not _PACKAGE_ID_PATTERN.fullmatch(package_id):
        raise ValueError(
            "package_id must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )

    output_root = Path(output_dir).expanduser().resolve()
    destination = output_root / package_id
    active_model = model_file_for_location(location_id).resolve()
    active_models_dir = (PROJECT_ROOT / "models").resolve()
    if (
        output_root == active_model
        or destination == active_model
        or destination / MODEL_PACKAGE_NAME == active_model
        or output_root == active_models_dir
        or active_models_dir in output_root.parents
    ):
        raise ValueError(
            "F6 candidate output must be outside the active models directory and "
            f"cannot target {active_model}"
        )
    if destination.exists():
        raise FileExistsError(f"Candidate package already exists: {destination}")
    return destination


def load_validated_attendance_csv(path: str | Path) -> pd.DataFrame:
    attendance_path = Path(path).expanduser().resolve()
    if not attendance_path.is_file():
        raise FileNotFoundError(f"Attendance CSV does not exist: {attendance_path}")
    frame = pd.read_csv(attendance_path)
    missing = {DATE_COL, TARGET_COL}.difference(frame.columns)
    if missing:
        raise ValueError(f"Attendance CSV is missing columns: {sorted(missing)}")
    frame = frame[[DATE_COL, TARGET_COL]].copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise").dt.normalize()
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
    if frame.empty:
        raise ValueError("Attendance CSV is empty")
    if frame[DATE_COL].duplicated().any():
        raise ValueError("Attendance CSV contains duplicate service dates")
    if not np.isfinite(frame[TARGET_COL]).all() or (frame[TARGET_COL] < 0).any():
        raise ValueError("Attendance CSV visitors must be finite non-negative numbers")
    return frame.sort_values(DATE_COL, kind="stable").reset_index(drop=True)


def candidate_package_metadata(
    *,
    package_id: str,
    location_id: str,
    created_at_utc: str,
    attendance_csv: Path,
    attendance: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    contract = locked_feature_contract_metadata()
    return {
        "package_id": package_id,
        "package_status": "candidate_not_active",
        "created_at_utc": created_at_utc,
        "location_id": location_id,
        "model_package_schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
        "feature_contract": contract,
        "recommendation_policy_id": RECOMMENDATION_POLICY_ID,
        "training": {
            "entrypoint": "scripts/train_f6_candidate.py",
            "min_train_size_per_segment": LOCKED_MIN_TRAIN_SIZE,
            "quantile": LOCKED_QUANTILE,
            "training_window_id": contract["training_window_id"],
            "sample_weight_id": contract["sample_weight_id"],
            "segmentation": contract["segmentation"],
            "point_model_class": "RandomForestRegressor",
            "point_model_parameters": {
                "n_estimators": 400,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "quantile_model_class": "HistGradientBoostingRegressor",
            "quantile_model_parameters": {
                "loss": "quantile",
                "quantile": LOCKED_QUANTILE,
                "learning_rate": 0.05,
                "max_depth": 4,
                "max_iter": 500,
                "random_state": 42,
            },
        },
        "attendance_input": {
            "path": str(attendance_csv),
            "sha256": sha256_file(attendance_csv),
            "row_count": int(len(attendance)),
            "minimum_service_date": attendance[DATE_COL].min().date().isoformat(),
            "maximum_service_date": attendance[DATE_COL].max().date().isoformat(),
        },
        "tracked_contract": {
            "path": str(TRACKED_F6_CONTRACT.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(TRACKED_F6_CONTRACT),
        },
        "metrics": metrics,
        "activation": {
            "active_model_changed": False,
            "requires_separate_activation_stage": True,
        },
    }


def train_f6_candidate(
    *,
    location_id: str,
    attendance_csv: str | Path,
    output_dir: str | Path,
    package_id: str,
) -> Path:
    destination = candidate_package_dir(location_id, output_dir, package_id)
    attendance_path = Path(attendance_csv).expanduser().resolve()
    attendance = load_validated_attendance_csv(attendance_path)
    bundle = build_locked_f6_training_frame(attendance)
    if bundle.df.empty:
        raise ValueError(f"Not enough historical data to train location '{location_id}'.")

    outputs = rolling_backtest_by_daytype(
        bundle.df,
        bundle.feature_cols,
        min_train_size=LOCKED_MIN_TRAIN_SIZE,
        quantile=LOCKED_QUANTILE,
    )
    models, quantile_models, preprocessors = fit_final_models_by_daytype(
        bundle.df,
        bundle.feature_cols,
        quantile=LOCKED_QUANTILE,
        return_preprocessors=True,
    )
    required_segments = {"sat", "sun"}
    if set(models) != required_segments:
        raise ValueError("F6 candidate training must produce separate Saturday and Sunday models")
    if set(quantile_models) != required_segments or set(preprocessors) != required_segments:
        raise ValueError("F6 candidate package is missing a segment quantile model or preprocessor")

    metrics = {
        key: outputs.get(key, {}).get("metrics", {"BacktestRows": 0})
        for key in ("overall", "sat", "sun")
    }
    created_at_utc = datetime.now(timezone.utc).isoformat()
    location = get_location(location_id)
    package = {
        "package_id": package_id,
        "package_status": "candidate_not_active",
        "created_at_utc": created_at_utc,
        "location_id": location_id,
        "model_package_schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
        "models": models,
        "quantile_models": quantile_models,
        "preprocessors": preprocessors,
        "feature_cols": bundle.feature_cols,
        "feature_contract": locked_feature_contract_metadata(),
        "history_df": bundle.history_df.copy(),
        "recommendation_policy_id": RECOMMENDATION_POLICY_ID,
        "default_meal_buffer_pct": 0.0,
        "residual_buffer_by_day": {"sat": 0.0, "sun": 0.0},
        "preprocessing_contract": {
            "class": "SimpleImputer",
            "strategy": "median",
            "keep_empty_features": True,
            "scope": "separate_saturday_sunday_full_training_segment",
        },
        "weather_context": {
            "zip_code": location.zip_code,
            "country_code": location.country_code,
            "timezone": location.timezone,
        },
        "metrics": metrics,
    }
    metadata = candidate_package_metadata(
        package_id=package_id,
        location_id=location_id,
        created_at_utc=created_at_utc,
        attendance_csv=attendance_path,
        attendance=attendance,
        metrics=metrics,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_id}.tmp-", dir=destination.parent))
    destination_created = False
    try:
        package_path = temporary / MODEL_PACKAGE_NAME
        metadata_path = temporary / METADATA_NAME
        joblib.dump(package, package_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        checksums = {
            "algorithm": "sha256",
            "files": {
                MODEL_PACKAGE_NAME: sha256_file(package_path),
                METADATA_NAME: sha256_file(metadata_path),
            },
        }
        (temporary / CHECKSUMS_NAME).write_text(
            json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
        )
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"Candidate package already exists: {destination}") from exc
        destination_created = True
        for source in temporary.iterdir():
            source.rename(destination / source.name)
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an isolated schema-v2 F6 candidate package without activation."
    )
    parser.add_argument("--location", required=True, help="Location ID from data/locations.json")
    parser.add_argument("--attendance-csv", required=True, help="Explicit attendance CSV input")
    parser.add_argument("--output-dir", required=True, help="Candidate package root outside models/")
    parser.add_argument("--package-id", required=True, help="Unique versioned candidate package ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = train_f6_candidate(
        location_id=args.location,
        attendance_csv=args.attendance_csv,
        output_dir=args.output_dir,
        package_id=args.package_id,
    )
    print(f"F6 candidate package created without activation: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
