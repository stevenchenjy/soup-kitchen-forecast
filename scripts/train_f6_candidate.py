from __future__ import annotations

import argparse
import csv
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

from src.config import (
    DATE_COL,
    MODEL_FILE,
    PROJECT_ROOT,
    TARGET_COL,
    model_file_for_location,
)
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
_VERSIONED_PACKAGE_ID_PATTERN = re.compile(r"(?:^|[-_])v[1-9][0-9]*$", re.IGNORECASE)
_RESERVED_PACKAGE_IDS = {
    "active",
    "current",
    "latest",
    "model",
    "model.joblib",
    "production",
}
_DATE_COLUMN_ALIASES = {"attendance date", "date", "service date"}
_VISITOR_COLUMN_ALIASES = {
    "actual visitors",
    "actual visitors served",
    "attendance",
    "visitor count",
    "visitors",
}


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
    if package_id.casefold() in _RESERVED_PACKAGE_IDS:
        raise ValueError(f"Reserved unversioned package_id is not allowed: {package_id}")
    if not _VERSIONED_PACKAGE_ID_PATTERN.search(package_id):
        raise ValueError("package_id must end in an explicit version such as _v1 or -v1")

    output_root = Path(output_dir).expanduser().resolve()
    destination = output_root / package_id
    active_models = {
        model_file_for_location(location_id).resolve(),
        MODEL_FILE.resolve(),
    }
    active_models_dir = (PROJECT_ROOT / "models").resolve()
    candidate_models_dir = (active_models_dir / "candidates").resolve()
    output_is_inside_models = (
        output_root == active_models_dir or active_models_dir in output_root.parents
    )
    output_is_inside_candidate_area = (
        output_root == candidate_models_dir or candidate_models_dir in output_root.parents
    )
    if (
        output_root in active_models
        or destination in active_models
        or any(destination / MODEL_PACKAGE_NAME == active_model for active_model in active_models)
        or (output_is_inside_models and not output_is_inside_candidate_area)
    ):
        raise ValueError(
            "F6 candidate output must be outside active model paths or inside the "
            f"dedicated candidate directory: {candidate_models_dir}"
        )
    if destination.exists():
        raise FileExistsError(f"Candidate package already exists: {destination}")
    return destination


def _normalized_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().casefold()).strip()


def _select_required_column(
    columns: list[str],
    aliases: set[str],
    canonical_name: str,
) -> str:
    matches = [column for column in columns if _normalized_column_name(column) in aliases]
    if not matches:
        raise ValueError(f"Attendance CSV has no recognizable {canonical_name} column")
    if len(matches) > 1:
        raise ValueError(
            f"Attendance CSV has ambiguous {canonical_name} columns: {matches}"
        )
    return matches[0]


def _decode_csv(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise ValueError("Attendance CSV has an invalid UTF-8 byte order mark") from exc
    try:
        return raw.decode("ascii"), "us-ascii"
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8"), "utf-8"
        except UnicodeDecodeError as exc:
            raise ValueError("Attendance CSV must be ASCII or UTF-8 encoded") from exc


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_attendance_csv(
    path: str | Path,
    *,
    location_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attendance_path = Path(path).expanduser().resolve()
    if not attendance_path.is_file():
        raise FileNotFoundError(f"Attendance CSV does not exist: {attendance_path}")
    raw = attendance_path.read_bytes()
    text, encoding = _decode_csv(raw)
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error as exc:
        raise ValueError("Attendance CSV delimiter could not be determined") from exc
    try:
        source_frame = pd.read_csv(
            attendance_path,
            encoding=encoding,
            sep=dialect.delimiter,
        )
    except Exception as exc:
        raise ValueError(f"Attendance CSV could not be parsed: {exc}") from exc
    if source_frame.empty:
        raise ValueError("Attendance CSV is empty")

    columns = [str(column) for column in source_frame.columns]
    date_column = _select_required_column(
        columns, _DATE_COLUMN_ALIASES, "service date"
    )
    visitors_column = _select_required_column(
        columns, _VISITOR_COLUMN_ALIASES, "visitors"
    )
    raw_dates = source_frame[date_column]
    raw_visitors = source_frame[visitors_column]
    missing_dates = raw_dates.isna() | raw_dates.astype("string").str.strip().eq("")
    missing_visitors = raw_visitors.isna() | raw_visitors.astype("string").str.strip().eq("")
    dates = pd.to_datetime(raw_dates, errors="coerce").dt.normalize()
    visitors = pd.to_numeric(raw_visitors, errors="coerce")
    invalid_dates = ~missing_dates & dates.isna()
    invalid_visitors = (
        ~missing_visitors
        & (visitors.isna() | ~np.isfinite(visitors) | (visitors < 0))
    )
    duplicate_mask = dates.notna() & dates.duplicated(keep=False)
    duplicate_dates = sorted(
        dates[duplicate_mask].dt.date.astype(str).drop_duplicates().tolist()
    )

    if missing_dates.any() or invalid_dates.any():
        raise ValueError(
            "Attendance CSV contains missing or invalid service dates: "
            f"missing={int(missing_dates.sum())}, invalid={int(invalid_dates.sum())}"
        )
    if missing_visitors.any() or invalid_visitors.any():
        raise ValueError(
            "Attendance CSV contains missing or invalid visitor values: "
            f"missing={int(missing_visitors.sum())}, invalid={int(invalid_visitors.sum())}"
        )
    if duplicate_dates:
        raise ValueError(
            f"Attendance CSV contains duplicate service dates: {duplicate_dates}"
        )

    location_columns = [
        column
        for column in columns
        if _normalized_column_name(column) == "location id"
    ]
    source_location_ids: list[str] = []
    if location_columns:
        location_values = source_frame[location_columns[0]].dropna().astype(str).str.strip()
        source_location_ids = sorted(value for value in location_values.unique() if value)
        if location_id is not None and source_location_ids != [location_id]:
            raise ValueError(
                "Attendance CSV location IDs do not match the requested location: "
                f"{source_location_ids}"
            )

    frame = pd.DataFrame(
        {
            DATE_COL: dates,
            TARGET_COL: visitors.astype(float),
        }
    ).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    non_weekend_mask = ~frame[DATE_COL].dt.weekday.isin([5, 6])
    report = {
        "path": _display_path(attendance_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "encoding": encoding,
        "delimiter": dialect.delimiter,
        "quotechar": dialect.quotechar,
        "column_names": columns,
        "date_column": date_column,
        "visitors_column": visitors_column,
        "additional_columns": [
            column for column in columns if column not in {date_column, visitors_column}
        ],
        "normalized_columns": [DATE_COL, TARGET_COL],
        "row_count": int(len(source_frame)),
        "minimum_service_date": frame[DATE_COL].min().date().isoformat(),
        "maximum_service_date": frame[DATE_COL].max().date().isoformat(),
        "duplicate_service_date_rows": int(duplicate_mask.sum()),
        "duplicate_service_dates": duplicate_dates,
        "missing_service_dates": int(missing_dates.sum()),
        "invalid_service_dates": int(invalid_dates.sum()),
        "missing_attendance_values": int(missing_visitors.sum()),
        "invalid_attendance_values": int(invalid_visitors.sum()),
        "non_weekend_record_count": int(non_weekend_mask.sum()),
        "non_weekend_service_dates": frame.loc[
            non_weekend_mask, DATE_COL
        ].dt.date.astype(str).tolist(),
        "training_eligible_weekend_rows": int((~non_weekend_mask).sum()),
        "source_location_ids": source_location_ids,
    }
    return frame, report


def load_validated_attendance_csv(path: str | Path) -> pd.DataFrame:
    frame, _ = validate_attendance_csv(path)
    return frame


def candidate_package_metadata(
    *,
    package_id: str,
    location_id: str,
    created_at_utc: str,
    attendance_validation: dict[str, Any],
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
        "attendance_input": attendance_validation,
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


def build_f6_model_package(
    *,
    location_id: str,
    attendance: pd.DataFrame,
    package_id: str,
    package_status: str,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the locked production package from an already-loaded attendance frame."""

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
        raise ValueError("F6 training must produce separate Saturday and Sunday models")
    if set(quantile_models) != required_segments or set(preprocessors) != required_segments:
        raise ValueError("F6 package is missing a segment quantile model or preprocessor")

    metrics = {
        key: outputs.get(key, {}).get("metrics", {"BacktestRows": 0})
        for key in ("overall", "sat", "sun")
    }
    created_at = created_at_utc or datetime.now(timezone.utc).isoformat()
    location = get_location(location_id)
    package = {
        "package_id": package_id,
        "package_status": package_status,
        "created_at_utc": created_at,
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
    return package, metrics


def train_f6_candidate(
    *,
    location_id: str,
    attendance_csv: str | Path,
    output_dir: str | Path,
    package_id: str,
    expected_latest_service_date: str | None = None,
    expected_row_count: int | None = None,
    expected_source_sha256: str | None = None,
) -> Path:
    destination = candidate_package_dir(location_id, output_dir, package_id)
    attendance_path = Path(attendance_csv).expanduser().resolve()
    attendance, attendance_validation = validate_attendance_csv(
        attendance_path,
        location_id=location_id,
    )
    if (
        expected_latest_service_date is not None
        and attendance_validation["maximum_service_date"]
        != pd.Timestamp(expected_latest_service_date).date().isoformat()
    ):
        raise ValueError(
            "Attendance CSV latest service date does not match the expected date: "
            f"{attendance_validation['maximum_service_date']}"
        )
    if (
        expected_row_count is not None
        and attendance_validation["row_count"] != int(expected_row_count)
    ):
        raise ValueError(
            "Attendance CSV row count does not match the expected count: "
            f"{attendance_validation['row_count']}"
        )
    if (
        expected_source_sha256 is not None
        and attendance_validation["sha256"] != expected_source_sha256
    ):
        raise ValueError(
            "Attendance CSV SHA-256 does not match the expected source fingerprint"
        )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    package, metrics = build_f6_model_package(
        location_id=location_id,
        attendance=attendance,
        package_id=package_id,
        package_status="candidate_not_active",
        created_at_utc=created_at_utc,
    )
    metadata = candidate_package_metadata(
        package_id=package_id,
        location_id=location_id,
        created_at_utc=created_at_utc,
        attendance_validation=attendance_validation,
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
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Candidate package root; models/candidates is the protected in-repo location",
    )
    parser.add_argument("--package-id", required=True, help="Unique versioned candidate package ID")
    parser.add_argument("--expected-latest-service-date")
    parser.add_argument("--expected-row-count", type=int)
    parser.add_argument("--expected-source-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = train_f6_candidate(
        location_id=args.location,
        attendance_csv=args.attendance_csv,
        output_dir=args.output_dir,
        package_id=args.package_id,
        expected_latest_service_date=args.expected_latest_service_date,
        expected_row_count=args.expected_row_count,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(f"F6 candidate package created without activation: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
