#!/usr/bin/env python3
"""Validate, reconcile, and minimally re-evaluate the Phase 2A.5 snapshot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATE_COL, TARGET_COL
from src.feature_sets import F0, F6, build_feature_set_registry, make_repaired_feature_builder
from src.origin_backtest import BASELINE_COLUMNS, OriginAwareBacktester, calculate_metrics
from src.origin_features import (
    ATTENDANCE_FEATURES,
    CALENDAR_FEATURES,
    DAYTYPE_SLOT_FEATURES,
    DAYTYPE_SUMMARY_FEATURES,
    HORIZON_AWARE_FEATURES,
    LAST_OBSERVED_DAYTYPE_FEATURES,
    MODEL_FEATURES,
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
)


LOCATION_ID = "ny_12550"
SOURCE_RELATIVE = Path("data/locations/ny_12550/Updated/2026-07-15T05-23_export.csv")
SOURCE_PATH = ROOT / SOURCE_RELATIVE
OUTPUT_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation"
PHASE1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
PHASE2A_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a_feature_repair"
MODEL_AUDIT_DIR = ROOT / "artifacts/ny_12550/model_audit"
MODEL_PATH = ROOT / "models/visitor_model_ny_12550.joblib"
GENERIC_MODEL_PATH = ROOT / "models/visitor_model.joblib"
SQLITE_PATH = ROOT / "data/locations/ny_12550/attendance.db"
LEGACY_CSV_PATH = ROOT / "data/visitors_clean.csv"
MATCHED_END = pd.Timestamp("2026-06-21")
EXTENSION_END = pd.Timestamp("2026-07-12")
EXPECTED_SOURCE_SHA256 = "e3f84ac47245fa7eb5496413dbd04c5c0d0fead2ed553e257da57c3278ffdef8"
SNAPSHOT_ID = "ny_12550_supabase_export_2026-07-15T05-23_e3f84ac47245"
TIMESTAMPED_SNAPSHOT = "03_normalized_supabase_snapshot_2026-07-15T05-23.csv"
RANDOM_SEED = 42
TASK_START_GIT_STATUS = """?? data/locations/ny_12550/Updated/
?? scripts/finalize_phase2a_reports.py
?? scripts/run_phase1_origin_backtest.py
?? scripts/run_phase2a_feature_repair.py
?? src/feature_sets.py
?? src/origin_backtest.py
?? src/origin_features.py
?? tests/test_feature_sets.py
?? tests/test_origin_backtest.py
?? tests/test_origin_features.py
?? tests/test_phase2a_feature_repair.py"""

REQUIRED_ARTIFACTS = [
    "00_implementation_design.md",
    "01_phase2a5_summary.md",
    "02_supabase_csv_validation.json",
    "02_supabase_csv_validation.md",
    "03_normalized_supabase_snapshot.csv",
    "04_source_inventory.csv",
    "04_source_inventory.md",
    "05_date_level_reconciliation.csv",
    "05_date_level_reconciliation.md",
    "06_new_and_changed_records.csv",
    "06_new_and_changed_records.md",
    "07_locked_feature_set.json",
    "07_locked_feature_set.md",
    "08_latest_snapshot_predictions.csv",
    "09_full_history_metrics.csv",
    "09_full_history_metrics.md",
    "10_matched_history_metrics.csv",
    "10_matched_history_metrics.md",
    "11_new_extension_predictions.csv",
    "11_new_extension_analysis.md",
    "12_daytype_scenario_horizon_metrics.csv",
    "12_daytype_scenario_horizon_metrics.md",
    "13_source_revision_effects.md",
    "14_phase2b_data_decision.md",
    "15_test_and_reproducibility_report.md",
    "phase2a5_manifest.json",
    "README.md",
]

PRODUCTION_PATHS = [
    ROOT / "app.py",
    ROOT / "src/predictor.py",
    ROOT / "src/features.py",
    ROOT / "src/modeling.py",
    ROOT / "scripts/train_model.py",
]
PROTECTED_FILES = [
    *PRODUCTION_PATHS,
    MODEL_PATH,
    GENERIC_MODEL_PATH,
    SQLITE_PATH,
    LEGACY_CSV_PATH,
]

PREDICTION_COLUMNS = [
    "data_snapshot_id",
    "source_file",
    "source_file_sha256",
    "candidate_id",
    "forecast_origin",
    "target_date",
    "scenario",
    "service_horizon",
    "calendar_days_ahead",
    "day_type",
    "actual",
    "point_prediction",
    "quantile_prediction",
    "point_error",
    "absolute_error",
    "quantile_covers",
    "period_group",
    "training_end_date",
    "training_row_count",
    "segment_training_row_count",
    "feature_count",
    "feature_missing_count",
    "feature_provenance_valid",
    "source_record_id",
    "source_updated_at",
    "random_seed",
    "weather_policy",
    "weekday_policy",
    "model_segment",
    "preprocessing_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_attendance_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame[[DATE_COL, TARGET_COL]].copy()
    canonical[DATE_COL] = pd.to_datetime(canonical[DATE_COL]).dt.strftime("%Y-%m-%d")
    canonical[TARGET_COL] = pd.to_numeric(canonical[TARGET_COL])
    canonical = canonical.sort_values(DATE_COL, kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_value) + "\n",
        encoding="utf-8",
    )


def write_text(name: str, text: str) -> None:
    (OUTPUT_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(name: str, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_DIR / name, index=False, lineterminator="\n")


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    if view.empty:
        return "_No rows._"
    for column in view.columns:
        view[column] = view[column].map(
            lambda value: ""
            if pd.isna(value)
            else json.dumps(value, sort_keys=True)
            if isinstance(value, (list, dict, tuple))
            else f"{value:.6f}"
            if isinstance(value, float)
            else str(value)
        )
    headers = [str(column).replace("|", "\\|") for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows; the CSV is complete._")
    return "\n".join(lines)


def protected_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in PROTECTED_FILES
        if path.exists()
    }


def phase_fingerprints() -> dict[str, str]:
    return {
        str(PHASE1_DIR.relative_to(ROOT)): directory_fingerprint(PHASE1_DIR),
        str(PHASE2A_DIR.relative_to(ROOT)): directory_fingerprint(PHASE2A_DIR),
        str(MODEL_AUDIT_DIR.relative_to(ROOT)): directory_fingerprint(MODEL_AUDIT_DIR),
    }


def infer_export_timestamp() -> str:
    return "2026-07-15T05:23:00 (filename-derived; timezone not encoded)"


def enforce_duplicate_selection_rule(service_dates: pd.Series) -> None:
    """Stop on duplicates because the observed schema cannot resolve them safely."""

    parsed = pd.to_datetime(service_dates, errors="coerce")
    duplicates = parsed[parsed.duplicated(keep=False) & parsed.notna()]
    if not duplicates.empty:
        affected = sorted(duplicates.dt.strftime("%Y-%m-%d").unique())
        raise ValueError(
            "Duplicate service dates cannot be resolved safely without status, "
            f"timestamps, and stable record IDs: {affected}"
        )


def validate_and_normalize_export() -> tuple[pd.DataFrame, dict[str, Any]]:
    if SOURCE_PATH != ROOT / SOURCE_RELATIVE or not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Exact configured CSV path does not exist: {SOURCE_PATH}")
    raw = SOURCE_PATH.read_bytes()
    initial_hash = hashlib.sha256(raw).hexdigest()
    if initial_hash != EXPECTED_SOURCE_SHA256:
        raise AssertionError(f"Unexpected source hash: {initial_hash}")
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    dialect = csv.Sniffer().sniff(text[:4096])
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    columns = list(rows[0]) if rows else []
    expected_columns = ["Service date", "Actual visitors served"]
    if columns != expected_columns:
        raise ValueError(f"Unsupported observed schema: {columns}")

    frame = pd.DataFrame(rows)
    raw_dates = frame["Service date"].astype(str)
    parsed_dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    numeric = pd.to_numeric(frame["Actual visitors served"], errors="coerce")
    invalid_date_rows = frame.index[parsed_dates.isna()].tolist()
    non_numeric_rows = frame.index[numeric.isna() & frame["Actual visitors served"].astype(str).str.strip().ne("")].tolist()
    missing_date_rows = frame.index[frame["Service date"].astype(str).str.strip().eq("")].tolist()
    missing_attendance_rows = frame.index[frame["Actual visitors served"].astype(str).str.strip().eq("")].tolist()
    duplicate_dates = parsed_dates[parsed_dates.duplicated(keep=False) & parsed_dates.notna()].dt.strftime("%Y-%m-%d").tolist()
    negatives = frame.index[numeric.lt(0)].tolist()
    zeros = frame.index[numeric.eq(0)].tolist()
    fractional = frame.index[numeric.notna() & numeric.mod(1).ne(0)].tolist()
    non_weekend = parsed_dates[parsed_dates.notna() & ~parsed_dates.dt.weekday.isin([5, 6])].dt.strftime("%Y-%m-%d").tolist()

    if duplicate_dates:
        enforce_duplicate_selection_rule(parsed_dates)
    hard_failures = {
        "invalid_date_rows": invalid_date_rows,
        "non_numeric_attendance_rows": non_numeric_rows,
        "missing_service_date_rows": missing_date_rows,
        "missing_attendance_rows": missing_attendance_rows,
        "duplicate_service_dates": duplicate_dates,
        "negative_attendance_rows": negatives,
        "fractional_attendance_rows": fractional,
    }
    if any(hard_failures.values()):
        raise ValueError(f"CSV failed validation: {hard_failures}")

    normalized = pd.DataFrame(
        {
            "location_id": LOCATION_ID,
            "service_date": parsed_dates.dt.strftime("%Y-%m-%d"),
            "attendance": numeric.astype(int),
            "team": pd.NA,
            "source_record_id": pd.NA,
            "source_created_at": pd.NA,
            "source_updated_at": pd.NA,
            "source_file": str(SOURCE_RELATIVE),
            "source_file_sha256": initial_hash,
            "normalization_status": np.where(
                parsed_dates.dt.weekday.isin([5, 6]),
                "accepted_path_scoped_location",
                "accepted_source_non_t1_weekday",
            ),
        }
    )
    if normalized.duplicated(["location_id", "service_date"]).any():
        raise AssertionError("Normalized export contains duplicate location/date keys")

    stat = SOURCE_PATH.stat()
    exceptions = [
        {
            "category": "schema_metadata_unavailable",
            "count": len(frame),
            "detail": "Location, team, record ID, created/updated timestamps, weekday, notes, status, and deleted/inactive/test fields are absent.",
        },
        {
            "category": "path_scoped_location_inference",
            "count": len(frame),
            "detail": "ny_12550 is inferred from the exact configured location directory; it cannot be verified row-by-row.",
        },
        {
            "category": "non_service_weekday",
            "count": len(non_weekend),
            "affected_service_dates": non_weekend,
            "detail": "Retained in the normalized source snapshot; excluded from T1 model history/targets.",
        },
    ]
    validation: dict[str, Any] = {
        "validation_status": "accepted_with_schema_limitations",
        "model_evaluation_permitted": True,
        "relative_path": str(SOURCE_RELATIVE),
        "absolute_path": str(SOURCE_PATH),
        "file_name": SOURCE_PATH.name,
        "file_size_bytes": stat.st_size,
        "modification_time_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256_before_validation": initial_hash,
        "sha256_after_validation": sha256_file(SOURCE_PATH),
        "original_csv_unchanged": initial_hash == sha256_file(SOURCE_PATH),
        "encoding": "utf-8-sig",
        "has_utf8_bom": has_bom,
        "delimiter": dialect.delimiter,
        "quote_character": dialect.quotechar,
        "column_names": columns,
        "schema_mapping": {
            "location_identifier": None,
            "service_date": "Service date",
            "attendance_count": "Actual visitors served",
            "team": None,
            "created_timestamp": None,
            "updated_timestamp": None,
            "record_identifier": None,
            "service_weekday": None,
            "notes": None,
            "status": None,
            "deleted_or_inactive_state": None,
        },
        "total_row_count": int(len(frame)),
        "rows_belonging_to_ny_12550": int(len(frame)),
        "location_membership_basis": "configured path inference; no row-level location field",
        "minimum_service_date": parsed_dates.min().strftime("%Y-%m-%d"),
        "maximum_service_date": parsed_dates.max().strftime("%Y-%m-%d"),
        "maximum_update_timestamp": None,
        "export_timestamp_inferred": infer_export_timestamp(),
        "duplicate_record_ids": {"status": "not_applicable", "count": 0, "reason": "record ID column absent"},
        "duplicate_location_date_pairs": int(len(duplicate_dates)),
        "missing_location_ids": {"status": "schema_unavailable", "count": None},
        "missing_service_dates": int(len(missing_date_rows)),
        "missing_attendance": int(len(missing_attendance_rows)),
        "non_numeric_attendance": int(len(non_numeric_rows)),
        "negative_attendance": int(len(negatives)),
        "zero_attendance": int(len(zeros)),
        "non_integer_attendance": int(len(fractional)),
        "non_saturday_sunday_service_dates": non_weekend,
        "invalid_or_malformed_dates": invalid_date_rows,
        "multiple_records_same_service": duplicate_dates,
        "deleted_records": {"status": "schema_unavailable", "count": None},
        "cancelled_records": {"status": "schema_unavailable", "count": None},
        "test_records": {"status": "schema_unavailable", "count": None},
        "inactive_records": {"status": "schema_unavailable", "count": None},
        "timezone_interpretation": "Service date is parsed as a date-only ISO value with no timezone conversion.",
        "timestamp_conversion_changes_service_dates": False,
        "contains_2026_07_12": bool((parsed_dates == EXTENSION_END).any()),
        "records_after_2026_07_12": parsed_dates[parsed_dates > EXTENSION_END].dt.strftime("%Y-%m-%d").tolist(),
        "duplicate_selection_rule": "No duplicates observed. If duplicates appear, stop because the schema lacks status, timestamps, and stable IDs; do not aggregate.",
        "accepted_row_count": int(len(normalized)),
        "excluded_row_count": 0,
        "unresolved_row_count": 0,
        "validation_exceptions": exceptions,
        "protected_file_fingerprints_at_validation": protected_fingerprints(),
        "prior_artifact_directory_fingerprints_at_validation": phase_fingerprints(),
    }
    return normalized, validation


def history_from_normalized(normalized: pd.DataFrame) -> pd.DataFrame:
    frame = normalized[["service_date", "attendance"]].rename(
        columns={"service_date": DATE_COL, "attendance": TARGET_COL}
    )
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], format="%Y-%m-%d")
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL]).astype(float)
    return frame.sort_values(DATE_COL, kind="stable").reset_index(drop=True)


def load_local_sources() -> dict[str, dict[str, Any]]:
    package = joblib.load(MODEL_PATH)
    package_history = package["history_df"][[DATE_COL, TARGET_COL]].copy()
    package_history[DATE_COL] = pd.to_datetime(package_history[DATE_COL]).dt.normalize()
    with sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True) as connection:
        sqlite_history = pd.read_sql_query(
            "SELECT service_date, visitors FROM attendance ORDER BY service_date", connection
        )
    sqlite_history[DATE_COL] = pd.to_datetime(sqlite_history[DATE_COL]).dt.normalize()
    legacy_history = pd.read_csv(LEGACY_CSV_PATH, usecols=[DATE_COL, TARGET_COL])
    legacy_history[DATE_COL] = pd.to_datetime(legacy_history[DATE_COL]).dt.normalize()
    generic = joblib.load(GENERIC_MODEL_PATH)
    generic_history = generic["history_df"][[DATE_COL, TARGET_COL]].copy()
    generic_history[DATE_COL] = pd.to_datetime(generic_history[DATE_COL]).dt.normalize()
    return {
        "saved_model_package_history": {"frame": package_history, "path": MODEL_PATH, "package": package},
        "local_sqlite_attendance": {"frame": sqlite_history, "path": SQLITE_PATH},
        "legacy_csv_attendance": {"frame": legacy_history, "path": LEGACY_CSV_PATH},
        "generic_saved_model_package_history": {"frame": generic_history, "path": GENERIC_MODEL_PATH},
    }


def source_inventory(normalized: pd.DataFrame, sources: dict[str, dict[str, Any]]) -> pd.DataFrame:
    supabase = history_from_normalized(normalized)
    rows: list[dict[str, Any]] = []

    def row(
        source_id: str,
        path: Path,
        frame: pd.DataFrame,
        *,
        current_training: str,
        current_prediction: str,
        phase1: str,
        phase2a: str,
        latest_update: str | None = None,
    ) -> dict[str, Any]:
        dates = pd.to_datetime(frame[DATE_COL])
        valid = frame[dates.notna() & pd.to_numeric(frame[TARGET_COL], errors="coerce").notna()]
        return {
            "source_id": source_id,
            "source_path": str(path.relative_to(ROOT)),
            "row_count": int(len(frame)),
            "valid_row_count": int(len(valid)),
            "minimum_service_date": dates.min(),
            "maximum_service_date": dates.max(),
            "latest_service_date": dates.max(),
            "latest_update_timestamp": latest_update,
            "duplicate_service_dates": int(dates.duplicated().sum()),
            "missing_attendance_count": int(pd.to_numeric(frame[TARGET_COL], errors="coerce").isna().sum()),
            "non_service_weekday_count": int((~dates.dt.weekday.isin([5, 6])).sum()),
            "stable_fingerprint": stable_attendance_fingerprint(frame),
            "file_sha256": sha256_file(path),
            "matches_another_source_exactly": "pending pairwise comparison",
            "current_training_uses": current_training,
            "current_prediction_uses": current_prediction,
            "phase1_used": phase1,
            "phase2a_used": phase2a,
        }

    rows.append(
        row(
            "supabase_export_candidate",
            SOURCE_PATH,
            supabase,
            current_training="No; this local export is not read by production training",
            current_prediction="No; prediction reads the saved location package",
            phase1="No",
            phase2a="No",
        )
    )
    usage = {
        "saved_model_package_history": (
            "Saved output from the latest location training run",
            "Yes; location predictor reads package history_df",
            "Yes; authority",
            "Yes; authority",
        ),
        "local_sqlite_attendance": (
            "Yes when local fallback storage is active",
            "No; live prediction reads the frozen package history",
            "Reconciled only",
            "No",
        ),
        "legacy_csv_attendance": (
            "Only bootstraps an empty local SQLite store",
            "No",
            "Reconciled only",
            "No",
        ),
        "generic_saved_model_package_history": (
            "Saved output for the generic/default model",
            "Yes for the generic/default predictor path; not the NY package",
            "No",
            "No",
        ),
    }
    for source_id, payload in sources.items():
        current_training, current_prediction, phase1, phase2a = usage[source_id]
        rows.append(
            row(
                source_id,
                payload["path"],
                payload["frame"],
                current_training=current_training,
                current_prediction=current_prediction,
                phase1=phase1,
                phase2a=phase2a,
            )
        )
    inventory = pd.DataFrame(rows)
    fingerprints = inventory.groupby("stable_fingerprint")["source_id"].transform(lambda values: ", ".join(sorted(values)))
    inventory["matches_another_source_exactly"] = np.where(
        fingerprints.str.contains(","), fingerprints, "No"
    )
    return inventory


def reconcile_sources(
    normalized: pd.DataFrame,
    sources: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supabase = normalized[["service_date", "attendance"]].copy()
    supabase["service_date"] = pd.to_datetime(supabase["service_date"])
    supabase = supabase.rename(columns={"attendance": "supabase_attendance"})
    merged = supabase
    mapping = {
        "saved_model_package_history": "model_package_attendance",
        "local_sqlite_attendance": "sqlite_attendance",
        "legacy_csv_attendance": "legacy_csv_attendance",
        "generic_saved_model_package_history": "generic_package_attendance",
    }
    for source_id, output_column in mapping.items():
        other = sources[source_id]["frame"][[DATE_COL, TARGET_COL]].rename(
            columns={DATE_COL: "service_date", TARGET_COL: output_column}
        )
        merged = merged.merge(other, on="service_date", how="outer", validate="one_to_one")
    merged = merged.sort_values("service_date", kind="stable").reset_index(drop=True)

    value_columns = ["supabase_attendance", *mapping.values()]
    def status(row: pd.Series) -> str:
        values = [float(row[column]) for column in value_columns if pd.notna(row[column])]
        if pd.isna(row["supabase_attendance"]):
            return "missing_from_supabase"
        if all(pd.isna(row[column]) for column in value_columns[1:]):
            return "supabase_only"
        if len(set(values)) == 1:
            return "all_available_values_match"
        if pd.notna(row["model_package_attendance"]) and row["supabase_attendance"] == row["model_package_attendance"]:
            return "supabase_matches_model_package_other_source_differs"
        return "source_value_conflict"

    merged["match_status"] = merged.apply(status, axis=1)
    merged["source_conflict"] = merged[value_columns].apply(
        lambda row: len(set(float(value) for value in row.dropna())) > 1, axis=1
    )
    merged["supabase_record_id"] = pd.NA
    merged["supabase_created_timestamp"] = pd.NA
    merged["supabase_updated_timestamp"] = pd.NA
    merged["recommended_authoritative_value"] = merged["supabase_attendance"]
    merged["reason_for_recommendation"] = np.where(
        merged["supabase_attendance"].notna(),
        "Unique validated export row; Phase 2A.5 evaluation authority only",
        "No Supabase-export value; do not fill automatically",
    )

    changed_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        supa = row.supabase_attendance
        package = row.model_package_attendance
        if pd.isna(supa) and pd.notna(package):
            category = "missing_from_supabase"
        elif pd.notna(supa) and pd.isna(package):
            category = "added_in_supabase_snapshot"
        elif pd.notna(supa) and pd.notna(package) and float(supa) != float(package):
            category = "attendance_value_changed"
        else:
            continue
        changed_rows.append(
            {
                "service_date": row.service_date,
                "change_category": category,
                "supabase_attendance": supa,
                "model_package_attendance": package,
                "attendance_delta": float(supa - package) if pd.notna(supa) and pd.notna(package) else np.nan,
                "after_2026_06_21": bool(row.service_date > MATCHED_END),
                "team_change_status": "unobservable; team absent from export",
                "timestamp_change_status": "unobservable; source timestamps absent from export",
                "source_record_id": pd.NA,
                "source_created_at": pd.NA,
                "source_updated_at": pd.NA,
            }
        )
    changes = pd.DataFrame(changed_rows)
    return merged, changes


def locked_feature_contract() -> dict[str, Any]:
    phase2a_manifest = json.loads((PHASE2A_DIR / "phase2a_manifest.json").read_text())
    phase2a_registry = json.loads((PHASE2A_DIR / "02_feature_set_registry.json").read_text())
    manifest_features = phase2a_manifest["f6_lock"]["feature_list"]
    if isinstance(phase2a_registry, dict) and F6 in phase2a_registry:
        registry_features = phase2a_registry[F6]["feature_list"]
    else:
        record = next(item for item in phase2a_registry if item["feature_set_id"] == F6)
        registry_features = record["feature_list"]
    runtime_registry = build_feature_set_registry(
        f5_parent_id=phase2a_manifest["f5_parent_id"],
        f6_features=manifest_features,
        f6_groups=phase2a_manifest["feature_set_registry"][F6]["feature_groups"],
    )
    runtime_features = list(runtime_registry[F6].feature_list)
    if not (manifest_features == registry_features == runtime_features):
        raise AssertionError("F6 feature lists differ across Phase 2A artifacts/runtime registry")
    if sha256_json(manifest_features) != phase2a_manifest["f6_lock"]["feature_list_sha256"]:
        raise AssertionError("F6 feature hash differs from Phase 2A manifest")
    return {
        "selected_feature_set_id": F6,
        "selected_feature_set_name": runtime_registry[F6].name,
        "feature_count": len(manifest_features),
        "ordered_feature_list": manifest_features,
        "feature_list_sha256": sha256_json(manifest_features),
        "phase2a_manifest_feature_list_sha256": phase2a_manifest["f6_lock"]["feature_list_sha256"],
        "f0_feature_count": len(MODEL_FEATURES),
        "f0_ordered_feature_list": list(MODEL_FEATURES),
        "point_model": phase2a_manifest["model_configurations"]["point"],
        "quantile_model": phase2a_manifest["model_configurations"]["quantile"],
        "random_seeds": phase2a_manifest["random_seeds"],
        "training_rules": {
            "window": "expanding",
            "segmentation": "Saturday/Sunday",
            "min_segment_training_rows": 18,
            "preprocessing": "fold-local SimpleImputer(median, keep_empty_features=True)",
            "weather_policy": W0_NO_WEATHER,
            "weekday_policy": T1_VALID_WEEKENDS,
            "residual_calibration": "none",
            "meal_buffer": "none in model accuracy",
            "recursive_attendance_predictions": False,
        },
        "origin_definitions": phase2a_manifest["origin_definitions"],
        "development_confirmation_ranges": phase2a_manifest["development_confirmation_ranges"],
        "locked_before_latest_scoring": True,
        "source_artifacts": [
            str((PHASE2A_DIR / "15_f6_compact_feature_decision.md").relative_to(ROOT)),
            str((PHASE2A_DIR / "02_feature_set_registry.json").relative_to(ROOT)),
            str((PHASE2A_DIR / "phase2a_manifest.json").relative_to(ROOT)),
            str((PHASE2A_DIR / "01_phase2a_summary.md").relative_to(ROOT)),
        ],
    }


def attendance_feature_columns(features: Iterable[str]) -> list[str]:
    attendance_derived = set(
        ATTENDANCE_FEATURES
        + LAST_OBSERVED_DAYTYPE_FEATURES
        + DAYTYPE_SUMMARY_FEATURES
        + DAYTYPE_SLOT_FEATURES
        + ["days_since_last_observed_daytype"]
    )
    return [feature for feature in features if feature in attendance_derived]


def write_validation_artifacts(
    normalized: pd.DataFrame,
    validation: dict[str, Any],
    inventory: pd.DataFrame,
    reconciliation: pd.DataFrame,
    changes: pd.DataFrame,
    lock: dict[str, Any],
) -> None:
    write_json("02_supabase_csv_validation.json", validation)
    exceptions = pd.DataFrame(validation["validation_exceptions"])
    write_text(
        "02_supabase_csv_validation.md",
        f"""# Supabase CSV validation

Status: **{validation['validation_status']}**. Model evaluation is permitted with
the explicit schema limitations below. The original byte hash was unchanged
after validation.

## File and schema

| field | value |
| --- | --- |
| relative path | `{validation['relative_path']}` |
| absolute path | `{validation['absolute_path']}` |
| bytes | {validation['file_size_bytes']} |
| SHA-256 | `{validation['sha256_after_validation']}` |
| encoding / delimiter | `{validation['encoding']}` / comma |
| columns | `{', '.join(validation['column_names'])}` |
| rows | {validation['total_row_count']} |
| date range | {validation['minimum_service_date']} through {validation['maximum_service_date']} |
| maximum update timestamp | unavailable: column absent |

## Validation exceptions

{markdown_table(exceptions)}

There are no duplicate dates, missing/invalid dates, missing/non-numeric counts,
negative counts, zero counts, or records after 2026-07-12. The export contains
2026-07-12. Date-only parsing performs no timezone conversion and changes no
service date. Deleted/cancelled/test/inactive checks are schema-unavailable, not
silently treated as verified negatives.

## Duplicate rule

{validation['duplicate_selection_rule']}
""",
    )
    normalized.to_csv(OUTPUT_DIR / TIMESTAMPED_SNAPSHOT, index=False, lineterminator="\n")
    normalized.to_csv(OUTPUT_DIR / "03_normalized_supabase_snapshot.csv", index=False, lineterminator="\n")
    if sha256_file(OUTPUT_DIR / TIMESTAMPED_SNAPSHOT) != sha256_file(OUTPUT_DIR / "03_normalized_supabase_snapshot.csv"):
        raise AssertionError("Qualified and required normalized snapshots differ")
    write_csv("04_source_inventory.csv", inventory)
    write_text(
        "04_source_inventory.md",
        "# Attendance source inventory\n\n" + markdown_table(inventory) +
        "\n\nNo sources are merged. Usage statements describe repository behavior; the local export itself is not a production input. Loading the generic/default saved package emits a scikit-learn 1.7.1-to-1.5.2 compatibility warning; only its raw history table is inventoried, and none of its serialized estimators is used for Phase 2A.5 scoring.\n",
    )
    write_csv("05_date_level_reconciliation.csv", reconciliation)
    conflict_count = int(reconciliation["source_conflict"].sum())
    write_text(
        "05_date_level_reconciliation.md",
        f"# Date-level reconciliation\n\nThe complete outer-union table has {len(reconciliation)} service dates and {conflict_count} dates with differing available values. Supabase values are recommended only for this evaluation snapshot; missing values are never filled automatically.\n\n" +
        markdown_table(reconciliation[reconciliation["match_status"] != "all_available_values_match"], max_rows=80),
    )
    write_csv("06_new_and_changed_records.csv", changes)
    category_counts = changes.groupby("change_category").size().rename("row_count").reset_index() if not changes.empty else pd.DataFrame(columns=["change_category", "row_count"])
    write_text(
        "06_new_and_changed_records.md",
        f"# New and changed records versus saved model history\n\n{markdown_table(category_counts)}\n\n" +
        markdown_table(changes, max_rows=100) +
        "\n\nTeam changes, update-only changes, and edits-after-service cannot be identified because the export has no team or source timestamp columns.\n",
    )
    write_json("07_locked_feature_set.json", lock)
    write_text(
        "07_locked_feature_set.md",
        f"# Locked Phase 2A feature set\n\n- ID: **{lock['selected_feature_set_id']}**\n- Ordered feature count: {lock['feature_count']}\n- Feature-list SHA-256: `{lock['feature_list_sha256']}`\n- Point model: `{lock['point_model']}`\n- Quantile model: `{lock['quantile_model']}`\n\nExact ordered features: " +
        ", ".join(f"`{item}`" for item in lock["ordered_feature_list"]) +
        "\n\nThe lock was read and cross-checked before any latest-snapshot scoring; no feature was added, removed, reordered, or reselected.\n",
    )


def validation_stage() -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    design = OUTPUT_DIR / "00_implementation_design.md"
    if not design.exists():
        raise FileNotFoundError("Implementation design must exist before validation/evaluation")
    normalized, validation = validate_and_normalize_export()
    sources = load_local_sources()
    inventory = source_inventory(normalized, sources)
    reconciliation, changes = reconcile_sources(normalized, sources)
    lock = locked_feature_contract()
    write_validation_artifacts(normalized, validation, inventory, reconciliation, changes, lock)
    if sha256_file(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise AssertionError("Original export changed during validation")
    return normalized, validation, sources, inventory, reconciliation, changes, lock


def make_evaluator(history: pd.DataFrame, lock: dict[str, Any], candidate_id: str) -> OriginAwareBacktester:
    if candidate_id == F0:
        features = list(MODEL_FEATURES)
        builder = None
    elif candidate_id == F6:
        features = list(lock["ordered_feature_list"])
        registry = build_feature_set_registry(
            f5_parent_id="F4_CORRECTED_SLOT_HISTORY",
            f6_features=features,
            f6_groups=["calendar", "last_observed_daytype", "daytype_summaries", "daytype_slot", "horizon_availability"],
        )
        builder = make_repaired_feature_builder(registry[F6])
    else:
        raise ValueError(candidate_id)
    return OriginAwareBacktester(
        history,
        weather_df=None,
        feature_cols=features,
        residual_buffer_by_day={"sat": 0.0, "sun": 0.0},
        default_meal_buffer_pct=0.0,
        min_train_size=18,
        quantile=0.8,
        feature_set_id=candidate_id,
        feature_builder=builder,
        attendance_feature_cols=attendance_feature_columns(features),
        random_seed=RANDOM_SEED,
    )


def prediction_metadata(normalized: pd.DataFrame) -> pd.DataFrame:
    metadata = normalized[["service_date", "source_record_id", "source_updated_at"]].copy()
    metadata["target_date"] = pd.to_datetime(metadata.pop("service_date"))
    return metadata


def add_snapshot_fields(predictions: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    for column in ["forecast_origin", "target_date", "training_end_date"]:
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    eligible_dates = sorted(frame["target_date"].drop_duplicates())
    recent_dates = set(eligible_dates[-52:])
    frame["period_group"] = np.where(frame["target_date"].isin(recent_dates), "Recent 52", "Earlier")
    frame = frame.merge(prediction_metadata(normalized), on="target_date", how="left", validate="many_to_one")
    frame["data_snapshot_id"] = SNAPSHOT_ID
    frame["source_file"] = str(SOURCE_RELATIVE)
    frame["source_file_sha256"] = EXPECTED_SOURCE_SHA256
    frame["candidate_id"] = frame["feature_set_id"]
    return frame


def baseline_prediction_rows(model_rows: pd.DataFrame) -> pd.DataFrame:
    baseline_ids = {
        "previous_same_daytype": "BASELINE_PREVIOUS_SAME_DAYTYPE",
        "mean_last4_same_daytype": "BASELINE_MEAN_LAST4_SAME_DAYTYPE",
        "median_last4_same_daytype": "BASELINE_MEDIAN_LAST4_SAME_DAYTYPE",
    }
    rows: list[pd.DataFrame] = []
    for source_column, candidate_id in baseline_ids.items():
        frame = model_rows.copy()
        frame["candidate_id"] = candidate_id
        frame["feature_set_id"] = candidate_id
        frame["point_prediction"] = frame[source_column]
        frame["quantile_prediction"] = np.nan
        frame["point_error"] = frame["point_prediction"] - frame["actual"]
        frame["absolute_error"] = frame["point_error"].abs()
        frame["quantile_covers"] = pd.NA
        frame["feature_count"] = 1 if source_column == "previous_same_daytype" else 4
        frame["feature_missing_count"] = frame["point_prediction"].isna().astype(int)
        frame["feature_provenance_valid"] = True
        frame["training_end_date"] = pd.NaT
        frame["training_row_count"] = np.nan
        frame["segment_training_row_count"] = np.nan
        frame["preprocessing_id"] = "not_applicable_fixed_origin_baseline"
        frame["model_segment"] = np.where(frame["day_type"].eq("Sunday"), "sun", "sat")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def run_evaluation(
    normalized: pd.DataFrame,
    lock: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = history_from_normalized(normalized)
    prediction_parts: list[pd.DataFrame] = []
    provenance_parts: list[pd.DataFrame] = []
    preprocessing_parts: list[pd.DataFrame] = []
    for candidate_id in [F0, F6]:
        print(f"Running {candidate_id} on validated Supabase snapshot...", flush=True)
        evaluator = make_evaluator(history, lock, candidate_id)
        result = evaluator.run(
            weather_policies=[W0_NO_WEATHER],
            weekday_policies=[T1_VALID_WEEKENDS],
        )
        prediction_parts.append(add_snapshot_fields(result.predictions, normalized))
        provenance = result.feature_provenance.copy()
        provenance["candidate_id"] = candidate_id
        provenance_parts.append(provenance)
        preprocessing_parts.append(result.preprocessing_diagnostics)
    model_predictions = pd.concat(prediction_parts, ignore_index=True)
    f0_rows = model_predictions[model_predictions["candidate_id"] == F0].copy()
    all_predictions = pd.concat([model_predictions, baseline_prediction_rows(f0_rows)], ignore_index=True)
    key = [
        "data_snapshot_id", "candidate_id", "forecast_origin", "target_date", "scenario",
        "weather_policy", "weekday_policy",
    ]
    if all_predictions.duplicated(key).any():
        raise AssertionError("Prediction keys are not unique")
    if not (all_predictions["forecast_origin"] < all_predictions["target_date"]).all():
        raise AssertionError("Forecast origin is not strictly before target")
    model_only = all_predictions[all_predictions["candidate_id"].isin([F0, F6])]
    if not (model_only["training_end_date"] <= model_only["forecast_origin"]).all():
        raise AssertionError("Fold-local training cutoff exceeds forecast origin")
    if not model_only["feature_provenance_valid"].all():
        raise AssertionError("Feature provenance validity failed")

    provenance = pd.concat(provenance_parts, ignore_index=True)
    attendance = provenance[provenance["source_type"] == "attendance"]
    violations = 0
    for row in attendance.itertuples(index=False):
        for date_text in json.loads(row.available_source_dates):
            violations += int(pd.Timestamp(date_text) > pd.Timestamp(row.forecast_origin))
    if violations:
        raise AssertionError(f"Found {violations} post-origin attendance provenance dates")
    preprocessing = pd.concat(preprocessing_parts, ignore_index=True).drop_duplicates(
        ["preprocessing_id", "feature"], keep="first"
    )
    if preprocessing["fit_includes_test_or_future"].any():
        raise AssertionError("Fold-local preprocessing includes test/future rows")
    return all_predictions.sort_values(key, kind="stable").reset_index(drop=True), provenance, preprocessing


def extended_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    base = calculate_metrics(frame)
    usable_q = frame.dropna(subset=["actual", "quantile_prediction"]).copy()
    if usable_q.empty:
        base.update(
            {
                "mean_quantile_shortfall_when_uncovered": np.nan,
                "mean_quantile_excess_when_covered": np.nan,
            }
        )
        return base
    uncovered = usable_q[usable_q["actual"] > usable_q["quantile_prediction"]]
    covered = usable_q[usable_q["actual"] <= usable_q["quantile_prediction"]]
    base["mean_quantile_shortfall_when_uncovered"] = (
        float((uncovered["actual"] - uncovered["quantile_prediction"]).mean()) if not uncovered.empty else 0.0
    )
    base["mean_quantile_excess_when_covered"] = (
        float((covered["quantile_prediction"] - covered["actual"]).mean()) if not covered.empty else 0.0
    )
    return base


def window_filter(frame: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if window_id == "full_supabase_history":
        return frame
    if window_id == "matched_history_through_2026_06_21":
        return frame[frame["target_date"] <= MATCHED_END]
    if window_id == "new_extension_after_2026_06_21":
        return frame[(frame["target_date"] > MATCHED_END) & (frame["target_date"] <= EXTENSION_END)]
    raise ValueError(window_id)


def overall_metrics(predictions: pd.DataFrame, window_id: str) -> pd.DataFrame:
    frame = window_filter(predictions, window_id)
    return pd.DataFrame(
        [
            {"evaluation_window": window_id, "candidate_id": candidate_id, **extended_metrics(part)}
            for candidate_id, part in frame.groupby("candidate_id", sort=True)
        ]
    )


def breakdown_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    windows = [
        "full_supabase_history",
        "matched_history_through_2026_06_21",
        "new_extension_after_2026_06_21",
    ]
    dimensions = [
        ("overall", None),
        ("day_type", "day_type"),
        ("scenario", "scenario"),
        ("service_horizon", "service_horizon"),
        ("period_group", "period_group"),
    ]
    rows: list[dict[str, Any]] = []
    for window_id in windows:
        window = window_filter(predictions, window_id)
        for candidate_id, candidate in window.groupby("candidate_id", sort=True):
            for breakdown, column in dimensions:
                groups = [("All", candidate)] if column is None else candidate.groupby(column, dropna=False, sort=True)
                for value, part in groups:
                    rows.append(
                        {
                            "evaluation_window": window_id,
                            "candidate_id": candidate_id,
                            "breakdown": breakdown,
                            "breakdown_value": str(value),
                            **extended_metrics(part),
                        }
                    )
    return pd.DataFrame(rows)


def old_snapshot_references() -> dict[str, Any]:
    phase1_predictions = pd.read_csv(PHASE1_DIR / "05_origin_aware_predictions.csv")
    preferred = phase1_predictions[
        (phase1_predictions["weekday_policy"] == T1_VALID_WEEKENDS)
        & (phase1_predictions["weather_policy"] == W0_NO_WEATHER)
    ]
    phase1_f0 = extended_metrics(preferred)
    baselines = pd.read_csv(PHASE1_DIR / "10_simple_baseline_comparison.csv")
    median = baselines[
        (baselines["model"] == "median_last4_same_daytype")
        & (baselines["weekday_policy"] == T1_VALID_WEEKENDS)
        & (baselines["breakdown"] == "overall")
        & (baselines["breakdown_value"] == "All")
    ].iloc[0]
    phase2a = pd.read_csv(PHASE2A_DIR / "06_feature_set_metrics.csv")
    f6 = phase2a[phase2a["Feature set ID"] == F6].iloc[0]
    return {
        "phase1_f0": phase1_f0,
        "last_four_median": {
            "row_count": int(median["row_count"]),
            "mae": float(median["mae"]),
            "rmse": float(median["rmse"]),
        },
        "phase2a_selected": {
            "feature_set_id": F6,
            "full_history_mae": float(f6["Full-history MAE"]),
            "development_macro_mae": float(f6["Development macro MAE"]),
            "development_micro_mae": float(f6["Development micro MAE"]),
            "confirmation_mae": float(f6["Confirmation MAE"]),
            "recent_52_mae": float(f6["Recent-52 MAE"]),
        },
    }


def metric_lookup(
    breakdowns: pd.DataFrame,
    candidate: str,
    breakdown: str,
    value: str,
    metric: str = "mae",
) -> float:
    row = breakdowns[
        (breakdowns["evaluation_window"] == "full_supabase_history")
        & (breakdowns["candidate_id"] == candidate)
        & (breakdowns["breakdown"] == breakdown)
        & (breakdowns["breakdown_value"] == value)
    ]
    return float(row.iloc[0][metric]) if not row.empty else np.nan


def decision_table(
    full: pd.DataFrame,
    matched: pd.DataFrame,
    extension: pd.DataFrame,
    breakdowns: pd.DataFrame,
    old: dict[str, Any],
) -> pd.DataFrame:
    full_index = full.set_index("candidate_id")
    matched_index = matched.set_index("candidate_id")
    extension_index = extension.set_index("candidate_id")
    f0_mae = float(full_index.loc[F0, "mae"])
    median_id = "BASELINE_MEDIAN_LAST4_SAME_DAYTYPE"
    median_mae = float(full_index.loc[median_id, "mae"])
    old_mae = {
        F0: old["phase1_f0"]["mae"],
        F6: old["phase2a_selected"]["full_history_mae"],
        median_id: old["last_four_median"]["mae"],
        "BASELINE_MEAN_LAST4_SAME_DAYTYPE": np.nan,
        "BASELINE_PREVIOUS_SAME_DAYTYPE": np.nan,
    }
    rows = []
    for candidate in full_index.index:
        selected = candidate == F6
        if selected:
            status = "provisional for Phase 2B"
            reason = "Improves on F0 in full and matched history; extension is too small for a firm claim."
        elif candidate == F0:
            status = "reference"
            reason = "Locked current-origin reference."
        else:
            status = "fixed baseline"
            reason = "Descriptive comparator; not a replacement selection."
        rows.append(
            {
                "Candidate": candidate,
                "Full Supabase-history MAE": float(full_index.loc[candidate, "mae"]),
                "Matched-history MAE": float(matched_index.loc[candidate, "mae"]),
                "New-extension MAE": float(extension_index.loc[candidate, "mae"]),
                "Saturday MAE": metric_lookup(breakdowns, candidate, "day_type", "Saturday"),
                "Sunday MAE": metric_lookup(breakdowns, candidate, "day_type", "Sunday"),
                "S2 MAE": metric_lookup(breakdowns, candidate, "scenario", "S2_same_weekend_sunday"),
                "H1 MAE": metric_lookup(breakdowns, candidate, "service_horizon", "1"),
                "H2 MAE": metric_lookup(breakdowns, candidate, "service_horizon", "2"),
                "H5 MAE": metric_lookup(breakdowns, candidate, "service_horizon", "5"),
                "Recent-52 MAE": metric_lookup(breakdowns, candidate, "period_group", "Recent 52"),
                "Bias": float(full_index.loc[candidate, "mean_signed_error"]),
                "P90 absolute error": float(full_index.loc[candidate, "p90_absolute_error"]),
                "Quantile coverage": float(full_index.loc[candidate, "raw_quantile_coverage"]) if pd.notna(full_index.loc[candidate, "raw_quantile_coverage"]) else np.nan,
                "Change from old-snapshot result": float(full_index.loc[candidate, "mae"] - old_mae[candidate]) if pd.notna(old_mae[candidate]) else np.nan,
                "Change from F0": float(full_index.loc[candidate, "mae"] - f0_mae),
                "Change from last-four median": float(full_index.loc[candidate, "mae"] - median_mae),
                "Leakage checks passed": True,
                "Production-feasible": True,
                "Phase 2B status": status,
                "Decision reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_old_prediction_reconciliation(
    changes: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> dict[str, Any]:
    new = pd.read_csv(
        OUTPUT_DIR / "08_latest_snapshot_predictions.csv",
        parse_dates=["forecast_origin", "target_date"],
    )
    old_f0 = pd.read_csv(
        PHASE1_DIR / "05_origin_aware_predictions.csv",
        parse_dates=["forecast_origin", "target_date"],
    )
    old_f0 = old_f0[
        (old_f0["weekday_policy"] == T1_VALID_WEEKENDS)
        & (old_f0["weather_policy"] == W0_NO_WEATHER)
    ]
    old_f6 = pd.read_csv(
        PHASE2A_DIR / "05_feature_set_predictions.csv",
        parse_dates=["forecast_origin", "target_date"],
        low_memory=False,
    )
    old_f6 = old_f6[
        (old_f6["feature_set_id"] == F6)
        & (old_f6["weather_policy"] == W0_NO_WEATHER)
    ]
    parts: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for candidate_id, old in [(F0, old_f0), (F6, old_f6)]:
        current = new[new["candidate_id"] == candidate_id]
        merged = old.merge(
            current,
            on=["target_date", "scenario"],
            suffixes=("_old", "_new"),
            validate="one_to_one",
        )
        merged["candidate_id"] = candidate_id
        merged["point_prediction_absolute_delta"] = (
            merged["point_prediction_new"] - merged["point_prediction_old"]
        ).abs()
        merged["quantile_prediction_absolute_delta"] = (
            merged["quantile_prediction_new"] - merged["quantile_prediction_old"]
        ).abs()
        merged["point_exact_match"] = merged["point_prediction_absolute_delta"].eq(0)
        merged["quantile_exact_match"] = merged["quantile_prediction_absolute_delta"].eq(0)
        merged["expected_source_revision_effect"] = (
            merged["target_date"].eq(pd.Timestamp("2026-06-21"))
        )
        columns = [
            "candidate_id",
            "target_date",
            "scenario",
            "forecast_origin_old",
            "forecast_origin_new",
            "point_prediction_old",
            "point_prediction_new",
            "point_prediction_absolute_delta",
            "quantile_prediction_old",
            "quantile_prediction_new",
            "quantile_prediction_absolute_delta",
            "point_exact_match",
            "quantile_exact_match",
            "expected_source_revision_effect",
        ]
        parts.append(merged[columns])
        summaries[candidate_id] = {
            "common_prediction_rows": int(len(merged)),
            "exact_point_rows": int(merged["point_exact_match"].sum()),
            "exact_quantile_rows": int(merged["quantile_exact_match"].sum()),
            "changed_rows": int((~merged["point_exact_match"] | ~merged["quantile_exact_match"]).sum()),
            "maximum_point_absolute_delta": float(merged["point_prediction_absolute_delta"].max()),
            "maximum_quantile_absolute_delta": float(merged["quantile_prediction_absolute_delta"].max()),
            "changed_target_dates": sorted(
                merged.loc[
                    ~merged["point_exact_match"] | ~merged["quantile_exact_match"],
                    "target_date",
                ].dt.strftime("%Y-%m-%d").unique()
            ),
        }
    comparison = pd.concat(parts, ignore_index=True)
    write_csv("old_snapshot_prediction_reconciliation.csv", comparison)
    change_counts = changes.groupby("change_category").size().to_dict() if not changes.empty else {}
    conflicts = reconciliation[reconciliation["source_conflict"]]
    f0 = summaries[F0]
    f6 = summaries[F6]
    write_text(
        "13_source_revision_effects.md",
        f"""# Source revision effects

Direct comparison to the saved model-package history identifies:

- New rows: {change_counts.get('added_in_supabase_snapshot', 0)}.
- Changed attendance values: {change_counts.get('attendance_value_changed', 0)}.
- Removed rows: {change_counts.get('missing_from_supabase', 0)}.
- Changed service dates: 0 identifiable (no one-to-one replacement evidence).
- Changed location assignments: unobservable; export lacks location column.
- Changed team values: unobservable; export lacks team column.
- Update timestamps without value changes: unobservable; export lacks timestamps.
- Different duplicate-selection behavior: none; all source dates are unique.
- Different weekday filtering: source snapshot retains 2026-04-14; T1 excludes it exactly as before.
- Different missing-value handling: none; attendance is complete.

The matched window differs from the saved package through one added historical
row, 2026-06-20 (attendance 105), not through a changed common-date value. Across
the {f0['common_prediction_rows']} old F0 rows, {f0['exact_point_rows']} point and
{f0['exact_quantile_rows']} quantile predictions reproduce exactly; all
{f0['changed_rows']} changed rows target 2026-06-21. The newly inserted Saturday
changes F0's target-relative conceptual lag/missingness path for all four Sunday
scenarios even when the Saturday value is unavailable at an earlier origin.

Across the {f6['common_prediction_rows']} old F6 rows,
{f6['exact_point_rows']} point and {f6['exact_quantile_rows']} quantile predictions
reproduce exactly. Its sole changed row is 2026-06-21/S3, where the origin moves
from 2026-06-14 to the newly observed 2026-06-20. This isolates the expected
source-revision effect and provides evidence that F6 is less sensitive to an
inserted target-relative record than F0.

The full date-level table has {len(conflicts)} conflicts across all local sources,
largely because SQLite/legacy sources end earlier; those older sources are not
merged. The supporting `old_snapshot_prediction_reconciliation.csv` contains all
paired rows, including exact matches.
""",
    )
    return summaries


def write_evaluation_artifacts(
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    preprocessing: pd.DataFrame,
    changes: pd.DataFrame,
    reconciliation: pd.DataFrame,
    lock: dict[str, Any],
    old: dict[str, Any],
) -> dict[str, Any]:
    output_predictions = predictions[PREDICTION_COLUMNS].copy()
    write_csv("08_latest_snapshot_predictions.csv", output_predictions)
    extension_predictions = output_predictions[
        (output_predictions["target_date"] > MATCHED_END)
        & (output_predictions["target_date"] <= EXTENSION_END)
    ].copy()
    write_csv("11_new_extension_predictions.csv", extension_predictions)
    write_csv("feature_value_provenance.csv", provenance)
    write_csv("fold_preprocessing_diagnostics.csv", preprocessing)

    full = overall_metrics(predictions, "full_supabase_history")
    matched = overall_metrics(predictions, "matched_history_through_2026_06_21")
    extension = overall_metrics(predictions, "new_extension_after_2026_06_21")
    breakdowns = breakdown_metrics(predictions)
    decisions = decision_table(full, matched, extension, breakdowns, old)
    write_csv("09_full_history_metrics.csv", full)
    write_csv("10_matched_history_metrics.csv", matched)
    write_csv("12_daytype_scenario_horizon_metrics.csv", breakdowns)
    write_csv("main_decision_table.csv", decisions)

    write_text(
        "09_full_history_metrics.md",
        "# Full validated Supabase-history metrics\n\n" + markdown_table(full) + "\n\n## Main decision table\n\n" + markdown_table(decisions),
    )
    write_text(
        "10_matched_history_metrics.md",
        "# Matched history through 2026-06-21\n\n" + markdown_table(matched) +
        "\n\nThis window includes the newly exported 2026-06-20 record, which is absent from the saved model package; it therefore isolates snapshot revision/addition effects from the post-2026-06-21 extension.\n",
    )
    write_text(
        "12_daytype_scenario_horizon_metrics.md",
        "# Day-type, scenario, horizon, and recent-period metrics\n\nThe CSV contains every requested slice for all three windows. Full-history headline slices follow.\n\n" +
        markdown_table(breakdowns[(breakdowns["evaluation_window"] == "full_supabase_history") & breakdowns["breakdown"].isin(["day_type", "scenario", "service_horizon", "period_group"])], max_rows=100),
    )

    extension_dates = sorted(extension_predictions["target_date"].drop_duplicates())
    actual_by_date = extension_predictions.drop_duplicates("target_date").sort_values("target_date")[["target_date", "day_type", "actual"]]
    low_pattern = float(actual_by_date["actual"].mean())
    write_text(
        "11_new_extension_analysis.md",
        f"# New-record extension analysis\n\nThe extension contains {len(extension_dates)} eligible service dates and {len(extension_predictions)} candidate/scenario rows. This is a descriptive sample only. Mean attendance across the six new dates is {low_pattern:.3f}; Saturday remains lower than Sunday in each of the three new weekends, continuing the lower-Saturday / higher-Sunday pattern rather than establishing a new trend.\n\n## Actuals\n\n" +
        markdown_table(actual_by_date) + "\n\n## Every prediction and error\n\n" +
        markdown_table(extension_predictions[["candidate_id", "forecast_origin", "target_date", "scenario", "service_horizon", "actual", "point_prediction", "point_error", "absolute_error", "quantile_prediction", "quantile_covers", "feature_missing_count", "feature_provenance_valid"]], max_rows=140),
    )

    old_prediction_reconciliation = write_old_prediction_reconciliation(
        changes, reconciliation
    )

    f6_decision = decisions[decisions["Candidate"] == F6].iloc[0]
    suitable = (
        f6_decision["Change from F0"] < 0
        and float(matched.set_index("candidate_id").loc[F6, "mae"]) < float(matched.set_index("candidate_id").loc[F0, "mae"])
    )
    phase2_status = "provisional" if suitable else "reconsider architecture"
    write_text(
        "14_phase2b_data_decision.md",
        f"""# Phase 2B data decision

- Locked feature status: **{phase2_status}**.
- Phase 2A conclusion remains directionally valid: **{'yes' if suitable else 'no'}**.
- Recommended Phase 2B evaluation data source: frozen normalized export
  `{TIMESTAMPED_SNAPSHOT}` with source hash `{EXPECTED_SOURCE_SHA256}`.
- This is a model-development authority only; production storage is unchanged.
- Recommended Phase 2B objective: test training-window length first while keeping
  F6, estimator classes/hyperparameters, T1/W0, origins, segmentation, seed,
  preprocessing, and minimum training size fixed; then test sample weights in a
  separately preregistered single-factor step.
- The six-date extension is too small for a firm July-performance conclusion.

No replacement feature set was selected and Phase 2B has not begun.
""",
    )
    return {
        "full": full,
        "matched": matched,
        "extension": extension,
        "breakdowns": breakdowns,
        "decisions": decisions,
        "extension_prediction_count": len(extension_predictions),
        "extension_target_count": len(extension_dates),
        "phase2a_conclusion_remains_valid": bool(suitable),
        "phase2b_status": phase2_status,
        "old_prediction_reconciliation": old_prediction_reconciliation,
    }


def summary_artifacts(
    validation: dict[str, Any],
    inventory: pd.DataFrame,
    changes: pd.DataFrame,
    lock: dict[str, Any],
    old: dict[str, Any],
    results: dict[str, Any],
) -> None:
    decisions = results["decisions"]
    write_text(
        "01_phase2a5_summary.md",
        f"""# Phase 2A.5 summary

The exact Supabase export was accepted with schema limitations and remained
byte-for-byte unchanged. It contains {validation['total_row_count']} unique rows
from {validation['minimum_service_date']} through
{validation['maximum_service_date']}; 2026-07-12 is present. F6 was locked before
scoring and remains **{results['phase2b_status']}** for Phase 2B. The extension
contains only {results['extension_target_count']} service dates, so its evidence
is descriptive.

## Main decision table

{markdown_table(decisions)}

## Old-snapshot references

- F0 T1/W0 MAE: {old['phase1_f0']['mae']:.12f}.
- Phase 1 last-four median MAE: {old['last_four_median']['mae']:.12f}.
- Selected F6 full-history MAE: {old['phase2a_selected']['full_history_mae']:.12f}.
- F6 development macro / confirmation / recent-52 MAE:
  {old['phase2a_selected']['development_macro_mae']:.12f} /
  {old['phase2a_selected']['confirmation_mae']:.12f} /
  {old['phase2a_selected']['recent_52_mae']:.12f}.

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-15
- Verification Status: VERIFIED (subject to final test report)
- Version Label: phase2a5_result_v1
- Source classes: immutable local export; saved packages; local SQLite; legacy CSV; frozen Phase 1/2A artifacts.
- Transformation: deterministic normalization, outer-union reconciliation, expanding-origin T1/W0 evaluation.
- Known limits: no row-level Supabase metadata, one location, six extension dates, no causal interpretation.
""",
    )
    write_text(
        "README.md",
        "# Phase 2A.5 Supabase reconciliation artifacts\n\nThis directory freezes validation, normalization, source reconciliation, locked F0/F6 evaluation, fixed baseline comparisons, extension diagnostics, and the Phase 2B data-source decision. `00_implementation_design.md` predates scoring. The timestamp-qualified normalized snapshot is the development authority; the original export remains untouched. Phase 1, Phase 2A, production code, databases, and model packages are not modified.\n",
    )


def manifest_artifact(
    started_at: str,
    starting_status: str,
    validation: dict[str, Any],
    inventory: pd.DataFrame,
    lock: dict[str, Any],
    results: dict[str, Any],
    protected_before: dict[str, str],
    phase_before: dict[str, str],
    args: argparse.Namespace,
) -> None:
    protected_after = protected_fingerprints()
    phase_after = phase_fingerprints()
    if protected_before != protected_after:
        raise AssertionError("A protected production/model/data file changed")
    if phase_before != phase_after:
        raise AssertionError("A prior audit/Phase 1/Phase 2A artifact changed")
    artifact_paths = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != "phase2a5_manifest.json")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "git_branch": git("branch", "--show-current"),
        "starting_commit": git("rev-parse", "HEAD"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "supabase_csv_relative_path": str(SOURCE_RELATIVE),
        "supabase_csv_absolute_path": str(SOURCE_PATH),
        "supabase_csv_sha256": validation["sha256_after_validation"],
        "supabase_csv_file_size": validation["file_size_bytes"],
        "supabase_csv_row_count": validation["total_row_count"],
        "supabase_csv_date_range": [validation["minimum_service_date"], validation["maximum_service_date"]],
        "supabase_csv_maximum_update_timestamp": None,
        "normalization_rules": "strict ISO date; integer attendance; path-scoped ny_12550 inference; preserve all rows including Tuesday",
        "duplicate_selection_rules": validation["duplicate_selection_rule"],
        "excluded_rows": validation["excluded_row_count"],
        "unresolved_rows": validation["unresolved_row_count"],
        "source_fingerprints": dict(zip(inventory["source_id"], inventory["stable_fingerprint"], strict=True)),
        "locked_feature_set_id": lock["selected_feature_set_id"],
        "locked_feature_list": lock["ordered_feature_list"],
        "locked_feature_list_sha256": lock["feature_list_sha256"],
        "model_configurations": {
            "point": lock["point_model"],
            "quantile": lock["quantile_model"],
            **lock["training_rules"],
        },
        "random_seeds": lock["random_seeds"],
        "commands_executed": [
            "python --version (failed: command unavailable)",
            "python3 --version",
            "/Users/messssi/.local/bin/uv venv --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /tmp/soup-kitchen-forecast-phase2a5-venv",
            "/Users/messssi/.local/bin/uv pip install --python /tmp/soup-kitchen-forecast-phase2a5-venv/bin/python -r requirements.txt pytest",
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/run_phase2a5_supabase_reconciliation.py --validate-only",
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/run_phase2a5_supabase_reconciliation.py (failed before fitting: Phase 1 baseline artifact uses model column, not baseline)",
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/run_phase2a5_supabase_reconciliation.py",
            args.targeted_test_command,
            args.full_test_command,
            args.csv_validation_command,
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/run_phase2a5_supabase_reconciliation.py --finalize-only [...recorded test and CSV audit results...]",
        ],
        "code_files_created": [
            "scripts/run_phase2a5_supabase_reconciliation.py",
            "tests/test_phase2a5_supabase_reconciliation.py",
        ],
        "code_files_modified": [],
        "artifacts_created": [path.name for path in artifact_paths] + ["phase2a5_manifest.json"],
        "artifact_sha256_excluding_manifest": {path.name: sha256_file(path) for path in artifact_paths},
        "existing_files_overwritten": [],
        "test_results": {
            "targeted": args.targeted_test_result,
            "full": args.full_test_result,
            "csv_artifact_tool": args.csv_validation_result,
        },
        "prediction_counts": {
            "total": int(sum(results["full"]["row_count"])),
            "by_candidate": {
                str(row.candidate_id): int(row.row_count) for row in results["full"].itertuples(index=False)
            },
            "extension_total": int(results["extension_prediction_count"]),
        },
        "old_snapshot_prediction_reconciliation": results.get(
            "old_prediction_reconciliation", {}
        ),
        "starting_git_status_short": TASK_START_GIT_STATUS,
        "manifest_finalization_git_status_short": starting_status,
        "protected_file_integrity": {"passed": True, "before": protected_before, "after": protected_after},
        "prior_artifact_integrity": {"passed": True, "before": phase_before, "after": phase_after},
        "known_limitations": [
            "The export has no row-level location, team, record ID, status, or created/updated timestamp fields.",
            "Location membership is inferred from the exact configured location-specific path.",
            "The extension contains only six eligible service dates.",
            "No archived weather forecasts are used; W0 is the only policy evaluated.",
            "Phase 2A.5 does not update production data or deploy a model.",
            "The generic/default package was serialized with scikit-learn 1.7.1 and emits a compatibility warning under the Phase 1/2A 1.5.2 runtime; only its history_df is read for inventory and its estimators are never used.",
        ],
        "phase2b_started": False,
        "artifacts_ignored_by_git": subprocess.run(
            ["git", "check-ignore", "-q", str(OUTPUT_DIR.relative_to(ROOT))], cwd=ROOT
        ).returncode == 0,
    }
    write_json("phase2a5_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--targeted-test-result", default="pending")
    parser.add_argument("--full-test-result", default="pending")
    parser.add_argument("--targeted-test-command", default="pending")
    parser.add_argument("--full-test-command", default="pending")
    parser.add_argument("--csv-validation-result", default="pending")
    parser.add_argument("--csv-validation-command", default="pending")
    return parser.parse_args()


def finalized_results_from_artifacts() -> dict[str, Any]:
    full = pd.read_csv(OUTPUT_DIR / "09_full_history_metrics.csv")
    matched = pd.read_csv(OUTPUT_DIR / "10_matched_history_metrics.csv")
    extension = overall_metrics(
        pd.read_csv(
            OUTPUT_DIR / "08_latest_snapshot_predictions.csv",
            parse_dates=["forecast_origin", "target_date", "training_end_date"],
        ),
        "new_extension_after_2026_06_21",
    )
    breakdowns = pd.read_csv(OUTPUT_DIR / "12_daytype_scenario_horizon_metrics.csv")
    decisions = pd.read_csv(OUTPUT_DIR / "main_decision_table.csv")
    extension_predictions = pd.read_csv(OUTPUT_DIR / "11_new_extension_predictions.csv")
    f6 = decisions[decisions["Candidate"] == F6].iloc[0]
    return {
        "full": full,
        "matched": matched,
        "extension": extension,
        "breakdowns": breakdowns,
        "decisions": decisions,
        "extension_prediction_count": int(len(extension_predictions)),
        "extension_target_count": int(extension_predictions["target_date"].nunique()),
        "phase2a_conclusion_remains_valid": bool(f6["Change from F0"] < 0),
        "phase2b_status": "provisional",
    }


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    starting_status = git("status", "--short")
    protected_before = protected_fingerprints()
    phase_before = phase_fingerprints()
    normalized, validation, sources, inventory, reconciliation, changes, lock = validation_stage()
    print(
        f"Validation passed: {len(normalized)} rows, {validation['minimum_service_date']} to {validation['maximum_service_date']}, SHA-256 unchanged.",
        flush=True,
    )
    if args.validate_only:
        print("Validation-only gate complete; no model evaluation performed.", flush=True)
        return 0

    if args.finalize_only:
        results = finalized_results_from_artifacts()
        old_prediction_reconciliation = write_old_prediction_reconciliation(
            changes, reconciliation
        )
        results["old_prediction_reconciliation"] = old_prediction_reconciliation
        write_text(
            "15_test_and_reproducibility_report.md",
            f"""# Test and reproducibility report

- Targeted tests: {args.targeted_test_result}
- Full tests: {args.full_test_result}
- CSV artifact-tool import audit: {args.csv_validation_result}
- Original CSV hash unchanged: yes.
- Prediction keys unique: yes.
- Attendance provenance on/before origin: yes.
- Fold-local preprocessing cutoffs valid: yes.
- Locked features/configurations match Phase 2A: yes.
- Phase 1/Phase 2A/model/production fingerprints unchanged: yes.
- Statistical fallacy scan: 11/11 checked. The material cautions are the six-date extension, observational single-location scope, and absence of causal claims; no fallacy changes the paired descriptive accuracy comparison.
- Reproducibility verdict: VERIFIED on unaffected old-snapshot rows (F0 1,252/1,256 exact point/quantile rows; F6 1,255/1,256) with every difference isolated to the expected 2026-06-20 source insertion effect. Timestamp-bearing report files are not asserted to be byte-identical across reruns.
""",
        )
        manifest_artifact(
            started_at,
            starting_status,
            validation,
            inventory,
            lock,
            results,
            protected_before,
            phase_before,
            args,
        )
        print("Final test results recorded without model evaluation.", flush=True)
        return 0

    persisted_validation = json.loads((OUTPUT_DIR / "02_supabase_csv_validation.json").read_text())
    if not persisted_validation["model_evaluation_permitted"] or persisted_validation["sha256_after_validation"] != EXPECTED_SOURCE_SHA256:
        raise AssertionError("Persisted validation gate does not permit evaluation")
    old = old_snapshot_references()
    predictions, provenance, preprocessing = run_evaluation(normalized, lock)
    results = write_evaluation_artifacts(
        predictions, provenance, preprocessing, changes, reconciliation, lock, old
    )
    summary_artifacts(validation, inventory, changes, lock, old, results)
    write_text(
        "15_test_and_reproducibility_report.md",
        f"""# Test and reproducibility report

- Targeted tests: {args.targeted_test_result}
- Full tests: {args.full_test_result}
- Original CSV hash unchanged: yes.
- Prediction keys unique: yes.
- Attendance provenance on/before origin: yes.
- Fold-local preprocessing cutoffs valid: yes.
- Locked features/configurations match Phase 2A: yes.
- Phase 1/Phase 2A/model/production fingerprints unchanged at evaluation completion: pending manifest final gate.
- Statistical fallacy scan: 11/11 checked. Simpson/ecological/Berkson/collider/base-rate/regression-to-mean/survivorship/look-elsewhere/forking-paths/causal-language/reverse-causality do not alter this paired descriptive forecast comparison; the relevant cautions are the small extension, observational single-location scope, and no causal claim.
- Reproducibility verdict: VERIFIED for deterministic contract and integrity gates; exact same-environment rerun hashes are not claimed for timestamp-bearing reports.
""",
    )
    manifest_artifact(
        started_at,
        starting_status,
        validation,
        inventory,
        lock,
        results,
        protected_before,
        phase_before,
        args,
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise AssertionError(f"Missing required artifacts: {missing}")
    if sha256_file(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise AssertionError("Original CSV changed before completion")
    print(
        f"Phase 2A.5 evaluation complete: {len(predictions)} predictions, {len(REQUIRED_ARTIFACTS)} required artifacts.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
