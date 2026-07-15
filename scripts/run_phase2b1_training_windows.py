#!/usr/bin/env python3
"""Run the preregistered Phase 2B1 segment training-window experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase2a5_supabase_reconciliation import (
    EXPECTED_SOURCE_SHA256,
    SOURCE_PATH,
    attendance_feature_columns,
    directory_fingerprint,
    history_from_normalized,
    json_value,
    locked_feature_contract,
    markdown_table,
    sha256_file,
    sha256_json,
)
from src.feature_sets import F6, build_feature_set_registry, make_repaired_feature_builder
from src.origin_backtest import OriginAwareBacktester, calculate_metrics
from src.origin_features import T1_VALID_WEEKENDS, W0_NO_WEATHER
from src.training_windows import (
    TRAINING_WINDOW_CANDIDATES,
    TrainingWindowDefinition,
    registry_rows,
    select_training_window,
    validate_candidate_ids,
)


LOCATION_ID = "ny_12550"
OUTPUT_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2b1_training_windows"
PHASE1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
PHASE2A_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a_feature_repair"
PHASE2A5_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation"
SNAPSHOT_PATH = PHASE2A5_DIR / "03_normalized_supabase_snapshot_2026-07-15T05-23.csv"
REFERENCE_PREDICTIONS_PATH = PHASE2A5_DIR / "08_latest_snapshot_predictions.csv"
EXPECTED_F6_HASH = "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
DATA_SNAPSHOT_ID = "ny_12550_supabase_export_2026-07-15T05-23_e3f84ac47245"
MATCHED_END = pd.Timestamp("2026-06-21")
EXTENSION_END = pd.Timestamp("2026-07-12")
DEVELOPMENT_END = {"Saturday": pd.Timestamp("2025-05-24"), "Sunday": pd.Timestamp("2025-05-18")}
CONFIRMATION_START = {"Saturday": pd.Timestamp("2025-05-31"), "Sunday": pd.Timestamp("2025-05-25")}
POINT_RANDOM_SEED = 42
QUANTILE_RANDOM_SEED = 42
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_REPLICATIONS = 1000
REPRODUCTION_ATOL = 1e-10

PROTECTED_FILES = [
    ROOT / "app.py",
    ROOT / "src/predictor.py",
    ROOT / "src/features.py",
    ROOT / "src/modeling.py",
    ROOT / "scripts/train_model.py",
    ROOT / "models/visitor_model_ny_12550.joblib",
    ROOT / "models/visitor_model.joblib",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_clean(value: Any) -> Any:
    """Convert pandas/NumPy missing scalars to strict JSON null recursively."""

    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(
            json_clean(value),
            indent=2,
            sort_keys=True,
            default=json_value,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_text(name: str, value: str) -> None:
    (OUTPUT_DIR / name).write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(name: str, frame: pd.DataFrame) -> None:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_DIR / name, index=False, lineterminator="\n")


def protected_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in PROTECTED_FILES
        if path.is_file()
    }


def prior_phase_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): directory_fingerprint(path)
        for path in [PHASE1_DIR, PHASE2A_DIR, PHASE2A5_DIR]
    }


def validate_snapshot() -> pd.DataFrame:
    if sha256_file(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise AssertionError("The authoritative Supabase export hash changed")
    if not SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(SNAPSHOT_PATH)
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    if len(snapshot) != 360:
        raise AssertionError(f"Expected 360 snapshot rows, found {len(snapshot)}")
    dates = pd.to_datetime(snapshot["service_date"], format="%Y-%m-%d", errors="raise")
    attendance = pd.to_numeric(snapshot["attendance"], errors="raise")
    if dates.min() != pd.Timestamp("2023-01-01") or dates.max() != EXTENSION_END:
        raise AssertionError("Snapshot date range differs from the locked contract")
    if snapshot["service_date"].duplicated().any() or attendance.isna().any():
        raise AssertionError("Snapshot has duplicate dates or missing attendance")
    non_weekends = dates[~dates.dt.weekday.isin([5, 6])]
    if non_weekends.tolist() != [pd.Timestamp("2026-04-14")]:
        raise AssertionError("The locked T1 exclusion set changed")
    return snapshot


def validate_lock() -> dict[str, Any]:
    lock = locked_feature_contract()
    if lock["selected_feature_set_id"] != F6 or lock["feature_count"] != 33:
        raise AssertionError("Locked F6 identity/count changed")
    if lock["feature_list_sha256"] != EXPECTED_F6_HASH:
        raise AssertionError("Locked F6 feature hash changed")
    return lock


def period_role(target: pd.Timestamp, day_type: str) -> str:
    target = pd.Timestamp(target).normalize()
    return "development" if target <= DEVELOPMENT_END[day_type] else "confirmation"


def make_evaluator(
    history: pd.DataFrame,
    lock: dict[str, Any],
    window: TrainingWindowDefinition,
) -> OriginAwareBacktester:
    features = list(lock["ordered_feature_list"])
    registry = build_feature_set_registry(
        f5_parent_id="F4_CORRECTED_SLOT_HISTORY",
        f6_features=features,
        f6_groups=[
            "calendar",
            "last_observed_daytype",
            "daytype_summaries",
            "daytype_slot",
            "horizon_availability",
        ],
    )
    return OriginAwareBacktester(
        history,
        weather_df=None,
        feature_cols=features,
        residual_buffer_by_day={"sat": 0.0, "sun": 0.0},
        default_meal_buffer_pct=0.0,
        min_train_size=18,
        quantile=0.8,
        feature_set_id=F6,
        feature_builder=make_repaired_feature_builder(registry[F6]),
        attendance_feature_cols=attendance_feature_columns(features),
        random_seed=POINT_RANDOM_SEED,
        training_window=window,
    )


def enrich_predictions(frame: pd.DataFrame, lock: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column in ["forecast_origin", "target_date", "training_end_date", "retained_training_start_date", "retained_training_end_date"]:
        out[column] = pd.to_datetime(out[column]).dt.normalize()
    out["data_snapshot_id"] = DATA_SNAPSHOT_ID
    out["feature_set_hash"] = lock["feature_list_sha256"]
    out["period_role"] = [period_role(date, day) for date, day in zip(out["target_date"], out["day_type"], strict=True)]
    out["effective_window_rows"] = out["retained_segment_training_rows"]
    out["earliest_retained_training_date"] = out["retained_training_start_date"]
    out["latest_retained_training_date"] = out["retained_training_end_date"]
    out["effective_history_days"] = out["effective_window_days"]
    out["effective_history_years"] = out["effective_window_years"]
    out["preprocessing_cutoff_valid"] = out["training_end_date"] <= out["forecast_origin"]
    out["point_random_seed"] = POINT_RANDOM_SEED
    out["quantile_random_seed"] = QUANTILE_RANDOM_SEED
    eligible = sorted(out["target_date"].drop_duplicates())
    recent52 = set(eligible[-52:])
    out["recent_period"] = np.where(out["target_date"].isin(recent52), "Recent 52", "Earlier")
    out["matched_history"] = out["target_date"] <= MATCHED_END
    out["new_extension"] = (out["target_date"] > MATCHED_END) & (out["target_date"] <= EXTENSION_END)
    return out


PREDICTION_KEY = [
    "data_snapshot_id",
    "feature_set_id",
    "training_window_id",
    "forecast_origin",
    "target_date",
    "scenario",
    "weather_policy",
    "weekday_policy",
]
ALIGNMENT_KEY = [
    "data_snapshot_id",
    "feature_set_id",
    "forecast_origin",
    "target_date",
    "scenario",
    "weather_policy",
    "weekday_policy",
]


def check_expanding_reproduction(expanding: pd.DataFrame) -> dict[str, Any]:
    prior = pd.read_csv(
        REFERENCE_PREDICTIONS_PATH,
        parse_dates=["forecast_origin", "target_date", "training_end_date"],
    )
    prior = prior[prior["candidate_id"] == F6].copy()
    prior["feature_set_id"] = F6
    prior["data_snapshot_id"] = DATA_SNAPSHOT_ID
    left = expanding.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    right = prior.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    if len(left) != len(right) or not left[ALIGNMENT_KEY].equals(right[ALIGNMENT_KEY]):
        raise AssertionError("TW_EXPANDING prediction keys do not exactly align with Phase 2A.5 F6")
    point_diff = np.abs(left["point_prediction"].to_numpy() - right["point_prediction"].to_numpy())
    quantile_diff = np.abs(left["quantile_prediction"].to_numpy() - right["quantile_prediction"].to_numpy())
    actual_diff = np.abs(left["actual"].to_numpy() - right["actual"].to_numpy())
    payload = {
        "status": "passed",
        "reference_file": str(REFERENCE_PREDICTIONS_PATH.relative_to(ROOT)),
        "aligned_prediction_rows": int(len(left)),
        "exact_key_alignment": True,
        "maximum_actual_difference": float(actual_diff.max(initial=0.0)),
        "maximum_point_prediction_difference": float(point_diff.max(initial=0.0)),
        "maximum_quantile_prediction_difference": float(quantile_diff.max(initial=0.0)),
        "absolute_tolerance": REPRODUCTION_ATOL,
    }
    if payload["maximum_actual_difference"] != 0 or payload["maximum_point_prediction_difference"] > REPRODUCTION_ATOL or payload["maximum_quantile_prediction_difference"] > REPRODUCTION_ATOL:
        raise AssertionError(f"TW_EXPANDING reproduction failed: {payload}")
    return payload


def extended_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = calculate_metrics(frame)
    usable_q = frame.dropna(subset=["actual", "quantile_prediction"])
    uncovered = usable_q[usable_q["actual"] > usable_q["quantile_prediction"]]
    covered = usable_q[usable_q["actual"] <= usable_q["quantile_prediction"]]
    metrics.update(
        {
            "mean_quantile_shortfall_when_uncovered": float((uncovered["actual"] - uncovered["quantile_prediction"]).mean()) if not uncovered.empty else 0.0,
            "mean_quantile_excess_when_covered": float((covered["quantile_prediction"] - covered["actual"]).mean()) if not covered.empty else 0.0,
            "mean_available_segment_training_rows": float(frame["available_segment_training_rows"].mean()),
            "minimum_available_segment_training_rows": int(frame["available_segment_training_rows"].min()),
            "maximum_available_segment_training_rows": int(frame["available_segment_training_rows"].max()),
            "mean_retained_segment_training_rows": float(frame["retained_segment_training_rows"].mean()),
            "minimum_retained_segment_training_rows": int(frame["retained_segment_training_rows"].min()),
            "maximum_retained_segment_training_rows": int(frame["retained_segment_training_rows"].max()),
            "mean_effective_history_days": float(frame["effective_history_days"].mean()),
            "mean_effective_history_years": float(frame["effective_history_years"].mean()),
            "window_constrained_fold_count": int(frame["window_constrained"].sum()),
            "window_constrained_fold_pct": float(frame["window_constrained"].mean() * 100),
        }
    )
    return metrics


def development_recent_tail(frame: pd.DataFrame) -> pd.DataFrame:
    development = frame[frame["period_role"] == "development"]
    dates: set[pd.Timestamp] = set()
    for _, part in development[["target_date", "day_type"]].drop_duplicates().groupby("day_type"):
        dates.update(part.sort_values("target_date").tail(26)["target_date"])
    return development[development["target_date"].isin(dates)]


def development_selection_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_id, part in predictions.groupby("training_window_id", sort=False):
        development = part[part["period_role"] == "development"]
        scenario_mae = development.groupby("scenario")["absolute_error"].mean()
        tail = development_recent_tail(part)
        metrics = extended_metrics(development)
        rows.append(
            {
                "training_window_id": window_id,
                "configured_window_rows": part["configured_window_rows"].iloc[0],
                "development_macro_s1_s5_mae": float(scenario_mae.mean()),
                "development_micro_mae": metrics["mae"],
                "development_recent_tail_mae": float(tail["absolute_error"].mean()),
                "development_s2_mae": float(development.loc[development["scenario"] == "S2_same_weekend_sunday", "absolute_error"].mean()),
                "development_p90_absolute_error": metrics["p90_absolute_error"],
                "development_mean_signed_error": metrics["mean_signed_error"],
                "development_row_count": metrics["row_count"],
                "development_recent_tail_row_count": int(len(tail)),
            }
        )
    return pd.DataFrame(rows)


def write_prerun_contract(lock: dict[str, Any], snapshot: pd.DataFrame) -> dict[str, Any]:
    registry = registry_rows()
    validate_candidate_ids(item["training_window_id"] for item in registry)
    write_json("02_training_window_registry.json", registry)
    write_text(
        "02_training_window_registry.md",
        "# Training-window registry\n\n" + markdown_table(pd.DataFrame(registry)) +
        "\n\nWindows are applied after T1 filtering and segment assignment, and before fold-local imputation/model fitting. If fewer than N rows are available, every available row is retained and the fold is marked unconstrained.",
    )
    contract = {
        "contract_frozen_at_utc": now_iso(),
        "data_snapshot_id": DATA_SNAPSHOT_ID,
        "snapshot_path": str(SNAPSHOT_PATH.relative_to(ROOT)),
        "snapshot_sha256": sha256_file(SNAPSHOT_PATH),
        "source_export_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_export_sha256": EXPECTED_SOURCE_SHA256,
        "snapshot_row_count": int(len(snapshot)),
        "snapshot_date_range": [snapshot["service_date"].min(), snapshot["service_date"].max()],
        "t1_excluded_dates": ["2026-04-14"],
        "feature_set_id": F6,
        "feature_set_hash": lock["feature_list_sha256"],
        "ordered_feature_list": lock["ordered_feature_list"],
        "training_window_registry": registry,
        "development_end": {key: value.strftime("%Y-%m-%d") for key, value in DEVELOPMENT_END.items()},
        "confirmation_start": {key: value.strftime("%Y-%m-%d") for key, value in CONFIRMATION_START.items()},
        "selection_hierarchy": ["development_macro_s1_s5_mae", "development_recent_tail_mae", "development_s2_mae", "development_p90_absolute_error", "absolute_development_bias"],
        "longer_window_tolerance_mae": 0.10,
        "bootstrap": {"method": "target-date cluster bootstrap", "seed": BOOTSTRAP_SEED, "replications": BOOTSTRAP_REPLICATIONS},
        "model_and_feature_history_rule": "Full origin-available history builds features; the segment window limits model-fitting examples only.",
        "minimum_segment_rows": 18,
        "weather_policy": W0_NO_WEATHER,
        "weekday_policy": T1_VALID_WEEKENDS,
        "protected_fingerprints_before": protected_fingerprints(),
        "prior_phase_fingerprints_before": prior_phase_fingerprints(),
        "starting_branch": git("branch", "--show-current"),
        "starting_commit": git("rev-parse", "HEAD"),
    }
    write_json("03_locked_contract.json", contract)
    write_text(
        "03_locked_contract.md",
        f"# Locked Phase 2B1 contract\n\nThe contract was frozen at `{contract['contract_frozen_at_utc']}` before candidate scoring. F6 has 33 ordered features and hash `{lock['feature_list_sha256']}`. The authoritative snapshot has {len(snapshot)} rows from {snapshot['service_date'].min()} through {snapshot['service_date'].max()}; the source-export hash is `{EXPECTED_SOURCE_SHA256}`.\n\nSelection uses development rows only. Confirmation summaries are generated by a separate finalization script after `08_locked_window_decision.json` exists.",
    )
    return contract


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_DIR / "00_implementation_design.md").is_file():
        raise FileNotFoundError("00_implementation_design.md must predate implementation/evaluation")
    snapshot = validate_snapshot()
    lock = validate_lock()
    contract = write_prerun_contract(lock, snapshot)
    history = history_from_normalized(snapshot)
    prediction_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    expanding_reproduction: dict[str, Any] | None = None
    for index, window in enumerate(TRAINING_WINDOW_CANDIDATES):
        print(f"Running {window.training_window_id} ({index + 1}/{len(TRAINING_WINDOW_CANDIDATES)})...", flush=True)
        evaluator = make_evaluator(history, lock, window)
        result = evaluator.run(weather_policies=[W0_NO_WEATHER], weekday_policies=[T1_VALID_WEEKENDS])
        predictions = enrich_predictions(result.predictions, lock)
        if predictions.duplicated(PREDICTION_KEY).any():
            raise AssertionError(f"Duplicate prediction keys for {window.training_window_id}")
        if not predictions["preprocessing_cutoff_valid"].all() or not predictions["feature_provenance_valid"].all():
            raise AssertionError(f"Origin/cutoff validation failed for {window.training_window_id}")
        if not (predictions["earliest_retained_training_date"] <= predictions["latest_retained_training_date"]).all():
            raise AssertionError("Invalid retained training date order")
        if not (predictions["latest_retained_training_date"] <= predictions["forecast_origin"]).all():
            raise AssertionError("Retained training target exceeds forecast origin")
        if window.training_window_id == "TW_EXPANDING":
            expanding_reproduction = check_expanding_reproduction(predictions)
            write_text(
                "04_expanding_reproduction.md",
                "# TW_EXPANDING reproduction gate\n\nStatus: **passed**. The Phase 2B1 expanding run has exact key alignment with the Phase 2A.5 F6 rows.\n\n" + markdown_table(pd.DataFrame([expanding_reproduction])),
            )
            provenance = result.feature_provenance
            attendance = provenance[provenance["source_type"] == "attendance"]
            for row in attendance.itertuples(index=False):
                if any(pd.Timestamp(value) > pd.Timestamp(row.forecast_origin) for value in json.loads(row.available_source_dates)):
                    raise AssertionError("Feature provenance contains a post-origin attendance date")
            write_csv("supporting_feature_value_provenance.csv", provenance)
        prediction_parts.append(predictions)
        diagnostics = predictions[
            [
                "training_window_id", "forecast_origin", "target_date", "scenario", "day_type",
                "available_segment_training_rows", "retained_segment_training_rows",
                "earliest_retained_training_date", "latest_retained_training_date",
                "effective_history_days", "effective_history_years", "window_constrained",
                "preprocessing_id", "point_random_seed", "quantile_random_seed",
            ]
        ].copy()
        diagnostic_parts.append(diagnostics)
        write_csv(f"supporting_{window.training_window_id.lower()}_predictions.csv", predictions)
        print(f"Completed {window.training_window_id}: {len(predictions)} predictions.", flush=True)
    if expanding_reproduction is None:
        raise AssertionError("Expanding reproduction gate did not execute")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    counts = predictions.groupby("training_window_id").size()
    if counts.nunique() != 1 or counts.to_dict() != {item.training_window_id: 1284 for item in TRAINING_WINDOW_CANDIDATES}:
        raise AssertionError(f"Prediction-key counts do not align: {counts.to_dict()}")
    reference_keys = None
    for window_id, part in predictions.groupby("training_window_id"):
        keys = part[ALIGNMENT_KEY].sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise AssertionError(f"Paired keys do not align for {window_id}")
    write_csv("05_training_window_predictions.csv", predictions.sort_values(PREDICTION_KEY, kind="stable"))
    write_csv("16_fold_training_diagnostics.csv", diagnostics.sort_values(["training_window_id", "target_date", "scenario"], kind="stable"))

    raw_selection = development_selection_rows(predictions)
    selected, selection_audit = select_training_window(raw_selection)
    selection = raw_selection.merge(
        selection_audit[["training_window_id", "within_macro_tolerance", "longer_window_rank", "strict_hierarchy_rank", "selected"]],
        on="training_window_id", validate="one_to_one",
    )
    selection["selection_status"] = np.where(selection["selected"], "selected_and_locked", "rejected")
    selection["selection_reason"] = np.where(
        selection["selected"],
        "Locked development hierarchy with 0.10-MAE longer-window stability tolerance",
        "Not selected by locked development hierarchy and stability tolerance",
    )
    write_csv("07_development_selection_table.csv", selection)
    write_text(
        "07_development_selection_table.md",
        "# Development-only window selection\n\nThis table was calculated without confirmation metrics. The selected window is locked before the separate finalizer reads or aggregates confirmation rows.\n\n" + markdown_table(selection),
    )
    decision = {
        "locked_at_utc": now_iso(),
        "selected_training_window_id": selected,
        "configured_window_rows": next(item.configured_window_rows for item in TRAINING_WINDOW_CANDIDATES if item.training_window_id == selected),
        "selection_input_columns": [
            "development_macro_s1_s5_mae", "development_recent_tail_mae", "development_s2_mae",
            "development_p90_absolute_error", "development_mean_signed_error",
        ],
        "confirmation_metrics_read_or_calculated_before_lock": False,
        "selection_rule": contract["selection_hierarchy"],
        "longer_window_tolerance_mae": 0.10,
        "development_selection_records": selection.to_dict(orient="records"),
        "decision_sha256": hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest(),
    }
    write_json("08_locked_window_decision.json", decision)
    write_text(
        "08_locked_window_decision.md",
        f"# Locked training-window decision\n\n**{selected}** was locked at `{decision['locked_at_utc']}` using development evidence only. Confirmation metrics had not been read or calculated by this selection path. The 0.10-MAE stability rule prefers a longer window among candidates sufficiently close to the best development macro MAE.\n",
    )
    write_json(
        "supporting_run_state.json",
        {
            "run_completed_at_utc": now_iso(),
            "prediction_counts": {key: int(value) for key, value in counts.items()},
            "expanding_reproduction": expanding_reproduction,
            "feature_provenance_valid": True,
            "preprocessing_cutoff_valid": True,
            "retained_training_cutoff_valid": True,
            "paired_key_alignment_valid": True,
        },
    )
    print(f"Development lock written: {selected}. Run finalize_phase2b1_reports.py next.", flush=True)


if __name__ == "__main__":
    run()
