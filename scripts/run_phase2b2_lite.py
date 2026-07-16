#!/usr/bin/env python3
"""Run the minimal Phase 2B2-Lite recency-weight falsification test."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase2a5_supabase_reconciliation import (
    attendance_feature_columns,
    directory_fingerprint,
    history_from_normalized,
    locked_feature_contract,
    sha256_file,
)
from src.feature_sets import F6, build_feature_set_registry, make_repaired_feature_builder
from src.modeling import make_point_model, make_quantile_model
from src.origin_backtest import OriginAwareBacktester
from src.origin_features import T1_VALID_WEEKENDS, W0_NO_WEATHER, apply_calibration_placeholder
from src.sample_weights import (
    SAMPLE_WEIGHT_CANDIDATES,
    SW_UNIFORM,
    apply_confirmation_guardrail,
    candidate_registry_rows,
    effective_sample_size,
    generate_sample_weights,
    select_development_policy,
)
from src.training_windows import TW_EXPANDING


OUTPUT_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2b2_lite_sample_weights"
SNAPSHOT_PATH = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation/03_normalized_supabase_snapshot_2026-07-15T05-23.csv"
SOURCE_PATH = ROOT / "data/locations/ny_12550/Updated/2026-07-15T05-23_export.csv"
PHASE1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
PHASE2A_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a_feature_repair"
PHASE2A5_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation"
PHASE2B1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2b1_training_windows"
PHASE2B1_PREDICTIONS = PHASE2B1_DIR / "05_training_window_predictions.csv"
EXPECTED_SNAPSHOT_SHA256 = "eb3b6b5cfd4ed38718b21d401d715eee9ef0efb19b8272398826077b9e34ffed"
EXPECTED_SOURCE_SHA256 = "e3f84ac47245fa7eb5496413dbd04c5c0d0fead2ed553e257da57c3278ffdef8"
EXPECTED_F6_HASH = "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
DATA_SNAPSHOT_ID = "ny_12550_supabase_export_2026-07-15T05-23_e3f84ac47245"
DEVELOPMENT_END = {"Saturday": pd.Timestamp("2025-05-24"), "Sunday": pd.Timestamp("2025-05-18")}
CONFIRMATION_START = {"Saturday": pd.Timestamp("2025-05-31"), "Sunday": pd.Timestamp("2025-05-25")}
MATCHED_END = pd.Timestamp("2026-06-21")
EXTENSION_END = pd.Timestamp("2026-07-12")
POINT_SEED = 42
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_REPLICATIONS = 500
REPRODUCTION_ATOL = 1e-10

PROTECTED_PATHS = [
    ROOT / "app.py",
    ROOT / "src/predictor.py",
    ROOT / "src/features.py",
    ROOT / "src/modeling.py",
    ROOT / "models/visitor_model_ny_12550.joblib",
    ROOT / "models/visitor_model.joblib",
]
PRIOR_PHASE_DIRS = [PHASE1_DIR, PHASE2A_DIR, PHASE2A5_DIR, PHASE2B1_DIR]

PREDICTION_KEY = [
    "data_snapshot_id", "feature_set_id", "sample_weight_id", "forecast_origin",
    "target_date", "scenario", "weather_policy", "weekday_policy",
]
ALIGNMENT_KEY = [
    "data_snapshot_id", "feature_set_id", "forecast_origin", "target_date",
    "scenario", "weather_policy", "weekday_policy",
]

REQUIRED_ARTIFACTS_ALWAYS = [
    "00_implementation_design.md", "01_phase2b2_lite_summary.md", "02_locked_contract.json",
    "03_candidate_registry.json", "04_uniform_reproduction.md", "05_point_predictions.csv",
    "06_development_decision.csv", "07_locked_policy.json", "08_confirmation_guardrail.csv",
    "09_full_metric_comparison.csv", "10_weight_diagnostics.csv", "13_test_report.md",
    "phase2b2_lite_manifest.json", "README.md",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def write_json(name: str, value: Any) -> None:
    (OUTPUT_DIR / name).write_text(
        json.dumps(json_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for column in view.columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6f}" if isinstance(value, float) else str(value)
        )
    lines = [
        "| " + " | ".join(view.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def protected_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in PROTECTED_PATHS
        if path.is_file()
    }


def prior_phase_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): directory_fingerprint(path)
        for path in PRIOR_PHASE_DIRS
    }


def validate_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(SNAPSHOT_PATH) != EXPECTED_SNAPSHOT_SHA256:
        raise AssertionError("Normalized snapshot hash differs from the locked value")
    if sha256_file(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise AssertionError("Source-export hash differs from the locked value")
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    dates = pd.to_datetime(snapshot["service_date"], format="%Y-%m-%d", errors="raise")
    if len(snapshot) != 360 or dates.min() != pd.Timestamp("2023-01-01") or dates.max() != EXTENSION_END:
        raise AssertionError("Normalized snapshot shape/date range changed")
    if snapshot["service_date"].duplicated().any() or snapshot["attendance"].isna().any():
        raise AssertionError("Normalized snapshot contains duplicate or missing records")
    nonweekends = dates[~dates.dt.weekday.isin([5, 6])]
    if nonweekends.tolist() != [pd.Timestamp("2026-04-14")]:
        raise AssertionError("T1 exclusion set changed")
    lock = locked_feature_contract()
    if lock["selected_feature_set_id"] != F6 or lock["feature_list_sha256"] != EXPECTED_F6_HASH:
        raise AssertionError("F6 identity/order hash changed")
    if len(lock["ordered_feature_list"]) != 33:
        raise AssertionError("F6 feature count changed")
    return snapshot, lock


def build_evaluator(history: pd.DataFrame, lock: dict[str, Any]) -> OriginAwareBacktester:
    features = list(lock["ordered_feature_list"])
    registry = build_feature_set_registry(
        f5_parent_id="F4_CORRECTED_SLOT_HISTORY",
        f6_features=features,
        f6_groups=[
            "calendar", "last_observed_daytype", "daytype_summaries",
            "daytype_slot", "horizon_availability",
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
        random_seed=POINT_SEED,
        training_window=TW_EXPANDING,
    )


def role_map_for_history(history: pd.DataFrame) -> dict[pd.Timestamp, str]:
    roles: dict[pd.Timestamp, str] = {}
    for date in pd.to_datetime(history["service_date"]):
        day = "Sunday" if date.weekday() == 6 else "Saturday"
        if date.weekday() not in {5, 6}:
            continue
        roles[date.normalize()] = "development" if date <= DEVELOPMENT_END[day] else "confirmation"
    return roles


def write_contract(snapshot: pd.DataFrame, lock: dict[str, Any]) -> dict[str, Any]:
    point_params = make_point_model().get_params()
    registry = candidate_registry_rows()
    write_json("03_candidate_registry.json", registry)
    contract = {
        "frozen_at_utc": now_iso(),
        "snapshot_path": str(SNAPSHOT_PATH.relative_to(ROOT)),
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "source_export_sha256": EXPECTED_SOURCE_SHA256,
        "snapshot_rows": int(len(snapshot)),
        "snapshot_date_range": [snapshot["service_date"].min(), snapshot["service_date"].max()],
        "feature_set_id": F6,
        "feature_order_sha256": lock["feature_list_sha256"],
        "ordered_features": lock["ordered_feature_list"],
        "training_window_id": "TW_EXPANDING",
        "weekday_policy": T1_VALID_WEEKENDS,
        "weather_policy": W0_NO_WEATHER,
        "minimum_segment_training_rows": 18,
        "point_model_class": type(make_point_model()).__name__,
        "point_model_parameters": point_params,
        "point_model_seed": POINT_SEED,
        "preprocessing": "fold-local unweighted SimpleImputer(median, keep_empty_features=True)",
        "development_end": {key: value.strftime("%Y-%m-%d") for key, value in DEVELOPMENT_END.items()},
        "confirmation_start": {key: value.strftime("%Y-%m-%d") for key, value in CONFIRMATION_START.items()},
        "candidate_registry": registry,
        "development_thresholds": {
            "macro_mae_minimum_improvement": 0.25,
            "recent_tail_minimum_improvement": 0.15,
            "s2_maximum_worsening": 0.25,
            "saturday_maximum_worsening": 0.40,
            "sunday_maximum_worsening": 0.40,
            "p90_maximum_worsening": 0.50,
        },
        "confirmation_guardrails": {
            "mae_maximum_worsening": 0.20,
            "recent_52_maximum_worsening": 0.20,
            "s2_maximum_worsening": 0.50,
            "saturday_maximum_worsening": 0.50,
            "sunday_maximum_worsening": 0.50,
            "p90_maximum_worsening": 1.00,
        },
        "bootstrap": {"conditional": True, "replications": BOOTSTRAP_REPLICATIONS, "seed": BOOTSTRAP_SEED},
        "starting_branch": git("branch", "--show-current"),
        "starting_commit": git("rev-parse", "HEAD"),
        "protected_fingerprints_before": protected_fingerprints(),
        "prior_phase_fingerprints_before": prior_phase_fingerprints(),
        "phase2b1_prediction_sha256": sha256_file(PHASE2B1_PREDICTIONS),
    }
    write_json("02_locked_contract.json", contract)
    return contract


def recent_target_dates(frame: pd.DataFrame, count: int = 52) -> set[pd.Timestamp]:
    return set(sorted(pd.to_datetime(frame["target_date"]).drop_duplicates())[-count:])


def development_tail(frame: pd.DataFrame) -> pd.DataFrame:
    development = frame[frame["period_role"] == "development"]
    dates: set[pd.Timestamp] = set()
    unique = development[["target_date", "day_type"]].drop_duplicates()
    for _, part in unique.groupby("day_type"):
        dates.update(part.sort_values("target_date").tail(26)["target_date"])
    return development[development["target_date"].isin(dates)]


def basic_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    error = frame["point_error"].astype(float)
    absolute = error.abs()
    return {
        "row_count": int(len(frame)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "p90_absolute_error": float(absolute.quantile(0.90)),
    }


def development_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, part in predictions.groupby("sample_weight_id", sort=False):
        development = part[part["period_role"] == "development"]
        scenario = development.groupby("scenario")["absolute_error"].mean()
        if len(scenario) != 5:
            raise AssertionError("Development macro metric requires all S1-S5 scenarios")
        metrics = basic_metrics(development)
        rows.append(
            {
                "sample_weight_id": policy,
                "development_macro_mae": float(scenario.mean()),
                "development_micro_mae": metrics["mae"],
                "development_recent_tail_mae": float(development_tail(part)["absolute_error"].mean()),
                "development_s2_mae": float(development.loc[development["scenario"] == "S2_same_weekend_sunday", "absolute_error"].mean()),
                "development_saturday_mae": float(development.loc[development["day_type"] == "Saturday", "absolute_error"].mean()),
                "development_sunday_mae": float(development.loc[development["day_type"] == "Sunday", "absolute_error"].mean()),
                "development_p90_absolute_error": metrics["p90_absolute_error"],
                "development_bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows)


def confirmation_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    recent52 = recent_target_dates(predictions)
    rows: list[dict[str, Any]] = []
    for policy, part in predictions.groupby("sample_weight_id", sort=False):
        confirmation = part[part["period_role"] == "confirmation"]
        rows.append(
            {
                "sample_weight_id": policy,
                "confirmation_mae": float(confirmation["absolute_error"].mean()),
                "confirmation_recent_52_mae": float(confirmation.loc[confirmation["target_date"].isin(recent52), "absolute_error"].mean()),
                "confirmation_s2_mae": float(confirmation.loc[confirmation["scenario"] == "S2_same_weekend_sunday", "absolute_error"].mean()),
                "confirmation_saturday_mae": float(confirmation.loc[confirmation["day_type"] == "Saturday", "absolute_error"].mean()),
                "confirmation_sunday_mae": float(confirmation.loc[confirmation["day_type"] == "Sunday", "absolute_error"].mean()),
                "confirmation_p90_absolute_error": float(confirmation["absolute_error"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def full_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    recent52 = recent_target_dates(predictions)
    uniform = predictions[predictions["sample_weight_id"] == SW_UNIFORM.sample_weight_id]
    uniform_pair = uniform[ALIGNMENT_KEY + ["absolute_error"]].rename(columns={"absolute_error": "uniform_absolute_error"})
    rows: list[dict[str, Any]] = []
    for policy, part in predictions.groupby("sample_weight_id", sort=False):
        merged = part.merge(uniform_pair, on=ALIGNMENT_KEY, validate="one_to_one")
        target_change = merged.assign(
            improvement=merged["uniform_absolute_error"] - merged["absolute_error"]
        ).groupby("target_date")["improvement"].mean()
        metrics = basic_metrics(part)
        rows.append(
            {
                "sample_weight_id": policy,
                "full_history_mae": metrics["mae"],
                "confirmation_mae": float(part.loc[part["period_role"] == "confirmation", "absolute_error"].mean()),
                "recent_52_mae": float(part.loc[part["target_date"].isin(recent52), "absolute_error"].mean()),
                "new_extension_mae": float(part.loc[(part["target_date"] > MATCHED_END) & (part["target_date"] <= EXTENSION_END), "absolute_error"].mean()),
                "saturday_mae": float(part.loc[part["day_type"] == "Saturday", "absolute_error"].mean()),
                "sunday_mae": float(part.loc[part["day_type"] == "Sunday", "absolute_error"].mean()),
                "s2_mae": float(part.loc[part["scenario"] == "S2_same_weekend_sunday", "absolute_error"].mean()),
                "h1_mae": float(part.loc[part["service_horizon"] == 1, "absolute_error"].mean()),
                "h2_mae": float(part.loc[part["service_horizon"] == 2, "absolute_error"].mean()),
                "h5_mae": float(part.loc[part["service_horizon"] == 5, "absolute_error"].mean()),
                "bias": metrics["bias"],
                "p90_absolute_error": metrics["p90_absolute_error"],
                "prediction_rows_improved_pct": float((merged["absolute_error"] < merged["uniform_absolute_error"]).mean() * 100),
                "target_dates_improved_pct": float((target_change > 0).mean() * 100),
                "largest_target_date_improvement": float(target_change.max()),
                "largest_target_date_improvement_date": target_change.idxmax(),
                "largest_target_date_deterioration": float(target_change.min()),
                "largest_target_date_deterioration_date": target_change.idxmin(),
                "median_effective_sample_size": float(part["effective_sample_size"].median()),
                "minimum_effective_sample_size": float(part["effective_sample_size"].min()),
            }
        )
    return pd.DataFrame(rows)


def check_uniform_reproduction(uniform: pd.DataFrame) -> dict[str, Any]:
    reference = pd.read_csv(
        PHASE2B1_PREDICTIONS,
        parse_dates=["forecast_origin", "target_date"],
        low_memory=False,
    )
    reference = reference[reference["training_window_id"] == "TW_EXPANDING"].copy()
    reference = reference.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    actual = uniform.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    if len(reference) != len(actual) or not reference[ALIGNMENT_KEY].equals(actual[ALIGNMENT_KEY]):
        raise AssertionError("SW_UNIFORM keys do not align with Phase 2B1 TW_EXPANDING")
    difference = np.abs(reference["point_prediction"].to_numpy() - actual["point_prediction"].to_numpy())
    payload = {
        "status": "passed",
        "aligned_rows": int(len(actual)),
        "exact_key_alignment": True,
        "maximum_point_prediction_difference": float(difference.max(initial=0.0)),
        "absolute_tolerance": REPRODUCTION_ATOL,
    }
    if payload["maximum_point_prediction_difference"] > REPRODUCTION_ATOL:
        raise AssertionError(f"SW_UNIFORM reproduction failed: {payload}")
    return payload


def bootstrap_rows(predictions: pd.DataFrame, qualified_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    uniform = predictions[predictions["sample_weight_id"] == SW_UNIFORM.sample_weight_id][ALIGNMENT_KEY + ["absolute_error"]].rename(columns={"absolute_error": "uniform_absolute_error"})
    rows: list[dict[str, Any]] = []
    recent52 = recent_target_dates(predictions)
    for policy in qualified_ids:
        paired = predictions[predictions["sample_weight_id"] == policy].merge(uniform, on=ALIGNMENT_KEY, validate="one_to_one")
        scopes = {
            "development": paired[paired["period_role"] == "development"],
            "full_history": paired,
            "recent_52": paired[paired["target_date"].isin(recent52)],
        }
        for scope, part in scopes.items():
            clusters = part.assign(diff=part["absolute_error"] - part["uniform_absolute_error"]).groupby("target_date")["diff"].mean().to_numpy()
            draws = rng.choice(clusters, size=(BOOTSTRAP_REPLICATIONS, len(clusters)), replace=True).mean(axis=1)
            rows.append(
                {
                    "sample_weight_id": policy, "evaluation_scope": scope,
                    "target_date_cluster_count": int(len(clusters)),
                    "replications": BOOTSTRAP_REPLICATIONS, "seed": BOOTSTRAP_SEED,
                    "observed_mae_difference": float(clusters.mean()),
                    "ci_2_5_pct": float(np.quantile(draws, 0.025)),
                    "ci_97_5_pct": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_improvement": float((draws < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def quantile_diagnostic(cache, selected_policy: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policies = [SW_UNIFORM.sample_weight_id, selected_policy]
    for policy in policies:
        models = {}
        for key, fold in cache.training_folds.items():
            model = make_quantile_model(quantile=0.8)
            weights = generate_sample_weights(policy, len(fold.y_train))
            model.fit(fold.x_train, fold.y_train, sample_weight=weights)
            models[key] = model
        predictions = []
        for fold in cache.prediction_folds:
            actual = float(fold.metadata["actual"])
            prediction = float(models[fold.training_cache_key].predict(fold.x_test)[0])
            predictions.append({**fold.metadata, "quantile_prediction": prediction, "covers": actual <= prediction})
        frame = pd.DataFrame(predictions)
        for scope, part in {
            "full_history": frame,
            "development": frame[frame["period_role"] == "development"],
            "confirmation": frame[frame["period_role"] == "confirmation"],
        }.items():
            rows.append(
                {
                    "sample_weight_id": policy, "evaluation_scope": scope,
                    "row_count": int(len(part)), "nominal_quantile": 0.8,
                    "raw_coverage": float(part["covers"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_DIR / "00_implementation_design.md").is_file():
        raise FileNotFoundError("Implementation design must exist before scoring")
    # Refuse stale optional artifacts from a prior attempt.
    for optional in ["11_bootstrap_stability.csv", "12_quantile_diagnostic.csv"]:
        if (OUTPUT_DIR / optional).exists():
            raise FileExistsError(f"Remove stale optional artifact before rerun: {optional}")
    snapshot, lock = validate_inputs()
    contract = write_contract(snapshot, lock)
    history = history_from_normalized(snapshot)
    evaluator = build_evaluator(history, lock)

    cache_started = time.perf_counter()
    cache = evaluator.prepare_point_fold_cache(
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
        period_role_by_target=role_map_for_history(history),
    )
    cache_seconds = time.perf_counter() - cache_started
    print(
        f"Cached {len(cache.training_folds)} training contexts and "
        f"{len(cache.prediction_folds)} prediction keys in {cache_seconds:.2f}s.",
        flush=True,
    )

    prediction_parts: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    point_fit_count = 0
    candidate_runtime: dict[str, float] = {}
    training_hashes = {
        key: array_sha256(fold.x_train) for key, fold in cache.training_folds.items()
    }
    for policy in SAMPLE_WEIGHT_CANDIDATES:
        policy_started = time.perf_counter()
        print(f"Fitting {policy.sample_weight_id}...", flush=True)
        models = {}
        ess_by_key: dict[Any, float] = {}
        for key, fold in cache.training_folds.items():
            weights = generate_sample_weights(policy, len(fold.y_train))
            model = make_point_model()
            model.fit(fold.x_train, fold.y_train, sample_weight=weights)
            models[key] = model
            point_fit_count += 1
            ess = effective_sample_size(weights)
            ess_by_key[key] = ess
            weight_rows.append(
                {
                    "sample_weight_id": policy.sample_weight_id,
                    "half_life_rows": policy.half_life_rows,
                    "fold_cache_id": fold.preprocessing_id,
                    "segment": fold.segment,
                    "training_row_count": int(len(fold.y_train)),
                    "oldest_age": int(len(fold.y_train) - 1),
                    "newest_age": 0,
                    "minimum_weight": float(weights.min()),
                    "maximum_weight": float(weights.max()),
                    "mean_weight": float(weights.mean()),
                    "sum_weights": float(weights.sum()),
                    "effective_sample_size": ess,
                    "training_start_date": fold.training_start_date,
                    "training_end_date": fold.training_end_date,
                    "training_row_indices_sha256": hashlib.sha256(json.dumps(fold.training_row_indices).encode()).hexdigest(),
                    "x_train_sha256": training_hashes[key],
                    "imputer_statistics_sha256": array_sha256(fold.imputer_statistics),
                    "preprocessing_weighted": False,
                }
            )
        rows: list[dict[str, Any]] = []
        for fold in cache.prediction_folds:
            point = apply_calibration_placeholder(
                float(models[fold.training_cache_key].predict(fold.x_test)[0])
            )
            actual = float(fold.metadata["actual"])
            error = point - actual
            rows.append(
                {
                    "data_snapshot_id": DATA_SNAPSHOT_ID,
                    "feature_set_hash": EXPECTED_F6_HASH,
                    "training_window_id": "TW_EXPANDING",
                    "sample_weight_id": policy.sample_weight_id,
                    "half_life_rows": policy.half_life_rows,
                    **fold.metadata,
                    "point_prediction": point,
                    "point_error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "effective_sample_size": ess_by_key[fold.training_cache_key],
                    "fold_cache_id": fold.metadata["preprocessing_id"],
                    "x_train_sha256": training_hashes[fold.training_cache_key],
                    "x_test_sha256": array_sha256(fold.x_test),
                    "point_model_name": type(models[fold.training_cache_key]).__name__,
                    "point_random_seed": POINT_SEED,
                }
            )
        frame = pd.DataFrame(rows)
        prediction_parts.append(frame)
        candidate_runtime[policy.sample_weight_id] = time.perf_counter() - policy_started
        print(
            f"Completed {policy.sample_weight_id}: {len(frame)} predictions, "
            f"{len(models)} point fits.",
            flush=True,
        )
        if policy.sample_weight_id == SW_UNIFORM.sample_weight_id:
            reproduction = check_uniform_reproduction(frame)
            write_text(
                "04_uniform_reproduction.md",
                "# SW_UNIFORM reproduction\n\nStatus: **passed**. Exact Phase 2B1 TW_EXPANDING keys align and point predictions are equal within the locked tolerance.\n\n"
                + markdown_table(pd.DataFrame([reproduction])),
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["forecast_origin"] = pd.to_datetime(predictions["forecast_origin"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["training_end_date"] = pd.to_datetime(predictions["training_end_date"])
    predictions["training_start_date"] = pd.to_datetime(predictions["training_start_date"])
    if predictions.duplicated(PREDICTION_KEY).any():
        raise AssertionError("Duplicate point-prediction keys")
    reference_keys = None
    for policy, part in predictions.groupby("sample_weight_id"):
        keys = part[ALIGNMENT_KEY].sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise AssertionError(f"Prediction keys do not align for {policy}")
    if not predictions["feature_provenance_valid"].astype(bool).all():
        raise AssertionError("Feature provenance validation failed")
    if not (predictions["training_end_date"] <= predictions["forecast_origin"]).all():
        raise AssertionError("Cached training cutoff exceeds forecast origin")
    if predictions.groupby(ALIGNMENT_KEY)["fold_cache_id"].nunique().max() != 1:
        raise AssertionError("Candidates did not reuse identical cached fold IDs")
    if predictions.groupby(ALIGNMENT_KEY)["x_train_sha256"].nunique().max() != 1:
        raise AssertionError("Candidate training matrices differ")
    if predictions.groupby(ALIGNMENT_KEY)["x_test_sha256"].nunique().max() != 1:
        raise AssertionError("Candidate test rows differ")
    write_csv("05_point_predictions.csv", predictions.sort_values(PREDICTION_KEY, kind="stable"))
    write_csv("10_weight_diagnostics.csv", pd.DataFrame(weight_rows))

    # Development selection path intentionally sees development columns only.
    development = development_metric_table(predictions)
    development_selected, decision = select_development_policy(development)
    write_csv("06_development_decision.csv", decision)
    development_lock_time = now_iso()
    policy_lock = {
        "development_locked_at_utc": development_lock_time,
        "development_locked_policy": development_selected,
        "confirmation_metrics_read_or_calculated_before_development_lock": False,
        "development_qualification_records": decision.to_dict(orient="records"),
        "final_policy": None,
        "confirmation_guardrail_applied_at_utc": None,
    }
    write_json("07_locked_policy.json", policy_lock)

    # Confirmation is a reused diagnostic set and is aggregated only post-lock.
    confirmation = confirmation_metric_table(predictions)
    final_policy, guardrail = apply_confirmation_guardrail(
        development_selected, confirmation
    )
    write_csv("08_confirmation_guardrail.csv", guardrail)
    policy_lock.update(
        {
            "confirmation_guardrail_applied_at_utc": now_iso(),
            "confirmation_is_reused_diagnostic_not_pristine_holdout": True,
            "final_policy": final_policy,
            "weighted_policy_survived_development_and_confirmation": final_policy != SW_UNIFORM.sample_weight_id,
            "no_additional_half_life_searched": True,
        }
    )
    write_json("07_locked_policy.json", policy_lock)

    full_metrics = full_metric_table(predictions)
    write_csv("09_full_metric_comparison.csv", full_metrics)
    qualified_ids = decision.loc[decision["development_qualified"], "sample_weight_id"].tolist()
    bootstrap_ran = bool(qualified_ids)
    if bootstrap_ran:
        write_csv("11_bootstrap_stability.csv", bootstrap_rows(predictions, qualified_ids))
    quantile_ran = final_policy != SW_UNIFORM.sample_weight_id
    if quantile_ran:
        write_csv("12_quantile_diagnostic.csv", quantile_diagnostic(cache, final_policy))

    runtime_seconds = time.perf_counter() - started
    selected_full = full_metrics[full_metrics["sample_weight_id"] == final_policy].iloc[0]
    uniform_full = full_metrics[full_metrics["sample_weight_id"] == SW_UNIFORM.sample_weight_id].iloc[0]
    qualified_text = ", ".join(qualified_ids) if qualified_ids else "none"
    if final_policy == SW_UNIFORM.sample_weight_id:
        conclusion = (
            "Recency weighting did not produce a sufficiently large and stable gain. "
            "Point-model optimization is complete. The next technical phase should "
            "address quantile and meal-recommendation calibration. Future point-model "
            "changes require prospective evidence or materially new data."
        )
    else:
        conclusion = (
            f"{final_policy} passed the development and reused-confirmation gates. "
            "Freeze it for prospective evaluation before any production decision."
        )
    phase2b1_quantile = pd.read_csv(PHASE2B1_DIR / "17_quantile_coverage_analysis.csv")
    reused_coverage = float(
        phase2b1_quantile.loc[
            (phase2b1_quantile["training_window_id"] == "TW_EXPANDING")
            & (phase2b1_quantile["evaluation_scope"] == "full_history"),
            "raw_quantile_coverage",
        ].iloc[0]
    )
    write_text(
        "01_phase2b2_lite_summary.md",
        f"# Phase 2B2-Lite summary\n\nFinal policy: **{final_policy}**. Development-qualified weighted candidates: **{qualified_text}**. {conclusion}\n\n"
        + markdown_table(full_metrics)
        + f"\n\nSW_UNIFORM reuses the Phase 2B1 uncalibrated quantile result (full-history raw coverage {reused_coverage:.6f}); no candidate quantile screening was run. Runtime was {runtime_seconds:.2f} seconds with {point_fit_count} point-model fits. The origin-aware training frame was built once, {len(cache.training_folds)} training contexts were cached, and all {len(cache.prediction_folds)} prediction feature rows were reused across candidates.\n\nThe confirmation period and six-date extension were examined previously and are reused diagnostics, not pristine holdouts. Prospectively compare the frozen final policy with SW_UNIFORM over at least 8-12 new service dates after 2026-07-12. Do not reopen the half-life search with those records.",
    )
    write_text(
        "13_test_report.md",
        "# Test and reproducibility report\n\nTargeted tests: **pending final run**. Full suite: **pending final run**.\n\nThe runner verified snapshot/export/F6 hashes, exact SW_UNIFORM reproduction, aligned candidate keys, shared cached fold IDs and matrices, origin-safe feature provenance, unweighted preprocessing, protected files, and prior-phase fingerprints. Test results will be recorded after the requested commands execute against the final code state.",
    )
    write_text(
        "README.md",
        "# Phase 2B2-Lite sample-weight artifacts\n\nCompact preregistered falsification-test outputs. `05_point_predictions.csv` contains the three aligned point candidates; `06_development_decision.csv` contains qualification evidence; `07_locked_policy.json` records the pre-confirmation development lock and final guardrail result; `09_full_metric_comparison.csv` is the compact results table. Optional bootstrap and quantile CSVs exist only if their preregistered gates ran.\n",
    )

    integrity = {
        "snapshot_hash_unchanged": sha256_file(SNAPSHOT_PATH) == EXPECTED_SNAPSHOT_SHA256,
        "source_hash_unchanged": sha256_file(SOURCE_PATH) == EXPECTED_SOURCE_SHA256,
        "f6_hash_unchanged": locked_feature_contract()["feature_list_sha256"] == EXPECTED_F6_HASH,
        "production_and_saved_models_unchanged": protected_fingerprints() == contract["protected_fingerprints_before"],
        "prior_phase_artifacts_unchanged": prior_phase_fingerprints() == contract["prior_phase_fingerprints_before"],
        "prediction_keys_aligned": True,
        "feature_provenance_valid": True,
        "preprocessing_unweighted": bool(not predictions["preprocessing_weighted"].astype(bool).any()),
        "cached_folds_reused": True,
    }
    if not all(integrity.values()):
        raise AssertionError(f"Final integrity validation failed: {integrity}")
    artifacts = list(REQUIRED_ARTIFACTS_ALWAYS)
    if bootstrap_ran:
        artifacts.append("11_bootstrap_stability.csv")
    if quantile_ran:
        artifacts.append("12_quantile_diagnostic.csv")
    manifest = {
        "timestamp_utc": now_iso(),
        "git_branch": contract["starting_branch"],
        "starting_commit": contract["starting_commit"],
        "ending_commit": git("rev-parse", "HEAD"),
        "python_version": platform.python_version(),
        "dependency_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "locked_contract": contract,
        "development_locked_policy": development_selected,
        "final_policy": final_policy,
        "bootstrap_ran": bootstrap_ran,
        "bootstrap_skip_reason": None if bootstrap_ran else "No weighted candidate passed every development qualification rule.",
        "quantile_models_ran": quantile_ran,
        "quantile_result_reused_from_phase2b1": not quantile_ran,
        "reused_uniform_full_history_raw_quantile_coverage": reused_coverage if not quantile_ran else None,
        "runtime_seconds": runtime_seconds,
        "fold_cache_build_seconds": cache_seconds,
        "training_frame_build_count": cache.training_frame_build_count,
        "unique_training_context_count": len(cache.training_folds),
        "prediction_key_count_per_candidate": len(cache.prediction_folds),
        "prediction_feature_build_count": cache.prediction_feature_build_count,
        "point_model_fit_count": point_fit_count,
        "point_fit_count_by_candidate": {item.sample_weight_id: len(cache.training_folds) for item in SAMPLE_WEIGHT_CANDIDATES},
        "candidate_runtime_seconds": candidate_runtime,
        "cached_fold_reuse": {
            "training_context_arrays_shared_across_candidates": True,
            "test_feature_arrays_shared_across_candidates": True,
            "fold_local_imputer_fit_once_per_training_context": True,
            "preprocessing_weighted": False,
        },
        "uniform_reproduction": reproduction,
        "integrity_checks": integrity,
        "test_results": {"targeted": "pending", "full": "pending"},
        "files_created": artifacts,
        "code_files_created": ["src/sample_weights.py", "scripts/run_phase2b2_lite.py", "tests/test_sample_weights.py", "tests/test_phase2b2_lite.py"],
        "code_files_modified": ["src/origin_backtest.py"],
        "commands": ["python scripts/run_phase2b2_lite.py", "python -m pytest -q tests/test_sample_weights.py tests/test_phase2b2_lite.py", "python -m pytest -q"],
        "prospective_evaluation": "Compare frozen final policy versus SW_UNIFORM on at least 8-12 service dates after 2026-07-12 without reopening weight search.",
    }
    write_json("phase2b2_lite_manifest.json", manifest)
    missing = [name for name in artifacts if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise AssertionError(f"Missing required artifacts: {missing}")
    print(
        f"Phase 2B2-Lite complete: {final_policy}; {point_fit_count} point fits; "
        f"{runtime_seconds:.2f}s total.",
        flush=True,
    )


if __name__ == "__main__":
    run()
