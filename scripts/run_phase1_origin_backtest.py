#!/usr/bin/env python3
"""Run Phase 1 without changing production code, packages, or legacy artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from src.origin_backtest import (
    BASELINE_COLUMNS,
    SCENARIO_DEFINITIONS,
    OriginAwareBacktester,
    calculate_metrics,
    metric_breakdowns,
)
from src.origin_features import (
    ATTENDANCE_FEATURES,
    CALENDAR_FEATURES,
    MODEL_FEATURES,
    T0_LEGACY_ALL,
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
    W1_OBSERVED_REPLAY,
    W2_ARCHIVED_FORECAST,
    WEATHER_FEATURES,
)


OUTPUT_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
MODEL_PATH = ROOT / "models/visitor_model_ny_12550.joblib"
SQLITE_PATH = ROOT / "data/locations/ny_12550/attendance.db"
CSV_PATH = ROOT / "data/visitors_clean.csv"
WEATHER_PATH = ROOT / "data/locations/ny_12550/weather_daily.csv"
LEGACY_PREDICTIONS_PATH = ROOT / "artifacts/ny_12550/backtest_predictions.csv"
AUDIT_DIR = ROOT / "artifacts/ny_12550/model_audit"

EXPECTED_ARTIFACTS = [
    "00_implementation_design.md",
    "01_phase1_summary.md",
    "02_data_source_reconciliation.csv",
    "02_data_source_reconciliation.md",
    "03_origin_and_horizon_definition.md",
    "04_feature_status_matrix.csv",
    "04_feature_status_matrix.md",
    "05_origin_aware_predictions.csv",
    "06_feature_provenance_samples.csv",
    "07_fold_preprocessing_diagnostics.csv",
    "08_weather_policy_comparison.csv",
    "08_weather_policy_comparison.md",
    "09_tuesday_record_analysis.md",
    "10_simple_baseline_comparison.csv",
    "10_simple_baseline_comparison.md",
    "11_legacy_vs_origin_comparison.csv",
    "11_legacy_vs_origin_comparison.md",
    "12_error_diagnostics.csv",
    "12_error_diagnostics.md",
    "13_test_and_reproducibility_report.md",
    "14_phase2_recommendation.md",
    "phase1_manifest.json",
    "README.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_attendance_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame[[DATE_COL, TARGET_COL]].copy()
    canonical[DATE_COL] = pd.to_datetime(canonical[DATE_COL]).dt.strftime("%Y-%m-%d")
    canonical[TARGET_COL] = pd.to_numeric(canonical[TARGET_COL])
    canonical = canonical.sort_values(DATE_COL, kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(command: list[str]) -> str:
    return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()


def markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    display = frame.head(max_rows) if max_rows else frame
    if display.empty:
        return "_No rows._"
    normalized = display.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: ""
            if pd.isna(value)
            else (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = [str(column).replace("|", "\\|") for column in normalized.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in normalized.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_text(name: str, text: str) -> None:
    (OUTPUT_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(name: str, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_DIR / name, index=False, lineterminator="\n")


def load_sources(package: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    package_history = package["history_df"][[DATE_COL, TARGET_COL]].copy()
    package_history[DATE_COL] = pd.to_datetime(package_history[DATE_COL]).dt.normalize()
    with sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True) as connection:
        sqlite_history = pd.read_sql_query(
            "SELECT service_date, visitors FROM attendance ORDER BY service_date", connection
        )
    sqlite_history[DATE_COL] = pd.to_datetime(sqlite_history[DATE_COL]).dt.normalize()
    csv_history = pd.read_csv(CSV_PATH, usecols=[DATE_COL, TARGET_COL], parse_dates=[DATE_COL])
    csv_history[DATE_COL] = pd.to_datetime(csv_history[DATE_COL]).dt.normalize()
    return package_history, sqlite_history, csv_history


def source_summary(
    source: str,
    frame: pd.DataFrame,
    *,
    path: Path,
    used_by_training: str,
    used_by_prediction: str,
    used_by_phase1: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "path": str(path.relative_to(ROOT)),
        "row_count": int(len(frame)),
        "earliest_service_date": pd.to_datetime(frame[DATE_COL]).min(),
        "latest_service_date": pd.to_datetime(frame[DATE_COL]).max(),
        "duplicate_dates": int(pd.to_datetime(frame[DATE_COL]).duplicated().sum()),
        "missing_target_values": int(frame[TARGET_COL].isna().sum()),
        "stable_attendance_sha256": stable_attendance_fingerprint(frame),
        "file_sha256": sha256_file(path),
        "used_by_current_training": used_by_training,
        "used_by_current_prediction": used_by_prediction,
        "used_by_phase1": used_by_phase1,
        "authority_reason": reason,
    }


def reconciliation_artifacts(
    package_history: pd.DataFrame,
    sqlite_history: pd.DataFrame,
    csv_history: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            source_summary(
                "saved_model_package_history",
                package_history,
                path=MODEL_PATH,
                used_by_training="Saved output of the last location training run",
                used_by_prediction="Yes; VisitorPredictor uses package history_df",
                used_by_phase1="Yes",
                reason="Only snapshot that exactly aligns with the committed 317-row Phase 0 legacy backtest and reaches 2026-06-21",
            ),
            source_summary(
                "local_sqlite_current_training_input",
                sqlite_history,
                path=SQLITE_PATH,
                used_by_training="Yes in local fallback mode via load_clean_data",
                used_by_prediction="No; prediction reads frozen package history",
                used_by_phase1="No",
                reason="Ends 2026-02-15 and therefore cannot reproduce the committed location-model evaluation snapshot",
            ),
            source_summary(
                "legacy_csv_bootstrap_snapshot",
                csv_history,
                path=CSV_PATH,
                used_by_training="Only bootstraps an empty local store",
                used_by_prediction="No",
                used_by_phase1="No",
                reason="Ends 2026-02-15 and omits 2023-01-01; not merged with other sources",
            ),
        ]
    )
    write_csv("02_data_source_reconciliation.csv", summary)

    def differences(other: pd.DataFrame, label: str) -> pd.DataFrame:
        left = package_history.rename(columns={TARGET_COL: "package_visitors"})
        right = other.rename(columns={TARGET_COL: f"{label}_visitors"})
        merged = left.merge(right, on=DATE_COL, how="outer", indicator=True)
        other_col = f"{label}_visitors"
        diff = merged[(merged["_merge"] != "both") | (merged["package_visitors"] != merged[other_col])].copy()
        diff["comparison"] = label
        diff["difference_type"] = np.select(
            [diff["_merge"] == "left_only", diff["_merge"] == "right_only"],
            ["package_only", f"{label}_only"],
            default="attendance_value_mismatch",
        )
        return diff[["comparison", DATE_COL, "package_visitors", other_col, "difference_type"]].rename(
            columns={other_col: "comparison_visitors"}
        )

    diff = pd.concat(
        [differences(sqlite_history, "sqlite"), differences(csv_history, "csv")],
        ignore_index=True,
    ).sort_values(["comparison", DATE_COL])
    write_text(
        "02_data_source_reconciliation.md",
        f"""# Data-source reconciliation

The saved model package history is authoritative for Phase 1 because it is the
only snapshot that aligns with the committed location backtest and its 2026-06-21
endpoint. SQLite remains the current local training source and the CSV remains a
bootstrap snapshot; neither is merged into the evaluation.

## Source summary

{markdown_table(summary.drop(columns=['path', 'file_sha256']))}

## Every record-level difference versus the Phase 1 authority

{markdown_table(diff)}

There are {len(diff)} comparison rows with a coverage or value difference. A
date can appear twice because SQLite and CSV are independent comparisons. This
table intentionally reports differences rather than resolving them.
""",
    )
    return summary


def add_periods(frame: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    valid = authority[pd.to_datetime(authority[DATE_COL]).dt.weekday.isin([5, 6])].sort_values(DATE_COL)
    recent_dates = set(pd.to_datetime(valid.tail(52)[DATE_COL]).dt.normalize())
    output["period"] = np.where(pd.to_datetime(output["target_date"]).isin(recent_dates), "Recent 52", "Earlier")
    return output


def feature_status_artifacts(provenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        subset = provenance[provenance["feature"] == feature]
        if feature in CALENDAR_FEATURES:
            status = "Origin-valid without change"
            reason = "Deterministic from target date"
        elif feature in WEATHER_FEATURES:
            status = "Weather-policy dependent"
            reason = "W0 missing; W1 realized-weather replay; W2 unavailable"
        elif "slot" in feature:
            status = "Semantically inconsistent"
            reason = "Origin-reconstructed but current slot_num grouping can mix day types"
        else:
            status = "Origin-valid with reconstruction"
            reason = "Target-relative source is missing when it lies after the origin"
        row = {
            "feature": feature,
            "phase1_status": status,
            "reason": reason,
            "fold_missing_value_treatment": "Preserve NaN, then fold-local median imputation",
            "overall_raw_missing_rate": float(subset["imputed"].mean()),
        }
        for scenario in SCENARIO_DEFINITIONS:
            scenario_values = subset[subset["scenario"] == scenario]
            row[f"missing_rate_{scenario.split('_')[0]}"] = (
                float(scenario_values["imputed"].mean()) if not scenario_values.empty else np.nan
            )
        rows.append(row)
    matrix = pd.DataFrame(rows)
    write_csv("04_feature_status_matrix.csv", matrix)
    write_text(
        "04_feature_status_matrix.md",
        f"""# Phase 1 feature-status matrix

The unchanged 26-feature contract is retained. Raw missingness is intentional and
is resolved only by the training-fold median. Same-slot features retain the
current cross-day-type grouping and are labeled inconsistent rather than repaired.

{markdown_table(matrix)}
""",
    )
    return matrix


def weather_comparison_artifacts(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (weekday_policy, weather_policy), policy in predictions.groupby(
        ["weekday_policy", "weather_policy"], sort=True
    ):
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "All", policy)]
        groups.extend(("day_type", str(value), part) for value, part in policy.groupby("day_type"))
        groups.extend(
            ("service_horizon", str(value), part) for value, part in policy.groupby("service_horizon")
        )
        for breakdown, value, part in groups:
            metrics = calculate_metrics(part)
            rows.append(
                {
                    "weekday_policy": weekday_policy,
                    "weather_policy": weather_policy,
                    "breakdown": breakdown,
                    "breakdown_value": value,
                    "missing_weather_rate": float(part["weather_missing_rate"].mean()),
                    "imputed_weather_row_rate": float(part["weather_imputed"].mean()),
                    **metrics,
                }
            )
    comparison = pd.DataFrame(rows)
    write_csv("08_weather_policy_comparison.csv", comparison)
    overall = comparison[comparison["breakdown"] == "overall"]
    write_text(
        "08_weather_policy_comparison.md",
        f"""# Weather-policy comparison

W0 disables weather and imputes all five fields within each fold. W1 replays
realized target-date weather when cached and imputes cache gaps. W1 is a hindsight
diagnostic, not an origin-valid forecast. W2 is unavailable because no trustworthy
issue-time/archived forecasts exist; no realized-weather proxy was substituted.

## Overall

{markdown_table(overall)}

## Saturday, Sunday, and service horizons

{markdown_table(comparison[comparison['breakdown'] != 'overall'])}
""",
    )
    return comparison


def baseline_comparison_artifacts(predictions: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    base = add_periods(predictions[predictions["weather_policy"] == W0_NO_WEATHER], authority)
    rows: list[dict[str, Any]] = []
    for weekday_policy, policy in base.groupby("weekday_policy", sort=True):
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "All", policy)]
        for dimension in ["day_type", "scenario", "service_horizon", "period"]:
            groups.extend((dimension, str(value), part) for value, part in policy.groupby(dimension))
        for model in BASELINE_COLUMNS:
            for breakdown, value, part in groups:
                rows.append(
                    {
                        "model": model,
                        "weekday_policy": weekday_policy,
                        "weather_policy": "origin-attendance-only",
                        "breakdown": breakdown,
                        "breakdown_value": value,
                        **calculate_metrics(part, prediction_col=model, quantile_col=None),
                    }
                )
    comparison = pd.DataFrame(rows)
    write_csv("10_simple_baseline_comparison.csv", comparison)
    overall = comparison[comparison["breakdown"] == "overall"].sort_values("mae")
    write_text(
        "10_simple_baseline_comparison.md",
        f"""# Simple origin-constrained baselines

All baseline windows are fixed in advance and use attendance dated on or before
the row's forecast origin. W0 rows are used once because the baseline values do
not depend on weather.

## Overall ranking

{markdown_table(overall)}

## Required breakdowns

{markdown_table(comparison[comparison['breakdown'] != 'overall'])}
""",
    )
    return comparison


def legacy_comparison_artifacts(
    predictions: pd.DataFrame,
    legacy_predictions: pd.DataFrame,
    authority: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    evaluated = add_periods(predictions, authority)
    rows: list[dict[str, Any]] = []
    for (weekday_policy, weather_policy), policy in evaluated.groupby(
        ["weekday_policy", "weather_policy"], sort=True
    ):
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "All", policy)]
        for dimension in ["day_type", "scenario", "service_horizon", "period"]:
            groups.extend((dimension, str(value), part) for value, part in policy.groupby(dimension))
        for breakdown, value, part in groups:
            origin_metrics = calculate_metrics(part)
            aligned = part.dropna(subset=["legacy_point_prediction"])
            legacy_metrics = calculate_metrics(
                aligned,
                prediction_col="legacy_point_prediction",
                quantile_col="legacy_quantile_prediction",
            )
            rows.append(
                {
                    "model": "origin_aware_current_model",
                    "weekday_policy": weekday_policy,
                    "weather_policy": weather_policy,
                    "breakdown": breakdown,
                    "breakdown_value": value,
                    **origin_metrics,
                    "aligned_legacy_row_count": legacy_metrics.get("row_count", 0),
                    "aligned_legacy_mae": legacy_metrics.get("mae", np.nan),
                    "aligned_legacy_rmse": legacy_metrics.get("rmse", np.nan),
                    "aligned_legacy_mean_signed_error": legacy_metrics.get("mean_signed_error", np.nan),
                    "aligned_legacy_quantile_coverage": legacy_metrics.get("raw_quantile_coverage", np.nan),
                    "mae_delta_vs_aligned_legacy": origin_metrics.get("mae", np.nan) - legacy_metrics.get("mae", np.nan),
                    "rmse_delta_vs_aligned_legacy": origin_metrics.get("rmse", np.nan) - legacy_metrics.get("rmse", np.nan),
                    "bias_delta_vs_aligned_legacy": origin_metrics.get("mean_signed_error", np.nan)
                    - legacy_metrics.get("mean_signed_error", np.nan),
                    "quantile_coverage_delta_vs_aligned_legacy": origin_metrics.get(
                        "raw_quantile_coverage", np.nan
                    )
                    - legacy_metrics.get("raw_quantile_coverage", np.nan),
                }
            )

    legacy = legacy_predictions.rename(columns={"pred": "legacy_point", "pred_q": "legacy_quantile"})
    legacy["day_type"] = np.where(legacy["is_sun"] == 1, "Sunday", "Saturday")
    legacy_groups = [("overall", "All", legacy)]
    legacy_groups.extend(("day_type", value, part) for value, part in legacy.groupby("day_type"))
    for breakdown, value, part in legacy_groups:
        metrics = calculate_metrics(part, prediction_col="legacy_point", quantile_col="legacy_quantile")
        rows.append(
            {
                "model": "legacy_backtest_native",
                "weekday_policy": "legacy_current_behavior",
                "weather_policy": "legacy_observed_weather_full_frame_imputation",
                "breakdown": breakdown,
                "breakdown_value": value,
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows)
    # Append the fixed baseline table so this remains the requested main comparison.
    baseline_main = baselines.copy()
    for column in comparison.columns:
        if column not in baseline_main.columns:
            baseline_main[column] = np.nan
    for column in baseline_main.columns:
        if column not in comparison.columns:
            comparison[column] = np.nan
    comparison = pd.concat([comparison, baseline_main[comparison.columns]], ignore_index=True)
    write_csv("11_legacy_vs_origin_comparison.csv", comparison)
    overall = comparison[comparison["breakdown"] == "overall"]
    origin_overall = overall[overall["model"] == "origin_aware_current_model"]
    preferred = origin_overall[
        (origin_overall["weekday_policy"] == T1_VALID_WEEKENDS)
        & (origin_overall["weather_policy"] == W0_NO_WEATHER)
    ].iloc[0]
    classification = "optimistic" if preferred["mae_delta_vs_aligned_legacy"] > 0 else "pessimistic"
    write_text(
        "11_legacy_vs_origin_comparison.md",
        f"""# Legacy versus origin-aware comparison

The main table contains the legacy backtest, both origin-aware weather policies,
both weekday policies, and every fixed simple baseline. Deltas use only rows with
an aligned legacy target prediction. The preferred validity result is T1/W0.

## Overall comparison

{markdown_table(overall)}

For preferred T1/W0, origin-aware MAE changes by
{preferred['mae_delta_vs_aligned_legacy']:.3f} visitors versus the aligned legacy
rows and RMSE changes by {preferred['rmse_delta_vs_aligned_legacy']:.3f}. On that
aggregate comparison the legacy evaluation is **{classification}**. Scenario-level
rows in the CSV show whether that direction is mixed across operational horizons.
""",
    )
    return comparison


def tuesday_artifact(predictions: pd.DataFrame, authority: pd.DataFrame) -> dict[str, Any]:
    invalid = authority[~pd.to_datetime(authority[DATE_COL]).dt.weekday.isin([5, 6])].copy()
    if invalid.empty:
        write_text("09_tuesday_record_analysis.md", "# Tuesday record analysis\n\nNo invalid weekday records found.")
        return {"invalid_rows": 0}
    tuesday = invalid.iloc[0]
    date = pd.Timestamp(tuesday[DATE_COL])
    t0 = predictions[predictions["weekday_policy"] == T0_LEGACY_ALL].copy()
    t1 = predictions[predictions["weekday_policy"] == T1_VALID_WEEKENDS].copy()
    merge_keys = ["target_date", "scenario", "weather_policy"]
    aligned = t0[t0["actual_weekday"].isin(["Saturday", "Sunday"])].merge(
        t1,
        on=merge_keys,
        suffixes=("_t0", "_t1"),
    )
    policy_rows = []
    for weather_policy in [W0_NO_WEATHER, W1_OBSERVED_REPLAY]:
        a = aligned[aligned["weather_policy"] == weather_policy]
        t0_metrics = calculate_metrics(
            a.rename(columns={"actual_t0": "actual", "point_prediction_t0": "prediction"}),
            prediction_col="prediction",
            quantile_col=None,
        )
        t1_metrics = calculate_metrics(
            a.rename(columns={"actual_t1": "actual", "point_prediction_t1": "prediction"}),
            prediction_col="prediction",
            quantile_col=None,
        )
        policy_rows.append(
            {
                "weather_policy": weather_policy,
                "aligned_valid_rows": len(a),
                "t0_mae": t0_metrics.get("mae", np.nan),
                "t1_mae": t1_metrics.get("mae", np.nan),
                "t1_minus_t0_mae": t1_metrics.get("mae", np.nan) - t0_metrics.get("mae", np.nan),
                "max_absolute_prediction_change": float(
                    (a["point_prediction_t1"] - a["point_prediction_t0"]).abs().max()
                ),
            }
        )
    policy_frame = pd.DataFrame(policy_rows)
    tuesday_tests = t0[pd.to_datetime(t0["target_date"]) == date]
    affected_training_predictions = int((pd.to_datetime(t0["forecast_origin"]) >= date).sum())
    write_text(
        "09_tuesday_record_analysis.md",
        f"""# Tuesday observation analysis

The package contains `{date:%Y-%m-%d}` ({date.day_name()}) with attendance
{float(tuesday[TARGET_COL]):.0f}. The current `is_sun = 0` rule routes it to the
Saturday segment. T0 preserves it; T1 removes it from targets, training histories,
combined lags, same-day-type lags, and slot-group histories.

T0 evaluates the Tuesday only as S3. It produces {len(tuesday_tests)} test rows
(one per available weather policy). There are {affected_training_predictions}
T0 prediction rows whose origin is on/after the record and whose training or
origin-history sequence can therefore be affected. T1 has zero Tuesday tests.

## Aligned valid-weekend metric effect

{markdown_table(policy_frame)}

## T0 Tuesday predictions

{markdown_table(tuesday_tests[['target_date', 'forecast_origin', 'scenario', 'weather_policy', 'actual', 'point_prediction', 'absolute_error']] if not tuesday_tests.empty else tuesday_tests)}

T1 is the preferred validity policy because `validate_forecast_target_date`
accepts only Saturday and Sunday.
""",
    )
    return {
        "invalid_rows": int(len(invalid)),
        "date": date.strftime("%Y-%m-%d"),
        "attendance": float(tuesday[TARGET_COL]),
        "t0_test_rows": int(len(tuesday_tests)),
        "t1_test_rows": 0,
        "training_or_history_rows_affected": affected_training_predictions,
        "policy_comparison": policy_rows,
    }


def error_artifacts(predictions: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    diagnostics = metric_breakdowns(predictions)
    write_csv("12_error_diagnostics.csv", diagnostics)
    preferred = diagnostics[
        (diagnostics["weekday_policy"] == T1_VALID_WEEKENDS)
        & (diagnostics["weather_policy"] == W0_NO_WEATHER)
    ]
    write_text(
        "12_error_diagnostics.md",
        f"""# Error diagnostics

Signed error is prediction minus actual. Recent 52 and attendance-quartile
boundaries are fixed from the authoritative valid-weekend history before scoring.
The CSV contains all required policy, day-type, scenario, horizon, calendar-day,
year, quarter, recent-period, and attendance-quartile slices.

## Preferred T1/W0 headline slices

{markdown_table(preferred[preferred['breakdown'].isin(['overall', 'day_type', 'scenario', 'service_horizon', 'period'])])}

Raw model errors live in `05_origin_aware_predictions.csv`. Production-rule replay
columns in that file are descriptive only because saved residual buffers overlap
the legacy evaluation; they are not out-of-sample calibrated coverage.
""",
    )
    return diagnostics


def origin_definition_artifact() -> None:
    rows = [
        ["S1", "Saturday", "target - 1 day (Friday EOD)", 1, 1],
        ["S2", "Sunday", "target - 2 days (Friday EOD)", 2, 2],
        ["S3", "policy-valid record", "immediately previous policy-valid record EOD", "variable", 1],
        ["S4", "Saturday/Sunday", "target - 7 days EOD", 7, 2],
        ["S5", "Saturday/Sunday", "target - 15 days EOD", 15, 5],
    ]
    table = pd.DataFrame(rows, columns=["scenario", "target", "origin rule", "calendar days", "service horizon"])
    write_text(
        "03_origin_and_horizon_definition.md",
        f"""# Origin and horizon definitions

Forecast origins are end-of-day information cutoffs. The target attendance and
every attendance after the origin are unavailable. Service horizon counts valid
Saturday/Sunday services strictly after the origin through the target; T0's
Tuesday S3 is the next recorded service by construction.

{markdown_table(table)}

S5 follows `FORECAST_MAX_DAYS_AHEAD = 16`: the live validator accepts today
through today + 15. Invalid T1 weekdays are excluded. Invalid or insufficient
folds are counted, never remapped. S1 and Saturday S3 remain separate evaluation
units even when their features coincide.
""",
    )


def provenance_sample_artifact(provenance: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_date", "scenario", "weather_policy", "weekday_policy"]
    selected = (
        predictions.sort_values("target_date")
        .groupby(["scenario", "weather_policy", "weekday_policy", "day_type"], dropna=False)
        .head(1)[keys]
    )
    selected = pd.concat(
        [
            selected,
            predictions.sort_values("target_date")
            .groupby(["scenario", "weather_policy", "weekday_policy", "day_type"], dropna=False)
            .tail(1)[keys],
            predictions[predictions["actual_weekday"] == "Tuesday"][keys],
        ],
        ignore_index=True,
    ).drop_duplicates()
    sample = provenance.merge(selected, on=keys, how="inner")
    write_csv("06_feature_provenance_samples.csv", sample)
    return sample


def validate_in_memory(
    predictions: pd.DataFrame,
    provenance: pd.DataFrame,
    preprocessing: pd.DataFrame,
) -> dict[str, Any]:
    key = ["target_date", "scenario", "weather_policy", "weekday_policy"]
    duplicate_count = int(predictions.duplicated(key).sum())
    attendance_provenance = provenance[provenance["source_type"] == "attendance"]
    future_available_sources = 0
    for row in attendance_provenance.itertuples(index=False):
        for value in json.loads(row.available_source_dates):
            future_available_sources += int(pd.Timestamp(value) > pd.Timestamp(row.forecast_origin))
    preprocessing_future = int(
        (pd.to_datetime(predictions["training_end_date"]) > pd.to_datetime(predictions["forecast_origin"])).sum()
    )
    s2 = provenance[
        (provenance["scenario"] == "S2_same_weekend_sunday")
        & (provenance["feature"].isin(ATTENDANCE_FEATURES))
    ]
    s2_saturday_available = 0
    for row in s2.itertuples(index=False):
        for value in json.loads(row.available_source_dates):
            date = pd.Timestamp(value)
            s2_saturday_available += int(date.weekday() == 5 and date > pd.Timestamp(row.forecast_origin))
    validations = {
        "duplicate_prediction_keys": duplicate_count,
        "attendance_sources_after_origin": future_available_sources,
        "training_end_after_origin": preprocessing_future,
        "s2_future_saturday_sources": s2_saturday_available,
        "preprocessing_rows_marked_future": int(preprocessing["fit_includes_test_or_future"].sum()),
        "nonfinite_point_predictions": int((~np.isfinite(predictions["point_prediction"])).sum()),
        "nonfinite_quantile_predictions": int((~np.isfinite(predictions["quantile_prediction"])).sum()),
    }
    if any(validations.values()):
        raise AssertionError(f"Phase 1 validation failed: {validations}")
    return validations


def phase2_recommendation(
    weather: pd.DataFrame,
    baselines: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    origin = comparison[
        (comparison["model"] == "origin_aware_current_model")
        & (comparison["weekday_policy"] == T1_VALID_WEEKENDS)
        & (comparison["weather_policy"] == W0_NO_WEATHER)
        & (comparison["breakdown"] == "overall")
    ].iloc[0]
    baseline = baselines[
        (baselines["weekday_policy"] == T1_VALID_WEEKENDS) & (baselines["breakdown"] == "overall")
    ].sort_values("mae").iloc[0]
    weather_overall = weather[
        (weather["weekday_policy"] == T1_VALID_WEEKENDS) & (weather["breakdown"] == "overall")
    ].set_index("weather_policy")
    w_delta = weather_overall.loc[W1_OBSERVED_REPLAY, "mae"] - weather_overall.loc[W0_NO_WEATHER, "mae"]
    recent = diagnostics[
        (diagnostics["weekday_policy"] == T1_VALID_WEEKENDS)
        & (diagnostics["weather_policy"] == W0_NO_WEATHER)
        & (diagnostics["breakdown"] == "period")
    ].set_index("breakdown_value")
    long_missing = feature_matrix[["feature", "missing_rate_S5"]].sort_values("missing_rate_S5", ascending=False).head(5)
    recommend_feature_repair = origin["mae"] > baseline["mae"] or long_missing["missing_rate_S5"].max() >= 0.25
    path = "Phase 2A: feature repair" if recommend_feature_repair else "Phase 2B: recency/window experiments"
    return f"""# Phase 2 recommendation

## Recommended next path: {path}

Preferred T1/W0 origin-aware MAE is {origin['mae']:.3f}. The best fixed simple
baseline is `{baseline['model']}` at {baseline['mae']:.3f}. The largest S5 raw
missingness among current features is {long_missing['missing_rate_S5'].max():.1%}.
These measurements {'trigger' if recommend_feature_repair else 'do not trigger'}
the Phase 1 feature-repair decision rule.

Recent-52 MAE is {recent.loc['Recent 52', 'mae']:.3f} versus
{recent.loc['Earlier', 'mae']:.3f} earlier, so recency/window experiments should
{'follow feature repair' if recommend_feature_repair else 'be the first experiment block'}.

W1 minus W0 MAE is {w_delta:+.3f}. W2 remains unavailable; therefore any claimed
weather benefit is still deployment-uncertain. A weather-data phase should precede
weather-dependent model optimization if that observed-replay gap is operationally
material.

## Most missing S5 features

{markdown_table(long_missing)}

Do not select or tune a new estimator yet. Phase 2 should first test repaired
origin-valid feature semantics and, separately, recency windows using the Phase 1
evaluation harness.
"""


def summary_artifact(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    baselines: pd.DataFrame,
    comparison: pd.DataFrame,
    validations: dict[str, Any],
) -> None:
    preferred = comparison[
        (comparison["model"] == "origin_aware_current_model")
        & (comparison["weekday_policy"] == T1_VALID_WEEKENDS)
        & (comparison["weather_policy"] == W0_NO_WEATHER)
        & (comparison["breakdown"].isin(["overall", "day_type", "service_horizon"]))
    ]
    headline = preferred[preferred["breakdown"] == "overall"].iloc[0]
    best = baselines[
        (baselines["weekday_policy"] == T1_VALID_WEEKENDS) & (baselines["breakdown"] == "overall")
    ].sort_values("mae").iloc[0]
    weather_overall = weather[weather["breakdown"] == "overall"]
    replay = predictions[
        (predictions["weekday_policy"] == T1_VALID_WEEKENDS)
        & (predictions["weather_policy"] == W0_NO_WEATHER)
    ]
    write_text(
        "01_phase1_summary.md",
        f"""# Phase 1 summary

Phase 1 is complete as an isolated evaluation path. The preferred validity result
is T1/W0: valid configured weekends with weather disabled and fold-locally imputed.
It evaluates {int(headline['row_count'])} scenario-target rows with MAE
{headline['mae']:.3f}, RMSE {headline['rmse']:.3f}, bias
{headline['mean_signed_error']:.3f}, and raw 0.8-quantile coverage
{headline['raw_quantile_coverage']:.1%}. Its aligned legacy MAE is
{headline['aligned_legacy_mae']:.3f}; the origin-aware delta is
{headline['mae_delta_vs_aligned_legacy']:+.3f}.

The best fixed attendance baseline is `{best['model']}` with MAE {best['mae']:.3f}.
Production-rule replay coverage is {replay['production_rule_replay_covers'].mean():.1%},
but this is descriptive only because the saved residual buffers overlap the legacy
backtest. Raw model and replay outputs remain separate in the predictions file.

## Preferred result by day type and service horizon

{markdown_table(preferred)}

## Weather-policy headline

{markdown_table(weather_overall)}

## Validity conclusions

- Every prediction has explicit origin and service horizon; {validations['attendance_sources_after_origin']} available provenance sources occur after an origin.
- Fold-local preprocessing has {validations['training_end_after_origin']} training cutoffs after an origin and {validations['preprocessing_rows_marked_future']} diagnostics marked as future-inclusive.
- W2 is unavailable because archived issue-time forecasts do not exist.
- T1 excludes the Tuesday record and is preferred under the configured Saturday/Sunday validator.
- The legacy framework is not trustworthy for operational horizons because it uses target-relative attendance that can be unavailable at forecast time and full-frame imputation.
- `artifacts/` is ignored by `.gitignore`; Phase 1 reports therefore appear only under `git status --short --ignored`.

See `14_phase2_recommendation.md` for the measured next-step decision. Phase 2 was
not started.
""",
    )


def test_report_artifact(
    *,
    targeted_result: str,
    full_result: str,
    validations: dict[str, Any],
    production_hashes: dict[str, str],
    model_hash: str,
) -> None:
    write_text(
        "13_test_and_reproducibility_report.md",
        f"""# Test and reproducibility report

## Test results

- Targeted: `{targeted_result}`
- Full suite: `{full_result}`
- In-memory acceptance checks: `{json.dumps(validations, sort_keys=True)}`

## Clean backtest command

```bash
/tmp/soup-kitchen-forecast-phase1-venv/bin/python scripts/run_phase1_origin_backtest.py \\
  --targeted-test-result "14 passed" \\
  --full-test-result "38 passed, 9 subtests passed"
```

The environment was created with Python 3.12 and `pip install -r requirements.txt
pytest`, yielding scikit-learn {sklearn.__version__}. The host Python 3.13/scikit-
learn 1.9 environment was rejected for saved-package loading.

## Unchanged protected files at generation time

{markdown_table(pd.DataFrame([{'path': key, 'sha256': value} for key, value in production_hashes.items()]))}

Saved model package SHA-256: `{model_hash}`. The script never dumps a model and
never calls the live predictor or weather network.
""",
    )


def readme_artifact() -> None:
    rows = []
    for name in EXPECTED_ARTIFACTS:
        if name == "README.md":
            purpose = "Artifact index and reproduction entry point"
        elif name.endswith(".csv"):
            purpose = "Machine-readable Phase 1 evidence"
        elif name.endswith(".json"):
            purpose = "Machine-readable run manifest"
        else:
            purpose = "Human-readable Phase 1 report"
        rows.append({"file": name, "purpose": purpose})
    write_text(
        "README.md",
        f"""# Phase 1 origin-aware backtest artifacts

This directory is the complete NY 12550 Phase 1 evaluation package. It uses the
saved package attendance snapshot, exact current estimators, fold-local median
imputation, five deterministic origin scenarios, W0/W1 weather policies, and
T0/T1 weekday policies. W2 is explicitly unavailable. No production or model
package file is changed.

## Artifact index

{markdown_table(pd.DataFrame(rows))}

Start with `01_phase1_summary.md`, then use the prediction and diagnostics CSVs
for exact row-level evidence. The documented command is in
`13_test_and_reproducibility_report.md`.
""",
    )


def manifest_artifact(
    *,
    args: argparse.Namespace,
    source_summary_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    skipped: pd.DataFrame,
    validations: dict[str, Any],
    tuesday: dict[str, Any],
    production_hashes: dict[str, str],
    commit_before: str,
    branch: str,
) -> None:
    input_paths = [
        MODEL_PATH,
        SQLITE_PATH,
        CSV_PATH,
        WEATHER_PATH,
        LEGACY_PREDICTIONS_PATH,
        *sorted(AUDIT_DIR.glob("*")),
    ]
    input_files = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in input_paths
        if path.is_file()
    ]
    artifact_files = [name for name in EXPECTED_ARTIFACTS if name != "phase1_manifest.json"]
    artifact_hashes = {
        name: sha256_file(OUTPUT_DIR / name) for name in artifact_files if (OUTPUT_DIR / name).exists()
    }
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_commit_before_work": commit_before,
        "git_commit_after_work": git(["rev-parse", "HEAD"]),
        "python_version": sys.version,
        "active_environment": sys.prefix,
        "dependency_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "commands_executed": [
            "find/rg/sed/nl repository and Phase 0 inspection commands",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv /tmp/soup-kitchen-forecast-phase1-venv",
            "/tmp/soup-kitchen-forecast-phase1-venv/bin/python -m pip install -r requirements.txt pytest",
            "/tmp/soup-kitchen-forecast-phase1-venv/bin/python -m pytest -q tests/test_origin_features.py tests/test_origin_backtest.py",
            "/tmp/soup-kitchen-forecast-phase1-venv/bin/python -m pytest -q",
            f"/tmp/soup-kitchen-forecast-phase1-venv/bin/python scripts/run_phase1_origin_backtest.py --targeted-test-result {args.targeted_test_result!r} --full-test-result {args.full_test_result!r}",
        ],
        "input_files": input_files,
        "input_fingerprints": source_summary_frame[
            ["source", "stable_attendance_sha256", "file_sha256"]
        ].to_dict("records"),
        "source_of_truth_attendance_decision": "saved_model_package_history: exact committed location-backtest snapshot through 2026-06-21; no source merge",
        "code_files_created": [
            "src/origin_features.py",
            "src/origin_backtest.py",
            "scripts/run_phase1_origin_backtest.py",
            "tests/test_origin_features.py",
            "tests/test_origin_backtest.py",
        ],
        "code_files_modified": [],
        "production_file_hashes_at_generation": production_hashes,
        "artifact_files_created": EXPECTED_ARTIFACTS,
        "artifact_sha256_excluding_manifest": artifact_hashes,
        "existing_files_overwritten": [],
        "test_results": {
            "targeted": args.targeted_test_result,
            "full": args.full_test_result,
            "acceptance_validations": validations,
        },
        "scenario_definitions": SCENARIO_DEFINITIONS,
        "weather_policies": {
            W0_NO_WEATHER: "all weather missing then fold-local imputation",
            W1_OBSERVED_REPLAY: "realized target-date cache replay; not origin-valid",
            W2_ARCHIVED_FORECAST: "unavailable; no issue-time archive",
        },
        "weekday_policies": {
            T0_LEGACY_ALL: "preserve all package records and is_sun routing",
            T1_VALID_WEEKENDS: "Saturday/Sunday targets, histories, and segments only; preferred",
        },
        "model_configurations": {
            "point": "RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=2,random_state=42)",
            "quantile": "HistGradientBoostingRegressor(loss='quantile',quantile=0.8,learning_rate=0.05,max_depth=4,max_iter=500,random_state=42)",
            "minimum_segment_training_rows": 18,
            "preprocessing": "SimpleImputer(strategy='median', keep_empty_features=True), fit per segment fold",
            "calibration": "identity placeholder only; no optimization",
        },
        "random_seeds": [42],
        "row_counts": {
            "predictions_total": int(len(predictions)),
            "by_scenario": predictions.groupby("scenario").size().astype(int).to_dict(),
            "by_weather_policy": predictions.groupby("weather_policy").size().astype(int).to_dict(),
            "by_weekday_policy": predictions.groupby("weekday_policy").size().astype(int).to_dict(),
            "skipped_insufficient_folds": int(len(skipped)),
        },
        "tuesday_record": tuesday,
        "known_limitations": [
            "W1 is realized-weather hindsight replay and W2 is unavailable.",
            "Training feature rows use one-step origins; scenario-specific missingness appears only in evaluation rows.",
            "Current same-slot features intentionally retain cross-day-type semantics.",
            "Saved residual-buffer replay is not out-of-sample calibrated coverage.",
            "Package history is an engineered saved snapshot because the raw workbook is absent.",
        ],
        "ignored_output_behavior": "artifacts/ is ignored by .gitignore; reports appear with git status --short --ignored, not ordinary status",
        "platform": platform.platform(),
    }
    (OUTPUT_DIR / "phase1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-test-result", default="not supplied")
    parser.add_argument("--full-test-result", default="not supplied")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    design = OUTPUT_DIR / "00_implementation_design.md"
    if not design.exists():
        raise FileNotFoundError("Write 00_implementation_design.md before running Phase 1")
    existing = [name for name in EXPECTED_ARTIFACTS[1:] if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Phase 1 artifacts: {existing}")

    branch = git(["branch", "--show-current"])
    commit_before = git(["rev-parse", "HEAD"])
    production_paths = [
        ROOT / "app.py",
        ROOT / "src/predictor.py",
        ROOT / "src/features.py",
        ROOT / "src/modeling.py",
        ROOT / "scripts/train_backtest.py",
    ]
    production_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in production_paths}
    model_hash = sha256_file(MODEL_PATH)

    print("Loading saved package and reconciling attendance sources...", flush=True)
    package = joblib.load(MODEL_PATH)
    authority, sqlite_history, csv_history = load_sources(package)
    source_summary_frame = reconciliation_artifacts(authority, sqlite_history, csv_history)
    weather_df = pd.read_csv(WEATHER_PATH)
    legacy_predictions = pd.read_csv(LEGACY_PREDICTIONS_PATH, parse_dates=[DATE_COL])

    print("Running exact expanding-fold models for W0/W1 and T0/T1...", flush=True)
    evaluator = OriginAwareBacktester(
        authority,
        weather_df=weather_df,
        feature_cols=package["feature_cols"],
        residual_buffer_by_day=package.get("residual_buffer_by_day"),
        default_meal_buffer_pct=package.get("default_meal_buffer_pct", 0.08),
        min_train_size=18,
        quantile=0.8,
    )
    result = evaluator.run(legacy_predictions=legacy_predictions)
    predictions = add_periods(result.predictions, authority)
    write_csv("05_origin_aware_predictions.csv", predictions)
    provenance_sample_artifact(result.feature_provenance, predictions)
    write_csv("07_fold_preprocessing_diagnostics.csv", result.preprocessing_diagnostics)

    print("Calculating comparisons and reports...", flush=True)
    validations = validate_in_memory(
        predictions,
        result.feature_provenance,
        result.preprocessing_diagnostics,
    )
    origin_definition_artifact()
    feature_matrix = feature_status_artifacts(result.feature_provenance)
    weather_comparison = weather_comparison_artifacts(predictions)
    tuesday = tuesday_artifact(predictions, authority)
    baseline_comparison = baseline_comparison_artifacts(predictions, authority)
    legacy_comparison = legacy_comparison_artifacts(
        predictions,
        legacy_predictions,
        authority,
        baseline_comparison,
    )
    diagnostics = error_artifacts(predictions, authority)
    test_report_artifact(
        targeted_result=args.targeted_test_result,
        full_result=args.full_test_result,
        validations=validations,
        production_hashes=production_hashes,
        model_hash=model_hash,
    )
    write_text(
        "14_phase2_recommendation.md",
        phase2_recommendation(
            weather_comparison,
            baseline_comparison,
            legacy_comparison,
            feature_matrix,
            diagnostics,
        ),
    )
    summary_artifact(
        predictions,
        weather_comparison,
        baseline_comparison,
        legacy_comparison,
        validations,
    )
    readme_artifact()
    manifest_artifact(
        args=args,
        source_summary_frame=source_summary_frame,
        predictions=predictions,
        skipped=result.skipped_folds,
        validations=validations,
        tuesday=tuesday,
        production_hashes=production_hashes,
        commit_before=commit_before,
        branch=branch,
    )

    missing = [name for name in EXPECTED_ARTIFACTS if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise AssertionError(f"Missing required artifacts: {missing}")
    current_production_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in production_paths
    }
    if current_production_hashes != production_hashes or sha256_file(MODEL_PATH) != model_hash:
        raise AssertionError("Protected production file or saved model changed during Phase 1")
    print(f"Phase 1 complete: {len(predictions)} predictions, {len(EXPECTED_ARTIFACTS)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
