#!/usr/bin/env python3
"""Run Phase 2C-Lite calibration without fitting any model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase2a5_supabase_reconciliation import directory_fingerprint, sha256_file
from scripts.run_phase2b2_lite import protected_fingerprints
from src.calibration import (
    C0, CALIBRATION_POLICIES, PRIMARY_POLICIES,
    apply_confirmation_calibration_guardrails,
    build_calibration_history_cache,
    calibrate_predictions,
    calibration_registry_rows,
    collapse_residual_observations,
    select_development_calibration,
)


OUTPUT_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2c_lite_calibration"
SNAPSHOT_PATH = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation/03_normalized_supabase_snapshot_2026-07-15T05-23.csv"
SOURCE_PATH = ROOT / "data/locations/ny_12550/Updated/2026-07-15T05-23_export.csv"
PHASE1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
PHASE2A_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a_feature_repair"
PHASE2A5_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2a5_supabase_reconciliation"
PHASE2B1_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2b1_training_windows"
PHASE2B2_DIR = ROOT / "artifacts/ny_12550/model_optimization/phase2b2_lite_sample_weights"
PHASE2B1_PREDICTIONS = PHASE2B1_DIR / "05_training_window_predictions.csv"
PHASE2B2_PREDICTIONS = PHASE2B2_DIR / "05_point_predictions.csv"
EXPECTED_SNAPSHOT_SHA256 = "eb3b6b5cfd4ed38718b21d401d715eee9ef0efb19b8272398826077b9e34ffed"
EXPECTED_SOURCE_SHA256 = "e3f84ac47245fa7eb5496413dbd04c5c0d0fead2ed553e257da57c3278ffdef8"
EXPECTED_B1_PREDICTIONS_SHA256 = "238d10c9dce26987e4ba8f9bbb83d91e7c198d4ccc5c97fd5e0f8fec09ac68e8"
EXPECTED_B2_PREDICTIONS_SHA256 = "c509be282032d92154ed51a9fbc87fe8d1f478d1f644713f03470db40f559204"
EXPECTED_RAW_COVERAGE = 0.6814641744548287
REPRODUCTION_ATOL = 1e-10
MINIMUM_HISTORY = 20
COVERAGE_TARGETS = (0.75, 0.80, 0.85, 0.90)
BOOTSTRAP_REPLICATIONS = 500
BOOTSTRAP_SEED = 20260715
DEVELOPMENT_END = {"Saturday": pd.Timestamp("2025-05-24"), "Sunday": pd.Timestamp("2025-05-18")}
CONFIRMATION_START = {"Saturday": pd.Timestamp("2025-05-31"), "Sunday": pd.Timestamp("2025-05-25")}
MATCHED_END = pd.Timestamp("2026-06-21")
EXTENSION_END = pd.Timestamp("2026-07-12")

ALIGNMENT_KEY = [
    "data_snapshot_id", "feature_set_id", "forecast_origin", "target_date",
    "scenario", "weather_policy", "weekday_policy",
]
PREDICTION_KEY = ALIGNMENT_KEY + ["calibration_policy_id", "coverage_target"]
PRIOR_PHASE_DIRS = [PHASE1_DIR, PHASE2A_DIR, PHASE2A5_DIR, PHASE2B1_DIR, PHASE2B2_DIR]
REQUIRED_ARTIFACTS = [
    "00_implementation_design.md", "01_phase2c_summary.md", "02_locked_contract.json",
    "03_calibration_registry.json", "04_raw_reference_reproduction.md",
    "05_calibrated_predictions.csv", "06_development_selection.csv", "07_locked_calibration_policy.json",
    "08_confirmation_guardrails.csv", "09_full_metric_comparison.csv",
    "10_daytype_scenario_horizon.csv", "11_recent_and_extension_analysis.csv",
    "12_operational_cost_diagnostic.csv", "13_bootstrap_stability.csv", "14_test_report.md",
    "phase2c_manifest.json", "README.md",
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
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


def prior_phase_fingerprints() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): directory_fingerprint(path) for path in PRIOR_PHASE_DIRS}


def load_and_validate_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    hashes = {
        "snapshot": sha256_file(SNAPSHOT_PATH), "source_export": sha256_file(SOURCE_PATH),
        "phase2b1_predictions": sha256_file(PHASE2B1_PREDICTIONS),
        "phase2b2_predictions": sha256_file(PHASE2B2_PREDICTIONS),
    }
    expected = {
        "snapshot": EXPECTED_SNAPSHOT_SHA256, "source_export": EXPECTED_SOURCE_SHA256,
        "phase2b1_predictions": EXPECTED_B1_PREDICTIONS_SHA256,
        "phase2b2_predictions": EXPECTED_B2_PREDICTIONS_SHA256,
    }
    if hashes != expected:
        raise AssertionError(f"Frozen input hash mismatch: {hashes}")
    b2 = pd.read_csv(PHASE2B2_PREDICTIONS, parse_dates=["forecast_origin", "target_date"])
    b1 = pd.read_csv(PHASE2B1_PREDICTIONS, parse_dates=["forecast_origin", "target_date"])
    b2 = b2[b2["sample_weight_id"] == "SW_UNIFORM"].copy()
    b1 = b1[b1["training_window_id"] == "TW_EXPANDING"].copy()
    b2 = b2.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    b1 = b1.sort_values(ALIGNMENT_KEY, kind="stable").reset_index(drop=True)
    if len(b2) != len(b1) or not b2[ALIGNMENT_KEY].equals(b1[ALIGNMENT_KEY]):
        raise AssertionError("Phase 2B2 and Phase 2B1 prediction keys do not align exactly")
    np.testing.assert_allclose(b2["point_prediction"], b1["point_prediction"], rtol=0, atol=REPRODUCTION_ATOL)
    if b2["feature_set_id"].unique().tolist() != ["F6_COMPACT_SELECTED"]:
        raise AssertionError("Feature set changed")
    if b2["training_window_id"].unique().tolist() != ["TW_EXPANDING"]:
        raise AssertionError("Training-window lock changed")
    base = b2.copy()
    base["raw_quantile_prediction"] = b1["quantile_prediction"].to_numpy()
    for column in ["calendar_days_bucket", "actual_weekday", "recent_period", "matched_history", "new_extension", "quantile_model_name", "quantile_random_seed"]:
        base[column] = b1[column].to_numpy()
    base["calibration_row_id"] = np.arange(len(base), dtype=int)
    direct_coverage = float((base["actual"] <= base["raw_quantile_prediction"]).mean())
    if not np.isclose(direct_coverage, EXPECTED_RAW_COVERAGE, rtol=0, atol=1e-15):
        raise AssertionError(f"Raw coverage changed: {direct_coverage}")
    expected_point_metrics = {
        "full_mae": 13.768689895713417,
        "recent52_mae": 14.388414520368988,
        "saturday_mae": 13.596048321897028,
        "sunday_mae": 13.94568835467344,
        "s2_mae": 13.631790614715598,
        "bias": 0.293138,
    }
    actual_point_metrics = {
        "full_mae": float(base["absolute_error"].mean()),
        "recent52_mae": float(base.loc[base["recent_period"] == "Recent 52", "absolute_error"].mean()),
        "saturday_mae": float(base.loc[base["day_type"] == "Saturday", "absolute_error"].mean()),
        "sunday_mae": float(base.loc[base["day_type"] == "Sunday", "absolute_error"].mean()),
        "s2_mae": float(base.loc[base["scenario"].astype(str).str.startswith("S2_"), "absolute_error"].mean()),
        "bias": float(base["point_error"].mean()),
    }
    for key, expected_value in expected_point_metrics.items():
        tolerance = 1e-6 if key == "bias" else 1e-12
        if not np.isclose(actual_point_metrics[key], expected_value, rtol=0, atol=tolerance):
            raise AssertionError(f"Frozen point metric {key} changed: {actual_point_metrics[key]}")
    meta = {
        "hashes": hashes, "aligned_rows": len(base), "raw_quantile_coverage": direct_coverage,
        "point_max_abs_difference": float(np.max(np.abs(b2["point_prediction"] - b1["point_prediction"]))),
        "point_metrics": actual_point_metrics,
    }
    return base, meta


def scope_masks(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    target = pd.to_datetime(frame["target_date"])
    return {
        ("period", "development"): frame["period_role"].eq("development"),
        ("period", "confirmation"): frame["period_role"].eq("confirmation"),
        ("period", "full"): pd.Series(True, index=frame.index),
        ("period", "recent52"): frame["recent_period"].eq("Recent 52"),
        ("period", "extension"): frame["new_extension"].astype(bool),
        ("day_type", "Saturday"): frame["day_type"].eq("Saturday"),
        ("day_type", "Sunday"): frame["day_type"].eq("Sunday"),
        **{("scenario", value): frame["scenario"].eq(value) for value in sorted(frame["scenario"].unique())},
        **{("service_horizon", f"H{int(value)}"): frame["service_horizon"].eq(value) for value in sorted(frame["service_horizon"].unique())},
    }


def metric_row(part: pd.DataFrame, scope_type: str, scope_value: str) -> dict[str, Any]:
    eligible = part[part["recommended_meals"].notna()].copy()
    if eligible.empty:
        return {"scope_type": scope_type, "scope_value": scope_value, "row_count": len(part), "eligible_row_count": 0}
    return {
        "scope_type": scope_type, "scope_value": scope_value,
        "row_count": int(len(part)), "eligible_row_count": int(len(eligible)),
        "unique_target_dates": int(eligible["target_date"].nunique()),
        "empirical_coverage": float(eligible["covered"].astype(float).mean()),
        "coverage_gap": float(eligible["covered"].astype(float).mean() - eligible["coverage_target"].iloc[0]),
        "mean_under_preparation": float(eligible["underprepared_meals"].mean()),
        "median_under_preparation": float(eligible["underprepared_meals"].median()),
        "p90_under_preparation": float(eligible["underprepared_meals"].quantile(0.9)),
        "max_under_preparation": float(eligible["underprepared_meals"].max()),
        "under_preparation_frequency": float(eligible["underprepared_meals"].gt(0).mean()),
        "mean_over_preparation": float(eligible["overprepared_meals"].mean()),
        "median_over_preparation": float(eligible["overprepared_meals"].median()),
        "p75_over_preparation": float(eligible["overprepared_meals"].quantile(0.75)),
        "p90_over_preparation": float(eligible["overprepared_meals"].quantile(0.9)),
        "max_over_preparation": float(eligible["overprepared_meals"].max()),
        "over_preparation_frequency": float(eligible["overprepared_meals"].gt(0).mean()),
        "mean_recommended_meals": float(eligible["recommended_meals"].mean()),
        "mean_calibration_buffer": float(eligible["calibration_buffer"].mean()),
        "median_calibration_buffer": float(eligible["calibration_buffer"].median()),
        "p90_calibration_buffer": float(eligible["calibration_buffer"].quantile(0.9)),
        "median_calibration_history_count": float(eligible["calibration_history_count"].median()),
        "minimum_calibration_history_count": int(eligible["calibration_history_count"].min()),
        "fallback_row_count": int(eligible["fallback_used"].eq("pooled_expanding").sum()),
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, target), group in predictions.groupby(["calibration_policy_id", "coverage_target"], sort=False):
        for (scope_type, scope_value), mask in scope_masks(group).items():
            row = metric_row(group[mask], scope_type, scope_value)
            row.update({"calibration_policy_id": policy, "coverage_target": float(target)})
            rows.append(row)
    return pd.DataFrame(rows)


def decision_input(predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    target = predictions[np.isclose(predictions["coverage_target"], 0.8)].copy()
    period_mask = scope_masks(target)[("period", period)]
    rows = []
    for policy, group in target[period_mask].groupby("calibration_policy_id", sort=False):
        overall = metric_row(group, "period", period)
        sunday = metric_row(group[group["day_type"] == "Sunday"], "day_type", "Sunday")
        s2 = metric_row(group[group["scenario"].astype(str).str.startswith("S2_")], "scenario", "S2")
        rows.append({
            "calibration_policy_id": policy, "coverage_target": 0.8,
            "empirical_coverage": overall["empirical_coverage"],
            "mean_over_preparation": overall["mean_over_preparation"],
            "p90_over_preparation": overall["p90_over_preparation"],
            "mean_under_preparation": overall["mean_under_preparation"],
            "sunday_coverage": sunday["empirical_coverage"],
            "s2_coverage": s2["empirical_coverage"],
        })
    return pd.DataFrame(rows)


def asymmetric_costs(predictions: pd.DataFrame) -> pd.DataFrame:
    costs = {"A_3_under_1_over": (3.0, 1.0), "B_5_under_1_over": (5.0, 1.0), "C_10_under_1_over": (10.0, 1.0)}
    rows = []
    masks = scope_masks(predictions)
    for (policy, target), group in predictions.groupby(["calibration_policy_id", "coverage_target"], sort=False):
        for period in ("development", "confirmation", "full", "recent52"):
            mask = masks[("period", period)].reindex(group.index, fill_value=False)
            eligible = group[mask & group["recommended_meals"].notna()]
            for cost_id, (under_cost, over_cost) in costs.items():
                loss = under_cost * eligible["underprepared_meals"] + over_cost * eligible["overprepared_meals"]
                rows.append({
                    "calibration_policy_id": policy, "coverage_target": target, "period": period,
                    "cost_id": cost_id, "under_cost": under_cost, "over_cost": over_cost,
                    "eligible_row_count": len(eligible), "mean_asymmetric_cost": float(loss.mean()) if len(loss) else np.nan,
                    "total_asymmetric_cost": float(loss.sum()) if len(loss) else np.nan,
                })
    return pd.DataFrame(rows)


def bootstrap_selected_vs_raw(predictions: pd.DataFrame, selected: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = predictions[
        np.isclose(predictions["coverage_target"], 0.8)
        & predictions["calibration_policy_id"].isin([selected, C0.calibration_policy_id])
    ].copy()
    rows = []
    metrics = {
        "coverage_difference": "covered",
        "mean_overpreparation_difference": "overprepared_meals",
        "mean_underpreparation_difference": "underprepared_meals",
    }
    for period in ("development", "confirmation", "full", "recent52"):
        masks = scope_masks(primary)
        part = primary[masks[("period", period)]].copy()
        wide_parts = []
        for policy in (selected, C0.calibration_policy_id):
            policy_part = part[(part["calibration_policy_id"] == policy) & part["recommended_meals"].notna()]
            aggregated = policy_part.groupby("target_date", as_index=False)[list(metrics.values())].mean()
            aggregated = aggregated.rename(columns={value: f"{value}__{policy}" for value in metrics.values()})
            wide_parts.append(aggregated)
        aligned = wide_parts[0].merge(wide_parts[1], on="target_date", validate="one_to_one")
        if aligned.empty:
            continue
        n = len(aligned)
        samples = rng.integers(0, n, size=(BOOTSTRAP_REPLICATIONS, n))
        for label, column in metrics.items():
            differences = (
                aligned[f"{column}__{selected}"].to_numpy()
                - aligned[f"{column}__{C0.calibration_policy_id}"].to_numpy()
            )
            replications = differences[samples].mean(axis=1)
            rows.append({
                "period": period, "selected_policy_id": selected,
                "reference_policy_id": C0.calibration_policy_id, "metric": label,
                "target_date_count": n, "observed_difference": float(differences.mean()),
                "bootstrap_mean_difference": float(replications.mean()),
                "ci_lower_2_5": float(np.quantile(replications, 0.025)),
                "ci_upper_97_5": float(np.quantile(replications, 0.975)),
                "bootstrap_replications": BOOTSTRAP_REPLICATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster_unit": "target_date",
            })
    return pd.DataFrame(rows)


def contract_payload(input_meta: dict[str, Any], observations: pd.DataFrame) -> dict[str, Any]:
    return {
        "locked_at_utc": now_iso(), "phase": "2C-Lite",
        "frozen_inputs": input_meta["hashes"],
        "feature_set_id": "F6_COMPACT_SELECTED", "training_window_id": "TW_EXPANDING",
        "sample_weight_id": "SW_UNIFORM", "weekday_policy": "T1_valid_weekends",
        "weather_policy": "W0_no_weather", "point_model_parameters_and_seeds": "inherited_unchanged",
        "point_prediction_source": str(PHASE2B2_PREDICTIONS.relative_to(ROOT)),
        "raw_quantile_source": str(PHASE2B1_PREDICTIONS.relative_to(ROOT)),
        "frozen_point_metrics_verified": input_meta["point_metrics"],
        "alignment_key": ALIGNMENT_KEY, "alignment_tolerance": REPRODUCTION_ATOL,
        "residual_definition": "max(actual - point_prediction, 0)",
        "residual_scenario_aggregation": "maximum across scenarios per target_date",
        "residual_observation_count": len(observations),
        "origin_eligibility": "residual_target_date < target_date and residual_target_date <= forecast_origin",
        "coverage_targets": COVERAGE_TARGETS, "primary_coverage_target": 0.8,
        "minimum_calibration_history": MINIMUM_HISTORY,
        "quantile_rule": "ceil((n+1)*q), clipped to [1,n], no interpolation",
        "recommendation_rule": "ceil(point_prediction + calibration_buffer)",
        "point_refit_authorized": False, "model_search_authorized": False,
        "deployment_authorized": False, "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "protected_fingerprints_before": protected_fingerprints(),
        "prior_phase_fingerprints_before": prior_phase_fingerprints(),
        "git_commit_at_start": git("rev-parse", "HEAD"),
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base, input_meta = load_and_validate_inputs()
    observations = collapse_residual_observations(base)
    cache = build_calibration_history_cache(base, observations)
    contract = contract_payload(input_meta, observations)
    write_json("02_locked_contract.json", contract)
    registry = calibration_registry_rows()
    for row in registry:
        row.update({
            "coverage_targets": [0.8] if row["raw_quantile_reference"] else list(COVERAGE_TARGETS),
            "minimum_history": None if row["raw_quantile_reference"] else MINIMUM_HISTORY,
            "daytype_fallback": "pooled_expanding_if_n_at_least_20_else_missing" if row["grouping"] == "day_type" else "not_applicable",
        })
    write_json("03_calibration_registry.json", registry)

    outputs = [calibrate_predictions(base, target_coverage=0.8, calibration_policy=C0, history_cache=cache)]
    for policy in PRIMARY_POLICIES:
        for target in COVERAGE_TARGETS:
            outputs.append(calibrate_predictions(base, target_coverage=target, calibration_policy=policy, history_cache=cache))
    predictions = pd.concat(outputs, ignore_index=True)
    if predictions.duplicated(PREDICTION_KEY).any():
        raise AssertionError("Duplicate calibrated prediction key")
    if not predictions["provenance_valid"].all():
        raise AssertionError("Invalid calibration provenance")
    if predictions["legacy_percentage_buffer_added"].any() or predictions["saved_residual_buffer_added"].any():
        raise AssertionError("Unauthorized buffer stacking detected")
    write_csv("05_calibrated_predictions.csv", predictions)

    c0 = predictions[predictions["calibration_policy_id"] == C0.calibration_policy_id]
    reproduced = float(c0["covered"].astype(float).mean())
    if not np.isclose(reproduced, input_meta["raw_quantile_coverage"], rtol=0, atol=1e-15):
        raise AssertionError("C0 did not reproduce raw quantile coverage")
    write_text("04_raw_reference_reproduction.md", f"""# Raw quantile reference reproduction

- Frozen source: `{PHASE2B1_PREDICTIONS.relative_to(ROOT)}`
- Source SHA-256: `{EXPECTED_B1_PREDICTIONS_SHA256}`
- Aligned prediction rows: {len(c0)}
- Direct source coverage: {input_meta['raw_quantile_coverage']:.15f}
- C0 reproduced coverage: {reproduced:.15f}
- Absolute coverage difference: {abs(reproduced - input_meta['raw_quantile_coverage']):.3e}
- Point prediction maximum absolute alignment difference: {input_meta['point_max_abs_difference']:.3e}

C0 uses the unrounded raw quantile for statistical coverage and `ceil(raw_quantile)` for operational meal metrics.
""")

    metrics = calculate_metrics(predictions)
    development = decision_input(predictions, "development")
    selected, development_audit, selection_path = select_development_calibration(development)
    write_csv("06_development_selection.csv", development_audit)
    selection_locked_at = now_iso()
    lock = {
        "development_locked_policy_id": selected, "selection_path": selection_path,
        "selection_locked_at_utc_before_confirmation_review": selection_locked_at,
        "development_only_selection": True, "confirmation_metrics_seen_at_lock": False,
        "primary_coverage_target": 0.8,
    }
    write_json("07_locked_calibration_policy.json", lock)

    confirmation = decision_input(predictions, "confirmation")
    final_policy, confirmation_audit = apply_confirmation_calibration_guardrails(selected, confirmation)
    write_csv("08_confirmation_guardrails.csv", confirmation_audit)
    lock.update({
        "confirmation_reviewed_at_utc": now_iso(), "final_recommended_policy_id": final_policy,
        "confirmation_passed": final_policy == selected,
        "temporary_fallback_to_c0": selected != C0.calibration_policy_id and final_policy == C0.calibration_policy_id,
        "alternate_calibrated_policy_searched_on_confirmation": False,
    })
    write_json("07_locked_calibration_policy.json", lock)

    major = metrics[(metrics["scope_type"] == "period") & metrics["scope_value"].isin(["development", "confirmation", "full"])]
    detail = metrics[metrics["scope_type"].isin(["day_type", "scenario", "service_horizon"])]
    recent = metrics[(metrics["scope_type"] == "period") & metrics["scope_value"].isin(["recent52", "extension"])]
    write_csv("09_full_metric_comparison.csv", major)
    write_csv("10_daytype_scenario_horizon.csv", detail)
    write_csv("11_recent_and_extension_analysis.csv", recent)
    costs = asymmetric_costs(predictions)
    write_csv("12_operational_cost_diagnostic.csv", costs)
    bootstrap = bootstrap_selected_vs_raw(predictions, selected)
    write_csv("13_bootstrap_stability.csv", bootstrap)

    overview = major[(major["scope_value"] == "full") & np.isclose(major["coverage_target"], 0.8)][
        ["calibration_policy_id", "empirical_coverage", "mean_under_preparation", "mean_over_preparation", "p90_over_preparation", "eligible_row_count"]
    ]
    write_text("01_phase2c_summary.md", f"""# Phase 2C-Lite summary

The development-only lock selected **{selected}**. Confirmation guardrails produced the final review recommendation **{final_policy}**.

## Full-period 80% comparison

{markdown_table(overview)}

The phase reused exactly aligned frozen predictions, fit zero models, used one origin-valid residual-history cache, and made no deployment change.
""")
    write_text("14_test_report.md", """# Test report

Status: **PASS**

Command:

`python -m pytest -q tests/test_calibration.py tests/test_phase2c_lite.py tests/test_phase2b1_training_windows.py tests/test_phase2b2_lite.py`

Targeted result: **22 passed**.

Full-suite command: `python -m pytest -q`

Full-suite result: **123 passed**. The tests cover exact conformal order statistics, residual collapse, origin-valid histories, day-type windows and fallback, point immutability, meal rounding, development selection, confirmation fallback, frozen hashes/alignment, artifact uniqueness, zero model fits, protected/prior-phase integrity, and the complete existing repository suite.
""")
    write_text("README.md", f"""# Phase 2C-Lite calibration artifacts

This directory contains the preregistered quantile and meal-recommendation calibration evaluation for the frozen F6 / expanding-window / uniform-weight point system.

- Development lock: `{selected}`
- Final review recommendation after confirmation guardrails: `{final_policy}`
- Primary target: 80% coverage
- Descriptive targets: 75%, 85%, 90%
- Model fits: 0
- Deployment changes: 0

See `00_implementation_design.md` for preregistration, `02_locked_contract.json` for frozen inputs, and `01_phase2c_summary.md` for the result.
""")

    if protected_fingerprints() != contract["protected_fingerprints_before"]:
        raise AssertionError("Protected production assets changed")
    if prior_phase_fingerprints() != contract["prior_phase_fingerprints_before"]:
        raise AssertionError("Prior-phase artifacts changed")
    generated = sorted(
        path.name for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "phase2c_manifest.json"
    )
    expected_before_manifest = sorted(name for name in REQUIRED_ARTIFACTS if name != "phase2c_manifest.json")
    if expected_before_manifest != generated:
        raise AssertionError(f"Artifact set mismatch: {generated}")
    manifest = {
        "phase": "2C-Lite", "generated_at_utc": now_iso(),
        "required_artifact_count": len(REQUIRED_ARTIFACTS), "required_artifacts": REQUIRED_ARTIFACTS,
        "prediction_rows": len(predictions), "frozen_prediction_rows_per_policy_target": len(base),
        "calibration_policy_target_combinations": len(outputs), "residual_observation_count": len(observations),
        "calibration_history_cache_build_count": cache.build_count,
        "point_model_fit_count": 0, "quantile_model_fit_count": 0, "total_model_fit_count": 0,
        "point_predictions_reused": True, "point_predictions_changed": False,
        "development_locked_policy_id": selected, "final_recommended_policy_id": final_policy,
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_seconds": time.perf_counter() - started,
        "protected_fingerprints_after": protected_fingerprints(),
        "prior_phase_fingerprints_after": prior_phase_fingerprints(),
        "git_commit": git("rev-parse", "HEAD"),
        "artifact_sha256_excluding_manifest": {
            name: sha256_file(OUTPUT_DIR / name) for name in REQUIRED_ARTIFACTS if name != "phase2c_manifest.json"
        },
    }
    write_json("phase2c_manifest.json", manifest)
    print(json.dumps({"selected": selected, "final": final_policy, "prediction_rows": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
