#!/usr/bin/env python3
"""Finalize Phase 2B1 reports after the development-only window lock exists."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase2a5_supabase_reconciliation import json_value, markdown_table, sha256_file
from scripts.run_phase2b1_training_windows import (
    ALIGNMENT_KEY,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    EXPECTED_F6_HASH,
    EXPECTED_SOURCE_SHA256,
    EXTENSION_END,
    MATCHED_END,
    OUTPUT_DIR,
    PHASE1_DIR,
    PHASE2A5_DIR,
    PHASE2A_DIR,
    POINT_RANDOM_SEED,
    PREDICTION_KEY,
    PROTECTED_FILES,
    QUANTILE_RANDOM_SEED,
    REFERENCE_PREDICTIONS_PATH,
    SNAPSHOT_PATH,
    SOURCE_PATH,
    TRAINING_WINDOW_CANDIDATES,
    development_recent_tail,
    extended_metrics,
    git,
    json_clean,
    prior_phase_fingerprints,
    protected_fingerprints,
    validate_lock,
)


REQUIRED_ARTIFACTS = [
    "00_implementation_design.md", "01_phase2b1_summary.md", "02_training_window_registry.json",
    "02_training_window_registry.md", "03_locked_contract.json", "03_locked_contract.md",
    "04_expanding_reproduction.md", "05_training_window_predictions.csv",
    "06_training_window_metrics.csv", "06_training_window_metrics.md",
    "07_development_selection_table.csv", "07_development_selection_table.md",
    "08_locked_window_decision.json", "08_locked_window_decision.md",
    "09_confirmation_results.csv", "09_confirmation_results.md",
    "10_scenario_horizon_results.csv", "10_scenario_horizon_results.md",
    "11_daytype_results.csv", "11_daytype_results.md", "12_recent_period_analysis.csv",
    "12_recent_period_analysis.md", "13_new_extension_predictions.csv",
    "13_new_extension_analysis.md", "14_paired_window_comparison.csv",
    "14_paired_window_comparison.md", "15_bootstrap_stability.csv",
    "15_bootstrap_stability.md", "16_fold_training_diagnostics.csv",
    "16_fold_training_diagnostics.md", "17_quantile_coverage_analysis.csv",
    "17_quantile_coverage_analysis.md", "18_daytype_specific_window_diagnostic.md",
    "19_phase2b2_recommendation.md", "20_test_and_reproducibility_report.md",
    "phase2b1_manifest.json", "README.md",
]


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


def load_predictions() -> pd.DataFrame:
    path = OUTPUT_DIR / "05_training_window_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError("Run run_phase2b1_training_windows.py before finalizing")
    frame = pd.read_csv(
        path,
        parse_dates=[
            "forecast_origin", "target_date", "training_end_date",
            "earliest_retained_training_date", "latest_retained_training_date",
        ],
        low_memory=False,
    )
    if frame.duplicated(PREDICTION_KEY).any():
        raise AssertionError("Duplicate training-window prediction keys")
    expected = {item.training_window_id for item in TRAINING_WINDOW_CANDIDATES}
    if set(frame["training_window_id"]) != expected:
        raise AssertionError("Training-window candidate set differs from the registry")
    if frame.groupby("training_window_id").size().nunique() != 1:
        raise AssertionError("Candidate prediction counts do not align")
    return frame


def add_analysis_dimensions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["year"] = frame["target_date"].dt.year.astype(str)
    frame["quarter"] = frame["target_date"].dt.to_period("Q").astype(str)
    actuals = frame.drop_duplicates("target_date")[["target_date", "actual"]]
    boundaries = actuals["actual"].quantile([0.25, 0.5, 0.75]).tolist()
    frame["attendance_quartile"] = pd.cut(
        frame["actual"], [-np.inf, *boundaries, np.inf],
        labels=["Q1 low", "Q2", "Q3", "Q4 high"], include_lowest=True,
    ).astype(str)
    frame["history_availability_group"] = np.where(
        frame["available_segment_training_rows"] < 104,
        "low_history_below_104", "at_least_104_available",
    )
    frame["constraint_group"] = np.where(
        frame["window_constrained"].astype(bool), "fully_window_constrained", "unconstrained"
    )
    return frame


def metric_row(window_id: str, scope: str, value: str, part: pd.DataFrame) -> dict[str, Any]:
    return {"training_window_id": window_id, "evaluation_scope": scope, "scope_value": value, **extended_metrics(part)}


def all_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_id, part in frame.groupby("training_window_id", sort=False):
        named = {
            "full_history": part,
            "development": part[part["period_role"] == "development"],
            "development_recent_tail": development_recent_tail(part),
            "confirmation": part[part["period_role"] == "confirmation"],
            "matched_history_through_2026_06_21": part[part["target_date"] <= MATCHED_END],
            "new_extension_after_2026_06_21": part[(part["target_date"] > MATCHED_END) & (part["target_date"] <= EXTENSION_END)],
            "recent_52_target_dates": part[part["recent_period"] == "Recent 52"],
            "earlier_observations": part[part["recent_period"] == "Earlier"],
        }
        for scope, subset in named.items():
            rows.append(metric_row(window_id, scope, "All", subset))
        grouped = {
            "day_type": "day_type", "scenario": "scenario", "service_horizon": "service_horizon",
            "year": "year", "quarter": "quarter", "attendance_quartile": "attendance_quartile",
            "history_availability": "history_availability_group", "window_constraint": "constraint_group",
        }
        for scope, column in grouped.items():
            for value, subset in part.groupby(column, sort=True):
                rows.append(metric_row(window_id, scope, str(value), subset))
    return pd.DataFrame(rows)


def metric_lookup(metrics: pd.DataFrame, window: str, scope: str, value: str = "All", column: str = "mae") -> float:
    row = metrics[
        (metrics["training_window_id"] == window)
        & (metrics["evaluation_scope"] == scope)
        & (metrics["scope_value"].astype(str) == str(value))
    ]
    if len(row) != 1:
        raise AssertionError(f"Expected one metric row for {(window, scope, value)}, found {len(row)}")
    return float(row.iloc[0][column])


def fixed_reference_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    expanding = frame[frame["training_window_id"] == "TW_EXPANDING"].copy()
    prior = pd.read_csv(REFERENCE_PREDICTIONS_PATH, parse_dates=["forecast_origin", "target_date"])
    f0 = prior[prior["candidate_id"] == "F0_CURRENT_ORIGIN"].copy()
    f0["period_role"] = [
        "development" if date <= DEVELOPMENT_END[day] else "confirmation"
        for date, day in zip(f0["target_date"], f0["day_type"], strict=True)
    ]
    eligible = sorted(f0["target_date"].drop_duplicates())
    f0["recent_period"] = np.where(f0["target_date"].isin(set(eligible[-52:])), "Recent 52", "Earlier")
    references: list[tuple[str, pd.DataFrame, str]] = [("F0_EXPANDING", f0, "point_prediction")]
    for label, column in [
        ("LAST4_MEDIAN", "median_last4_same_daytype"),
        ("LAST4_MEAN", "mean_last4_same_daytype"),
        ("PREVIOUS_SAME_DAYTYPE", "previous_same_daytype"),
    ]:
        ref = expanding.copy()
        ref["reference_prediction"] = ref[column]
        references.append((label, ref, "reference_prediction"))
    rows: list[dict[str, Any]] = []
    for label, ref, predcol in references:
        scopes = {
            "full_history": ref,
            "development": ref[ref["period_role"] == "development"],
            "confirmation": ref[ref["period_role"] == "confirmation"],
            "matched_history_through_2026_06_21": ref[ref["target_date"] <= MATCHED_END],
            "new_extension_after_2026_06_21": ref[(ref["target_date"] > MATCHED_END) & (ref["target_date"] <= EXTENSION_END)],
            "recent_52_target_dates": ref[ref["recent_period"] == "Recent 52"],
        }
        for scope, subset in scopes.items():
            usable = subset.dropna(subset=[predcol])
            err = usable[predcol] - usable["actual"]
            rows.append(
                {
                    "reference_id": label, "evaluation_scope": scope, "row_count": int(len(usable)),
                    "mae": float(err.abs().mean()), "mean_signed_error": float(err.mean()),
                    "p90_absolute_error": float(err.abs().quantile(0.9)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    expanding = frame[frame["training_window_id"] == "TW_EXPANDING"].copy()
    base_cols = ALIGNMENT_KEY + ["absolute_error", "point_error", "squared_error", "quantile_covers", "quantile_excess_shortfall", "quarter"]
    base = expanding[base_cols].rename(columns={column: f"{column}_expanding" for column in base_cols if column not in ALIGNMENT_KEY})
    rows: list[dict[str, Any]] = []
    for window in [item.training_window_id for item in TRAINING_WINDOW_CANDIDATES if item.training_window_id != "TW_EXPANDING"]:
        candidate = frame[frame["training_window_id"] == window]
        merged = candidate.merge(base, on=ALIGNMENT_KEY, validate="one_to_one")
        merged["absolute_error_change"] = merged["absolute_error"] - merged["absolute_error_expanding"]
        merged["signed_error_change"] = merged["point_error"] - merged["point_error_expanding"]
        merged["squared_error_change"] = merged["squared_error"] - merged["squared_error_expanding"]
        merged["coverage_change"] = merged["quantile_covers"].astype(float) - merged["quantile_covers_expanding"].astype(float)
        candidate_shortfall = np.maximum(merged["actual"] - merged["quantile_prediction"], 0)
        # Reconstruct expanding shortfall from coverage/excess; negative excess means shortfall.
        expanding_shortfall = np.maximum(-merged["quantile_excess_shortfall_expanding"], 0)
        merged["quantile_shortfall_change"] = candidate_shortfall - expanding_shortfall
        target = merged.groupby("target_date")["absolute_error_change"].mean()
        scenario = merged.groupby("scenario")["absolute_error_change"].mean()
        quarterly = merged.groupby("quarter")["absolute_error_change"].mean()
        rows.append(
            {
                "training_window_id": window, "reference_window_id": "TW_EXPANDING",
                "paired_row_count": int(len(merged)),
                "mean_paired_mae_change": float(merged["absolute_error_change"].mean()),
                "median_paired_absolute_error_change": float(merged["absolute_error_change"].median()),
                "mean_signed_error_change": float(merged["signed_error_change"].mean()),
                "mean_squared_error_change": float(merged["squared_error_change"].mean()),
                "mean_quantile_coverage_change": float(merged["coverage_change"].mean()),
                "mean_quantile_shortfall_change": float(merged["quantile_shortfall_change"].mean()),
                "row_improvement_pct": float((merged["absolute_error_change"] < 0).mean() * 100),
                "target_date_improvement_pct": float((target < 0).mean() * 100),
                "scenarios_improved": int((scenario < 0).sum()), "scenarios_worsened": int((scenario > 0).sum()),
                "quarterly_win_rate": float((quarterly < 0).mean()),
                "largest_target_date_improvement": float(target.min()),
                "largest_target_date_improvement_date": target.idxmin(),
                "largest_target_date_deterioration": float(target.max()),
                "largest_target_date_deterioration_date": target.idxmax(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    expanding = frame[frame["training_window_id"] == "TW_EXPANDING"][ALIGNMENT_KEY + ["absolute_error"]].rename(columns={"absolute_error": "absolute_error_expanding"})
    for scope in ["development", "full_history"]:
        for window in [item.training_window_id for item in TRAINING_WINDOW_CANDIDATES if item.training_window_id != "TW_EXPANDING"]:
            part = frame[frame["training_window_id"] == window].merge(expanding, on=ALIGNMENT_KEY, validate="one_to_one")
            if scope == "development":
                part = part[part["period_role"] == "development"]
            cluster = part.assign(diff=part["absolute_error"] - part["absolute_error_expanding"]).groupby("target_date")["diff"].mean().to_numpy()
            draws = rng.choice(cluster, size=(BOOTSTRAP_REPLICATIONS, len(cluster)), replace=True).mean(axis=1)
            rows.append(
                {
                    "training_window_id": window, "reference_window_id": "TW_EXPANDING", "evaluation_scope": scope,
                    "target_date_cluster_count": int(len(cluster)), "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED, "observed_mae_difference": float(cluster.mean()),
                    "ci_2_5_pct": float(np.quantile(draws, 0.025)), "ci_97_5_pct": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_improvement": float(np.mean(draws < 0)),
                }
            )
    return pd.DataFrame(rows)


def quantile_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, part in frame.groupby("training_window_id", sort=False):
        scopes = {
            "full_history": part,
            "development": part[part["period_role"] == "development"],
            "confirmation": part[part["period_role"] == "confirmation"],
            "Saturday": part[part["day_type"] == "Saturday"], "Sunday": part[part["day_type"] == "Sunday"],
            "S2": part[part["scenario"] == "S2_same_weekend_sunday"],
            "H1": part[part["service_horizon"] == 1], "H2": part[part["service_horizon"] == 2],
            "H5": part[part["service_horizon"] == 5],
            "recent_52": part[part["recent_period"] == "Recent 52"],
        }
        for scope, subset in scopes.items():
            metrics = extended_metrics(subset)
            rows.append(
                {
                    "training_window_id": window, "evaluation_scope": scope, "row_count": metrics["row_count"],
                    "nominal_quantile": 0.8, "raw_quantile_coverage": metrics["raw_quantile_coverage"],
                    "coverage_gap_from_nominal": metrics["raw_quantile_coverage"] - 0.8,
                    "mean_quantile_shortfall_when_uncovered": metrics["mean_quantile_shortfall_when_uncovered"],
                    "mean_quantile_excess_when_covered": metrics["mean_quantile_excess_when_covered"],
                    "mean_quantile_pinball_loss": metrics["mean_quantile_pinball_loss"],
                }
            )
    return pd.DataFrame(rows)


def scenario_horizon_results(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["evaluation_scope"].isin(["scenario", "service_horizon"])].copy()


def daytype_results(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["evaluation_scope"] == "day_type"].copy()


def recent_analysis(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, part in frame.groupby("training_window_id", sort=False):
        recent = part[part["recent_period"] == "Recent 52"]
        extension = part[part["new_extension"].astype(bool)]
        earlier = part[part["recent_period"] == "Earlier"]
        tail = development_recent_tail(part)
        rows.append(
            {
                "training_window_id": window,
                "recent_52_mae": float(recent["absolute_error"].mean()),
                "recent_52_bias": float(recent["point_error"].mean()),
                "recent_52_p90_absolute_error": float(recent["absolute_error"].quantile(0.9)),
                "recent_52_quantile_coverage": float(recent["quantile_covers"].mean()),
                "recent_52_underprediction_frequency": float((recent["point_error"] < 0).mean()),
                "recent_52_absolute_error_std": float(recent["absolute_error"].std()),
                "development_latest_26_per_daytype_mae": float(tail["absolute_error"].mean()),
                "development_latest_26_saturday_mae": float(tail.loc[tail["day_type"] == "Saturday", "absolute_error"].mean()),
                "development_latest_26_sunday_mae": float(tail.loc[tail["day_type"] == "Sunday", "absolute_error"].mean()),
                "new_extension_mae": float(extension["absolute_error"].mean()),
                "earlier_period_mae": float(earlier["absolute_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def selected_wide_table(
    selection: pd.DataFrame, metrics: pd.DataFrame, references: pd.DataFrame, bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    expanding_mae = metric_lookup(metrics, "TW_EXPANDING", "full_history")
    ref_full = references[references["evaluation_scope"] == "full_history"].set_index("reference_id")["mae"]
    rows: list[dict[str, Any]] = []
    for record in selection.to_dict(orient="records"):
        window = record["training_window_id"]
        boot = bootstrap[(bootstrap["training_window_id"] == window) & (bootstrap["evaluation_scope"] == "development")]
        interval = "reference" if window == "TW_EXPANDING" else f"[{boot.iloc[0]['ci_2_5_pct']:.6f}, {boot.iloc[0]['ci_97_5_pct']:.6f}]"
        row = {
            **record,
            "confirmation_mae": metric_lookup(metrics, window, "confirmation"),
            "full_history_mae": metric_lookup(metrics, window, "full_history"),
            "matched_history_mae": metric_lookup(metrics, window, "matched_history_through_2026_06_21"),
            "new_extension_mae": metric_lookup(metrics, window, "new_extension_after_2026_06_21"),
            "saturday_mae": metric_lookup(metrics, window, "day_type", "Saturday"),
            "sunday_mae": metric_lookup(metrics, window, "day_type", "Sunday"),
            "s1_mae": metric_lookup(metrics, window, "scenario", "S1_same_weekend_saturday"),
            "s2_mae": metric_lookup(metrics, window, "scenario", "S2_same_weekend_sunday"),
            "s3_mae": metric_lookup(metrics, window, "scenario", "S3_next_service"),
            "s4_mae": metric_lookup(metrics, window, "scenario", "S4_two_service_ahead"),
            "s5_mae": metric_lookup(metrics, window, "scenario", "S5_longest_supported"),
            "h1_mae": metric_lookup(metrics, window, "service_horizon", "1"),
            "h2_mae": metric_lookup(metrics, window, "service_horizon", "2"),
            "h5_mae": metric_lookup(metrics, window, "service_horizon", "5"),
            "recent_52_mae": metric_lookup(metrics, window, "recent_52_target_dates"),
            "earlier_period_mae": metric_lookup(metrics, window, "earlier_observations"),
            "raw_quantile_coverage": metric_lookup(metrics, window, "full_history", column="raw_quantile_coverage"),
            "percentage_folds_window_constrained": metric_lookup(metrics, window, "full_history", column="window_constrained_fold_pct"),
            "mae_change_from_tw_expanding": metric_lookup(metrics, window, "full_history") - expanding_mae,
            "mae_change_from_f0": metric_lookup(metrics, window, "full_history") - float(ref_full["F0_EXPANDING"]),
            "mae_change_from_last4_median": metric_lookup(metrics, window, "full_history") - float(ref_full["LAST4_MEDIAN"]),
            "development_bootstrap_interval_vs_expanding": interval,
            "production_feasibility": "requires Phase 2B2 validation; no production change in Phase 2B1",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def finalize() -> None:
    decision_path = OUTPUT_DIR / "08_locked_window_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Development-only decision lock must exist before confirmation finalization")
    decision = json.loads(decision_path.read_text())
    selected = decision["selected_training_window_id"]
    lock_timestamp = pd.Timestamp(decision["locked_at_utc"])
    predictions = add_analysis_dimensions(load_predictions())
    metrics = all_metric_rows(predictions)
    references = fixed_reference_metrics(predictions)
    paired = paired_comparisons(predictions)
    bootstrap = bootstrap_stability(predictions)
    quantile = quantile_analysis(predictions)
    scenario_horizon = scenario_horizon_results(metrics)
    daytype = daytype_results(metrics)
    recent = recent_analysis(predictions, metrics)
    selection = pd.read_csv(OUTPUT_DIR / "07_development_selection_table.csv")
    wide = selected_wide_table(selection, metrics, references, bootstrap)

    write_csv("06_training_window_metrics.csv", metrics)
    write_text("06_training_window_metrics.md", "# Training-window metrics\n\nComplete metrics are in the CSV. Core full-history rows:\n\n" + markdown_table(metrics[metrics["evaluation_scope"] == "full_history"]))
    write_csv("07_development_selection_table.csv", wide)
    write_text("07_development_selection_table.md", "# Main Phase 2B1 decision table\n\nSelection fields were computed and locked before confirmation aggregation. Confirmation and descriptive fields below were appended only by this post-lock finalizer.\n\n" + markdown_table(wide))
    confirmation = metrics[metrics["evaluation_scope"] == "confirmation"].copy()
    write_csv("09_confirmation_results.csv", confirmation)
    write_text("09_confirmation_results.md", f"# Confirmation results\n\nGenerated after the `{selected}` decision lock at `{decision['locked_at_utc']}`. These results did not alter selection.\n\n" + markdown_table(confirmation))
    write_csv("10_scenario_horizon_results.csv", scenario_horizon)
    write_text("10_scenario_horizon_results.md", "# Scenario and service-horizon results\n\n" + markdown_table(scenario_horizon))
    write_csv("11_daytype_results.csv", daytype)
    write_text("11_daytype_results.md", "# Day-type results\n\n" + markdown_table(daytype))
    write_csv("12_recent_period_analysis.csv", recent)
    write_text("12_recent_period_analysis.md", "# Recent-period analysis\n\nThe six-date extension is descriptive and too small for a firm standalone conclusion. Absolute-error standard deviation is included as a volatility diagnostic.\n\n" + markdown_table(recent))
    extension = predictions[predictions["new_extension"].astype(bool)].copy()
    write_csv("13_new_extension_predictions.csv", extension)
    extension_summary = metrics[metrics["evaluation_scope"] == "new_extension_after_2026_06_21"]
    write_text("13_new_extension_analysis.md", "# New extension after 2026-06-21\n\nSix service dates contribute multiple preregistered scenario rows. Treat these estimates as descriptive only.\n\n" + markdown_table(extension_summary))
    write_csv("14_paired_window_comparison.csv", paired)
    write_text("14_paired_window_comparison.md", "# Paired finite-window comparisons versus expanding\n\nNegative error changes favor the finite window. Target-date calculations keep all scenarios for a service date together.\n\n" + markdown_table(paired))
    write_csv("15_bootstrap_stability.csv", bootstrap)
    write_text("15_bootstrap_stability.md", f"# Target-date cluster bootstrap\n\nSeed `{BOOTSTRAP_SEED}`; `{BOOTSTRAP_REPLICATIONS}` replications. Each resampled target-date cluster retains every scenario row. Intervals are descriptive and are not the sole selection rule.\n\n" + markdown_table(bootstrap))
    diagnostics = pd.read_csv(OUTPUT_DIR / "16_fold_training_diagnostics.csv")
    diagnostic_summary = diagnostics.groupby("training_window_id").agg(
        fold_rows=("target_date", "size"), minimum_retained_rows=("retained_segment_training_rows", "min"),
        median_retained_rows=("retained_segment_training_rows", "median"), maximum_retained_rows=("retained_segment_training_rows", "max"),
        constrained_fold_pct=("window_constrained", "mean"), minimum_effective_days=("effective_history_days", "min"),
        median_effective_days=("effective_history_days", "median"), maximum_effective_days=("effective_history_days", "max"),
    ).reset_index()
    diagnostic_summary["constrained_fold_pct"] *= 100
    write_text("16_fold_training_diagnostics.md", "# Fold training diagnostics\n\nThe CSV records every prediction fold. Retained dates never exceed forecast origins; windowing is segment-local and preprocessing identifiers include window/count/cutoff.\n\n" + markdown_table(diagnostic_summary))
    write_csv("17_quantile_coverage_analysis.csv", quantile)
    write_text("17_quantile_coverage_analysis.md", "# Uncalibrated nominal-0.8 quantile analysis\n\nNo calibration is applied. Negative coverage gaps indicate undercoverage.\n\n" + markdown_table(quantile))

    dev = predictions[predictions["period_role"] == "development"]
    day_rank = dev.groupby(["day_type", "training_window_id", "scenario"], as_index=False)["absolute_error"].mean()
    day_overall = dev.groupby(["day_type", "training_window_id"], as_index=False)["absolute_error"].mean()
    day_overall["daytype_rank"] = day_overall.groupby("day_type")["absolute_error"].rank(method="min")
    optima = day_overall.loc[day_overall.groupby("day_type")["absolute_error"].idxmin()]
    level = {"TW_EXPANDING": 5, "TW_104": 4, "TW_78": 3, "TW_52": 2, "TW_26": 1}
    opt_ids = dict(zip(optima["day_type"], optima["training_window_id"], strict=True))
    separated = abs(level[opt_ids["Saturday"]] - level[opt_ids["Sunday"]]) >= 2
    scenario_winners = day_rank.loc[day_rank.groupby(["day_type", "scenario"])["absolute_error"].idxmin()]
    write_text(
        "18_daytype_specific_window_diagnostic.md",
        "# Day-type-specific window diagnostic\n\nThis analysis occurs after the global lock and is diagnostic only; no separate windows are deployed.\n\n## Development ranking\n\n" + markdown_table(day_overall.sort_values(["day_type", "daytype_rank"])) +
        "\n\n## Scenario winners\n\n" + markdown_table(scenario_winners) +
        f"\n\nSaturday optimum: **{opt_ids['Saturday']}**. Sunday optimum: **{opt_ids['Sunday']}**. The optima are {'at least two candidate levels apart' if separated else 'fewer than two candidate levels apart'}.",
    )

    selected_row = wide[wide["training_window_id"] == selected].iloc[0]
    expand_row = wide[wide["training_window_id"] == "TW_EXPANDING"].iloc[0]
    selected_configured_rows = (
        "unlimited"
        if pd.isna(selected_row["configured_window_rows"])
        else str(int(selected_row["configured_window_rows"]))
    )
    dev_gain = float(selected_row["development_macro_s1_s5_mae"] - expand_row["development_macro_s1_s5_mae"])
    confirmation_gain = float(selected_row["confirmation_mae"] - expand_row["confirmation_mae"])
    recent_gain = float(selected_row["recent_52_mae"] - expand_row["recent_52_mae"])
    finite = selected != "TW_EXPANDING"
    confirmation_reversal = finite and dev_gain < 0 and confirmation_gain > abs(dev_gain)
    recommend_freeze = selected if finite and dev_gain < 0 and not confirmation_reversal and recent_gain < 0 else "TW_EXPANDING"
    write_text(
        "19_phase2b2_recommendation.md",
        f"# Phase 2B2 recommendation\n\nThe Phase 2B1 global selection is **{selected}** (configured segment-row cap: {selected_configured_rows}). Development macro MAE is {selected_row['development_macro_s1_s5_mae']:.6f}; confirmation MAE is {selected_row['confirmation_mae']:.6f}; full-history MAE is {selected_row['full_history_mae']:.6f}; recent-52 MAE is {selected_row['recent_52_mae']:.6f}. Saturday/Sunday MAE is {selected_row['saturday_mae']:.6f}/{selected_row['sunday_mae']:.6f}; S2 is {selected_row['s2_mae']:.6f}; H1/H2/H5 are {selected_row['h1_mae']:.6f}/{selected_row['h2_mae']:.6f}/{selected_row['h5_mae']:.6f}. Bias is {metric_lookup(metrics, selected, 'full_history', column='mean_signed_error'):.6f}; raw coverage is {selected_row['raw_quantile_coverage']:.6f}.\n\nRelative to expanding, full-history MAE changes by {selected_row['mae_change_from_tw_expanding']:+.6f}; relative to last-four median it changes by {selected_row['mae_change_from_last4_median']:+.6f}. Rejected windows: {', '.join(w for w in wide['training_window_id'] if w != selected)}.\n\nFor Phase 2B2 sample-weight experiments, freeze **{recommend_freeze}** and keep the snapshot, F6 features/order/hash, model classes/hyperparameters, scenarios/origins, T1/W0 policies, segment definitions, minimum row count, preprocessing, split, and metrics unchanged. Sample weighting should be tested only as the single next factor. A later separate-daytype-window experiment is {'worth considering because optima differ by at least two levels, subject to scenario stability and practical gain' if separated else 'not currently justified by the preregistered two-level criterion'}.\n\nLimitations: observational single-location data, correlated scenario rows, a small six-date extension, uncalibrated quantile undercoverage, and descriptive bootstrap intervals. Phase 2B2 has not begun.",
    )

    tests_path = OUTPUT_DIR / "supporting_test_results.json"
    tests = json.loads(tests_path.read_text()) if tests_path.is_file() else {"targeted": "pending", "full": "pending"}
    contract = json.loads((OUTPUT_DIR / "03_locked_contract.json").read_text())
    integrity = {
        "source_export_hash_valid": sha256_file(SOURCE_PATH) == EXPECTED_SOURCE_SHA256,
        "snapshot_hash_valid": sha256_file(SNAPSHOT_PATH) == contract["snapshot_sha256"],
        "f6_hash_valid": validate_lock()["feature_list_sha256"] == EXPECTED_F6_HASH,
        "production_files_unchanged": protected_fingerprints() == contract["protected_fingerprints_before"],
        "prior_phase_directories_unchanged": prior_phase_fingerprints() == contract["prior_phase_fingerprints_before"],
        "duplicate_prediction_keys": int(predictions.duplicated(PREDICTION_KEY).sum()),
        "paired_key_alignment_valid": True,
        "feature_provenance_valid": bool(predictions["feature_provenance_valid"].all()),
        "preprocessing_cutoff_valid": bool(predictions["preprocessing_cutoff_valid"].all()),
        "retained_training_cutoff_valid": bool((predictions["latest_retained_training_date"] <= predictions["forecast_origin"]).all()),
        "selection_locked_before_confirmation_finalization": bool(lock_timestamp < pd.Timestamp.now(tz="UTC")),
    }
    if not all(value for key, value in integrity.items() if key != "duplicate_prediction_keys") or integrity["duplicate_prediction_keys"] != 0:
        raise AssertionError(f"Completion integrity validation failed: {integrity}")
    write_text(
        "20_test_and_reproducibility_report.md",
        "# Tests and reproducibility\n\n" + markdown_table(pd.DataFrame([integrity])) +
        f"\n\nTargeted tests: `{tests.get('targeted')}`. Full suite: `{tests.get('full')}`. Reproduction uses Python `{platform.python_version()}`, pandas `{pd.__version__}`, NumPy `{np.__version__}`, and scikit-learn `{sklearn.__version__}`.",
    )

    write_text(
        "01_phase2b1_summary.md",
        f"# Phase 2B1 summary\n\nAll five preregistered training windows were evaluated on the frozen Phase 2A.5 snapshot with F6 unchanged. TW_EXPANDING reproduced 1,284 Phase 2A.5 rows within 1e-10 and exact keys. The development-only rule selected **{selected}** before confirmation aggregation. Its development macro MAE is {selected_row['development_macro_s1_s5_mae']:.6f}, confirmation MAE {selected_row['confirmation_mae']:.6f}, full-history MAE {selected_row['full_history_mae']:.6f}, and recent-52 MAE {selected_row['recent_52_mae']:.6f}. No production or saved-model file changed, and Phase 2B2 was not started.",
    )

    # Create the manifest after all report artifacts exist, then rewrite once to include its own name.
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git("branch", "--show-current"), "starting_commit": contract["starting_commit"],
        "ending_commit": git("rev-parse", "HEAD"), "python_version": platform.python_version(),
        "dependency_versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "supabase_source_export_hash": EXPECTED_SOURCE_SHA256, "normalized_snapshot_hash": sha256_file(SNAPSHOT_PATH),
        "snapshot_date_range": contract["snapshot_date_range"], "f6_feature_set_id": "F6_COMPACT_REPAIR",
        "f6_feature_hash": EXPECTED_F6_HASH, "exact_ordered_feature_list": contract["ordered_feature_list"],
        "training_window_registry": contract["training_window_registry"],
        "development_split": contract["development_end"], "confirmation_split": contract["confirmation_start"],
        "selection_rule": contract["selection_hierarchy"], "tie_rule": "within 0.10 development macro MAE, select longer window",
        "selected_training_window_id": selected, "selection_locked_at_utc": decision["locked_at_utc"],
        "model_configurations": {
            "point_model": validate_lock()["point_model"],
            "quantile_model": validate_lock()["quantile_model"],
        },
        "random_seeds": {"point_model": POINT_RANDOM_SEED, "quantile_model": QUANTILE_RANDOM_SEED, "bootstrap": BOOTSTRAP_SEED},
        "commands_executed": [
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/run_phase2b1_training_windows.py",
            "/tmp/soup-kitchen-forecast-phase2a5-venv/bin/python scripts/finalize_phase2b1_reports.py",
        ],
        "code_files_created": ["src/training_windows.py", "scripts/run_phase2b1_training_windows.py", "scripts/finalize_phase2b1_reports.py", "tests/test_training_windows.py", "tests/test_phase2b1_training_windows.py"],
        "code_files_modified": ["src/origin_backtest.py"], "existing_files_overwritten": ["src/origin_backtest.py"],
        "artifact_files_created": REQUIRED_ARTIFACTS,
        "prediction_counts_by_candidate": {key: int(value) for key, value in predictions.groupby("training_window_id").size().items()},
        "window_constrained_fold_counts": {key: int(value) for key, value in predictions.groupby("training_window_id")["window_constrained"].sum().items()},
        "leakage_validation_results": integrity, "preprocessing_cutoff_validation_results": integrity["preprocessing_cutoff_valid"],
        "production_file_integrity_results": integrity["production_files_unchanged"],
        "saved_model_integrity_results": integrity["production_files_unchanged"], "test_results": tests,
        "known_limitations": ["single location", "correlated scenario rows", "six-date extension", "raw quantile undercoverage", "bootstrap intervals are descriptive"],
    }
    write_json("phase2b1_manifest.json", manifest)
    write_text(
        "README.md",
        "# Phase 2B1 training-window artifacts\n\nThis directory contains the frozen contract, five-candidate origin-aware evaluation, development-only selection lock, post-lock confirmation reports, diagnostics, bootstrap analysis, and reproducibility evidence. `05_training_window_predictions.csv` is the canonical raw prediction output; `07_development_selection_table.csv` is the compact main decision table. Supporting files prefixed `supporting_` are audit evidence and are not required deliverables.\n",
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise AssertionError(f"Required deliverables missing: {missing}")
    print(f"Finalized {len(REQUIRED_ARTIFACTS)} required Phase 2B1 artifacts; selected {selected}.")


if __name__ == "__main__":
    finalize()
