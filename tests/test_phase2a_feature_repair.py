from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_phase2a_feature_repair import (
    EXTRA_ARTIFACTS,
    FEATURE_DEFINITIONS,
    REQUIRED_ARTIFACTS,
    SCENARIO_SHORT,
    baseline_metrics,
    development_decision_metrics,
    fixed_split,
    main_decision_table,
    markdown_table,
    metric_breakdowns,
    moving_block_bootstrap,
    paired_comparisons,
    phase1_gate,
)
from src.feature_sets import (
    FEATURE_SET_IDS,
    F0,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    build_feature_set_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_gate_reproduces_all_published_references() -> None:
    gate, preferred = phase1_gate()
    assert gate["preferred_rows"] == len(preferred) == 1256
    assert gate["preferred_mae"] == pytest.approx(16.277079645681486, abs=1e-12)
    assert gate["median_mae"] == pytest.approx(14.835987, abs=5e-7)
    assert gate["legacy_rows"] == 317


def test_fixed_split_reserves_52_targets_per_daytype() -> None:
    _, preferred = phase1_gate()
    role, summary = fixed_split(preferred)
    assert (summary["confirmation_target_count"] == 52).all()
    assert sum(value == "confirmation" for value in role.values()) == 104
    assert summary.set_index("day_type").loc["Saturday", "confirmation_start"] == pd.Timestamp(
        "2025-05-31"
    )
    assert summary.set_index("day_type").loc["Sunday", "confirmation_start"] == pd.Timestamp(
        "2025-05-25"
    )


def test_moving_block_bootstrap_is_deterministic_and_ordered() -> None:
    series = pd.Series(
        range(20), index=pd.date_range("2024-01-01", periods=20, freq="D"), dtype=float
    )
    first = moving_block_bootstrap(series, 20260715)
    second = moving_block_bootstrap(series, 20260715)
    assert first == second
    assert first[0] <= first[1]


def test_required_artifact_names_are_unique_and_include_design_manifest_readme() -> None:
    assert len(REQUIRED_ARTIFACTS) == len(set(REQUIRED_ARTIFACTS)) == 31
    assert REQUIRED_ARTIFACTS[0] == "00_implementation_design.md"
    assert "phase2a_manifest.json" in REQUIRED_ARTIFACTS
    assert "README.md" in REQUIRED_ARTIFACTS
    assert not set(REQUIRED_ARTIFACTS).intersection(EXTRA_ARTIFACTS)


def test_design_exists_before_runner_is_allowed() -> None:
    design = (
        ROOT
        / "artifacts/ny_12550/model_optimization/phase2a_feature_repair/00_implementation_design.md"
    )
    assert design.exists()
    text = design.read_text()
    assert "experiment embargo" in text
    assert "2025-05-31" in text
    assert "F6_COMPACT_SELECTED" in text


def test_feature_definition_catalog_covers_registry() -> None:
    registry = build_feature_set_registry()
    for definition in registry.values():
        assert set(definition.feature_list).issubset(FEATURE_DEFINITIONS)


def test_registry_feature_counts_increase_through_controlled_candidates() -> None:
    registry = build_feature_set_registry(f5_parent_id=F4)
    counts = [registry[item].feature_count for item in [F1, F2, F3, F4]]
    assert counts == sorted(counts)
    assert registry[F0].feature_count == 26
    assert registry[F5].feature_count == registry[F4].feature_count + 11
    assert registry[F6].feature_count >= len(registry[F1].feature_list)


def test_paired_aggregation_preserves_existing_daytype_and_period_columns() -> None:
    _, preferred = phase1_gate()
    role, _ = fixed_split(preferred)
    base = preferred.copy()
    base["target_date"] = pd.to_datetime(base["target_date"]).dt.normalize()
    base["forecast_origin"] = pd.to_datetime(base["forecast_origin"]).dt.normalize()
    base["period_role"] = base["target_date"].map(role)
    base["recent_period"] = base["period"]
    candidates = []
    for feature_set_id in FEATURE_SET_IDS:
        candidate = base.copy()
        candidate["feature_set_id"] = feature_set_id
        candidates.append(candidate)
    paired, by_target = paired_comparisons(pd.concat(candidates, ignore_index=True))
    assert len(paired) == 7
    assert len(by_target) == 7 * base["target_date"].nunique()
    assert paired["mean_absolute_error_change_vs_f0"].eq(0).all()
    assert paired["saturday_mae_change"].eq(0).all()
    assert paired["recent_period_mae_change"].eq(0).all()


def test_main_decision_aggregation_runs_on_seven_fully_aligned_candidates() -> None:
    _, preferred = phase1_gate()
    role, _ = fixed_split(preferred)
    base = preferred.copy()
    base["target_date"] = pd.to_datetime(base["target_date"]).dt.normalize()
    base["forecast_origin"] = pd.to_datetime(base["forecast_origin"]).dt.normalize()
    base["period_role"] = base["target_date"].map(role)
    base["recent_period"] = base["period"]
    base["scenario_short"] = base["scenario"].map(SCENARIO_SHORT)
    base["year"] = base["target_date"].dt.year.astype(str)
    base["quarter"] = base["target_date"].dt.to_period("Q").astype(str)
    base["attendance_quartile"] = "test quartile"
    base["history_depth_group"] = "Sufficient history (>=6)"
    base["attendance_feature_missing_count"] = 0
    registry = build_feature_set_registry()
    candidates = []
    development = []
    for feature_set_id in FEATURE_SET_IDS:
        candidate = base.copy()
        candidate["feature_set_id"] = feature_set_id
        candidate["feature_count"] = registry[feature_set_id].feature_count
        candidates.append(candidate)
        development.append(development_decision_metrics(candidate))
    primary = pd.concat(candidates, ignore_index=True)
    metrics = metric_breakdowns(primary)
    paired, _ = paired_comparisons(primary)
    baselines = baseline_metrics(primary)
    decision = main_decision_table(
        metrics,
        paired,
        pd.DataFrame(development),
        baselines,
        registry,
        F6,
    )
    assert len(decision) == 7
    assert decision.loc[decision["Feature set ID"] == F0, "Full-history MAE"].iloc[0] == pytest.approx(
        16.277079645681486
    )


def test_markdown_table_serializes_list_cells() -> None:
    rendered = markdown_table(pd.DataFrame([{"group": "lags", "features": ["a", "b"]}]))
    assert '["a", "b"]' in rendered
