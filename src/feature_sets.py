"""Deterministic Phase 2A feature-set registry and selection helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd

from src.origin_features import (
    CALENDAR_FEATURES,
    DAYTYPE_SLOT_FEATURES,
    DAYTYPE_SUMMARY_FEATURES,
    HORIZON_AWARE_FEATURES,
    LAST_OBSERVED_DAYTYPE_FEATURES,
    MODEL_FEATURES,
    OriginFeatureResult,
    build_repaired_features_as_of,
)


F0 = "F0_CURRENT_ORIGIN"
F1 = "F1_CALENDAR_ONLY"
F2 = "F2_LAST_OBSERVED_DAYTYPE"
F3 = "F3_DAYTYPE_ROBUST_SUMMARIES"
F4 = "F4_CORRECTED_SLOT_HISTORY"
F5 = "F5_HORIZON_AWARE"
F6 = "F6_COMPACT_SELECTED"
FEATURE_SET_IDS = (F0, F1, F2, F3, F4, F5, F6)

COMPACT_LAST_OBSERVED = [
    "last_observed_daytype_1",
    "last_observed_daytype_2",
    "last_observed_daytype_4",
    "last_observed_daytype_6",
]
COMPACT_SUMMARIES = [
    "daytype_mean_last_2",
    "daytype_median_last_4",
    "daytype_median_last_6",
    "daytype_std_last_4",
    "daytype_recent_vs_previous_3",
    "daytype_mean2_minus_previous2",
]
COMPACT_SLOT = [
    "daytype_slot_last_observed",
    "daytype_slot_match_count",
    "daytype_slot_days_since_latest",
]
COMPACT_HORIZON_BASE = [
    "calendar_days_ahead",
    "service_horizon",
    "observed_daytype_count",
    "future_eligible_services_between",
    "days_since_last_observed_daytype",
]


@dataclass(frozen=True)
class FeatureSetDefinition:
    feature_set_id: str
    name: str
    feature_list: tuple[str, ...]
    feature_groups: tuple[str, ...]
    parent_feature_set: str | None
    controlled_change: str
    expected_value: str
    leakage_assessment: str
    production_availability: str

    @property
    def feature_count(self) -> int:
        return len(self.feature_list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["feature_list"] = list(self.feature_list)
        out["feature_groups"] = list(self.feature_groups)
        out["feature_count"] = self.feature_count
        return out


def _definition(
    feature_set_id: str,
    name: str,
    features: list[str],
    groups: list[str],
    parent: str | None,
    change: str,
    expected: str,
) -> FeatureSetDefinition:
    if len(features) != len(set(features)):
        raise ValueError(f"{feature_set_id} contains duplicate features")
    return FeatureSetDefinition(
        feature_set_id=feature_set_id,
        name=name,
        feature_list=tuple(features),
        feature_groups=tuple(groups),
        parent_feature_set=parent,
        controlled_change=change,
        expected_value=expected,
        leakage_assessment=(
            "Origin-valid by construction; attendance sources are restricted to "
            "matching records on or before each forecast origin."
            if feature_set_id != F0
            else "Exact Phase 1 semantics, including explicit missing values when "
            "target-relative conceptual sources occur after the origin."
        ),
        production_availability=(
            "Available from target calendar, stored attendance through the live "
            "origin, and deterministic target/origin horizon metadata."
            if feature_set_id not in {F0}
            else "Current production contract; W0 weather is intentionally absent."
        ),
    )


def build_feature_set_registry(
    *,
    f5_parent_id: str = F4,
    f6_features: list[str] | tuple[str, ...] | None = None,
    f6_groups: list[str] | tuple[str, ...] | None = None,
) -> OrderedDict[str, FeatureSetDefinition]:
    """Build the registry; F5 parent and locked F6 are explicit inputs."""

    if f5_parent_id not in {F2, F3, F4}:
        raise ValueError("F5 parent must be F2, F3, or F4")
    f1_features = list(CALENDAR_FEATURES)
    f2_features = f1_features + list(LAST_OBSERVED_DAYTYPE_FEATURES)
    f3_features = f2_features + list(DAYTYPE_SUMMARY_FEATURES)
    f4_features = f3_features + list(DAYTYPE_SLOT_FEATURES)
    parents = {F2: f2_features, F3: f3_features, F4: f4_features}
    f5_features = parents[f5_parent_id] + list(HORIZON_AWARE_FEATURES)
    realized_f6 = list(f6_features) if f6_features is not None else f1_features
    realized_f6_groups = list(f6_groups) if f6_groups is not None else ["calendar"]

    registry = OrderedDict(
        [
            (
                F0,
                _definition(
                    F0,
                    "Current origin-aware feature contract",
                    list(MODEL_FEATURES),
                    ["calendar", "target_relative_attendance", "weather"],
                    None,
                    "Exact Phase 1 reproduction.",
                    "Provides paired reference predictions and the reproduction gate.",
                ),
            ),
            (
                F1,
                _definition(
                    F1,
                    "Calendar only",
                    f1_features,
                    ["calendar"],
                    F0,
                    "Removes all attendance-history and weather fields.",
                    "Measures whether unstable current history fields add value.",
                ),
            ),
            (
                F2,
                _definition(
                    F2,
                    "Last observed matching day type",
                    f2_features,
                    ["calendar", "last_observed_daytype"],
                    F1,
                    "Adds ranks 1, 2, 3, 4, and 6 from origin-observed matching day types.",
                    "Keeps lag meaning invariant across forecast horizons.",
                ),
            ),
            (
                F3,
                _definition(
                    F3,
                    "Matching-day-type robust summaries",
                    f3_features,
                    ["calendar", "last_observed_daytype", "daytype_summaries"],
                    F2,
                    "Adds 12 fixed-window location, spread, extrema, and contrast summaries.",
                    "Tests robust recent attendance location and change signals.",
                ),
            ),
            (
                F4,
                _definition(
                    F4,
                    "Corrected day-type and monthly-slot history",
                    f4_features,
                    [
                        "calendar",
                        "last_observed_daytype",
                        "daytype_summaries",
                        "daytype_slot",
                    ],
                    F3,
                    "Adds five fields matching both target day type and month slot.",
                    "Tests a corrected form of sparse seasonal slot history.",
                ),
            ),
            (
                F5,
                _definition(
                    F5,
                    "Deployment-safe horizon and availability context",
                    f5_features,
                    list(build_feature_set_registry_groups(f5_parent_id)) + ["horizon_availability"],
                    f5_parent_id,
                    "Adds deterministic horizon, history-depth, and missingness fields.",
                    "Allows unchanged models to distinguish sparse and longer-horizon inputs.",
                ),
            ),
            (
                F6,
                _definition(
                    F6,
                    "Compact development-selected feature set",
                    realized_f6,
                    realized_f6_groups,
                    F5,
                    "Retains only preregistered compact representatives of groups supported on development rows.",
                    "Reduces redundancy while preserving demonstrated origin-valid signals.",
                ),
            ),
        ]
    )
    if tuple(registry) != FEATURE_SET_IDS:
        raise AssertionError("Feature-set IDs are not stable")
    return registry


def build_feature_set_registry_groups(feature_set_id: str) -> tuple[str, ...]:
    mapping = {
        F1: ("calendar",),
        F2: ("calendar", "last_observed_daytype"),
        F3: ("calendar", "last_observed_daytype", "daytype_summaries"),
        F4: (
            "calendar",
            "last_observed_daytype",
            "daytype_summaries",
            "daytype_slot",
        ),
    }
    return mapping[feature_set_id]


def make_repaired_feature_builder(
    definition: FeatureSetDefinition,
) -> Callable[..., OriginFeatureResult]:
    """Bind a registry entry to the centralized repaired builder."""

    if definition.feature_set_id == F0:
        raise ValueError("F0 must use the exact Phase 1 feature builder")

    def builder(
        attendance_history: pd.DataFrame,
        target_date: Any,
        forecast_origin: Any,
        *,
        weather_policy: str,
        weather_df: pd.DataFrame | None,
        weekday_policy: str,
        feature_cols: list[str],
        calendar_days_ahead: int,
        service_horizon: int,
    ) -> OriginFeatureResult:
        del weather_policy, weather_df
        return build_repaired_features_as_of(
            attendance_history,
            target_date,
            forecast_origin,
            calendar_days_ahead=calendar_days_ahead,
            service_horizon=service_horizon,
            weekday_policy=weekday_policy,
            feature_cols=feature_cols,
        )

    return builder


def select_f5_parent(development_metrics: pd.DataFrame) -> str:
    """Select F5's parent by the preregistered development-only ranking."""

    required = {"feature_set_id", "development_macro_mae", "development_s2_mae", "feature_count"}
    missing = required.difference(development_metrics.columns)
    if missing:
        raise ValueError(f"Development metrics missing columns: {sorted(missing)}")
    candidates = development_metrics[
        development_metrics["feature_set_id"].isin([F2, F3, F4])
    ].copy()
    if len(candidates) != 3:
        raise ValueError("F5 parent selection requires exactly F2, F3, and F4")
    ranked = candidates.sort_values(
        ["development_macro_mae", "development_s2_mae", "feature_count", "feature_set_id"],
        kind="stable",
    )
    return str(ranked.iloc[0]["feature_set_id"])


def _group_supported(
    metrics: pd.DataFrame,
    child: str,
    parent: str,
) -> tuple[bool, dict[str, Any]]:
    indexed = metrics.set_index("feature_set_id")
    child_row = indexed.loc[child]
    parent_row = indexed.loc[parent]
    macro_gain = float(parent_row["development_macro_mae"] - child_row["development_macro_mae"])
    s2_change = float(child_row["development_s2_mae"] - parent_row["development_s2_mae"])
    supported = macro_gain >= 0.05 and s2_change <= 0.25
    return supported, {
        "child": child,
        "parent": parent,
        "macro_mae_gain": macro_gain,
        "s2_mae_change": s2_change,
        "threshold_macro_gain": 0.05,
        "maximum_s2_deterioration": 0.25,
        "supported": bool(supported),
    }


def select_compact_f6(
    development_metrics: pd.DataFrame,
    *,
    f5_parent_id: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Apply the preregistered group-level compacting rule using development only."""

    required_ids = {F1, F2, F3, F4, F5}
    if not required_ids.issubset(set(development_metrics["feature_set_id"])):
        raise ValueError("F6 selection requires development metrics for F1-F5")

    checks = [
        ("last_observed_daytype", F2, F1, COMPACT_LAST_OBSERVED),
        ("daytype_summaries", F3, F2, COMPACT_SUMMARIES),
        ("daytype_slot", F4, F3, COMPACT_SLOT),
        ("horizon_availability", F5, f5_parent_id, COMPACT_HORIZON_BASE),
    ]
    features = list(CALENDAR_FEATURES)
    groups = ["calendar"]
    decisions: list[dict[str, Any]] = []
    retained_lags: list[str] = []
    slot_retained = False
    horizon_supported = False
    for group, child, parent, compact_features in checks:
        supported, evidence = _group_supported(development_metrics, child, parent)
        evidence["group"] = group
        evidence["compact_representatives"] = list(compact_features)
        evidence["reason"] = (
            "controlled development comparison met both preregistered thresholds"
            if supported
            else "controlled development comparison did not meet both preregistered thresholds"
        )
        decisions.append(evidence)
        if not supported:
            continue
        groups.append(group)
        features.extend(compact_features)
        if group == "last_observed_daytype":
            retained_lags = list(compact_features)
        elif group == "daytype_slot":
            slot_retained = True
        elif group == "horizon_availability":
            horizon_supported = True

    if horizon_supported:
        for lag in retained_lags:
            features.append(f"missing_{lag}")
        if slot_retained:
            features.append("daytype_slot_history_missing")

    if len(features) != len(set(features)):
        raise AssertionError("Compact F6 selection produced duplicate features")
    return features, groups, decisions

