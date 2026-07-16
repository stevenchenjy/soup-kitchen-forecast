"""Origin-valid residual calibration independent from model fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationPolicy:
    calibration_policy_id: str
    grouping: str
    maximum_history_dates: int | None
    raw_quantile_reference: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


C0 = CalibrationPolicy(
    "C0_EXISTING_RAW_QUANTILE",
    "raw_reference",
    None,
    True,
    "Existing Phase 2B1 raw quantile prediction; reference only.",
)
C1 = CalibrationPolicy(
    "C1_EXPANDING_DAYTYPE",
    "day_type",
    None,
    False,
    "All eligible earlier residual target dates matching target day type.",
)
C2 = CalibrationPolicy(
    "C2_LAST52_DAYTYPE",
    "day_type",
    52,
    False,
    "Latest 52 eligible earlier residual target dates matching target day type.",
)
C3 = CalibrationPolicy(
    "C3_LAST26_DAYTYPE",
    "day_type",
    26,
    False,
    "Latest 26 eligible earlier residual target dates matching target day type.",
)
C4 = CalibrationPolicy(
    "C4_EXPANDING_POOLED",
    "pooled",
    None,
    False,
    "All eligible earlier residual target dates pooled across both day types.",
)
CALIBRATION_POLICIES: tuple[CalibrationPolicy, ...] = (C0, C1, C2, C3, C4)
CALIBRATION_POLICIES_BY_ID = {
    item.calibration_policy_id: item for item in CALIBRATION_POLICIES
}
PRIMARY_POLICIES: tuple[CalibrationPolicy, ...] = (C1, C2, C3, C4)
CONSERVATISM_RANK = {
    item.calibration_policy_id: rank for rank, item in enumerate(PRIMARY_POLICIES)
}
DECISION_EPSILON = 1e-12


@dataclass(frozen=True)
class AvailableCalibrationHistory:
    pooled_indices: np.ndarray
    daytype_indices: np.ndarray


@dataclass
class CalibrationHistoryCache:
    observations: pd.DataFrame
    histories_by_row_id: dict[int, AvailableCalibrationHistory]
    build_count: int


def resolve_calibration_policy(
    value: CalibrationPolicy | str,
) -> CalibrationPolicy:
    if isinstance(value, CalibrationPolicy):
        canonical = CALIBRATION_POLICIES_BY_ID.get(value.calibration_policy_id)
        if canonical != value:
            raise ValueError(f"Non-canonical calibration policy: {value.calibration_policy_id}")
        return value
    try:
        return CALIBRATION_POLICIES_BY_ID[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported calibration policy: {value}") from exc


def conformal_empirical_quantile(
    residuals: np.ndarray | list[float], target_coverage: float
) -> tuple[float, int]:
    """Return clipped ceil((n+1)q) order statistic with no interpolation."""

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Conformal empirical quantile requires residual observations")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration residuals must be finite and nonnegative")
    coverage = float(target_coverage)
    if not 0 < coverage < 1:
        raise ValueError("target_coverage must be strictly between zero and one")
    sorted_values = np.sort(values, kind="stable")
    one_based = min(max(int(math.ceil((len(values) + 1) * coverage)), 1), len(values))
    return float(sorted_values[one_based - 1]), one_based


def collapse_residual_observations(frozen_predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse scenario residuals to one conservative observation per target date."""

    required = {"target_date", "day_type", "actual", "point_prediction", "scenario"}
    missing = required.difference(frozen_predictions.columns)
    if missing:
        raise ValueError(f"Frozen predictions missing columns: {sorted(missing)}")
    frame = frozen_predictions[list(required)].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise").dt.normalize()
    frame["underprediction_residual"] = np.maximum(
        frame["actual"].astype(float) - frame["point_prediction"].astype(float), 0.0
    )
    for _, part in frame.groupby("target_date", sort=False):
        if part["actual"].nunique() != 1 or part["day_type"].nunique() != 1:
            raise ValueError("Scenario rows disagree on target actual or day type")
    observations = (
        frame.groupby(["target_date", "day_type"], as_index=False, sort=True)
        .agg(
            residual_observation=("underprediction_residual", "max"),
            source_scenario_count=("scenario", "nunique"),
            actual=("actual", "first"),
        )
        .sort_values("target_date", kind="stable")
        .reset_index(drop=True)
    )
    if observations["target_date"].duplicated().any():
        raise AssertionError("A target date contributed more than one calibration observation")
    observations["observation_id"] = np.arange(len(observations), dtype=int)
    return observations


def build_calibration_history_cache(
    frozen_predictions: pd.DataFrame,
    observations: pd.DataFrame | None = None,
) -> CalibrationHistoryCache:
    """Build eligible residual-index histories once for every prediction row."""

    required = {"calibration_row_id", "target_date", "forecast_origin", "day_type"}
    missing = required.difference(frozen_predictions.columns)
    if missing:
        raise ValueError(f"Frozen predictions missing cache columns: {sorted(missing)}")
    if frozen_predictions["calibration_row_id"].duplicated().any():
        raise ValueError("calibration_row_id must be unique")
    obs = (
        collapse_residual_observations(frozen_predictions)
        if observations is None
        else observations.copy()
    )
    obs["target_date"] = pd.to_datetime(obs["target_date"], errors="raise").dt.normalize()
    obs = obs.sort_values("target_date", kind="stable").reset_index(drop=True)
    dates = obs["target_date"].to_numpy(dtype="datetime64[ns]")
    daytypes = obs["day_type"].astype(str).to_numpy()
    histories: dict[int, AvailableCalibrationHistory] = {}
    for row in frozen_predictions.itertuples(index=False):
        row_id = int(row.calibration_row_id)
        target = pd.Timestamp(row.target_date).normalize().to_datetime64()
        origin = pd.Timestamp(row.forecast_origin).normalize().to_datetime64()
        eligible = np.flatnonzero((dates < target) & (dates <= origin))
        matching = eligible[daytypes[eligible] == str(row.day_type)]
        eligible.setflags(write=False)
        matching.setflags(write=False)
        histories[row_id] = AvailableCalibrationHistory(
            pooled_indices=eligible,
            daytype_indices=matching,
        )
    return CalibrationHistoryCache(
        observations=obs,
        histories_by_row_id=histories,
        build_count=1,
    )


def calibrate_predictions(
    frozen_predictions: pd.DataFrame,
    *,
    target_coverage: float,
    calibration_policy: CalibrationPolicy | str,
    minimum_history: int = 20,
    history_cache: CalibrationHistoryCache | None = None,
) -> pd.DataFrame:
    """Apply one deterministic policy without fitting or changing point predictions."""

    policy = resolve_calibration_policy(calibration_policy)
    minimum = int(minimum_history)
    if minimum < 1:
        raise ValueError("minimum_history must be positive")
    frame = frozen_predictions.copy()
    required = {
        "calibration_row_id",
        "target_date",
        "forecast_origin",
        "day_type",
        "actual",
        "point_prediction",
        "raw_quantile_prediction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frozen predictions missing columns: {sorted(missing)}")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise").dt.normalize()
    frame["forecast_origin"] = pd.to_datetime(
        frame["forecast_origin"], errors="raise"
    ).dt.normalize()
    cache = history_cache or build_calibration_history_cache(frame)
    observation_values = cache.observations["residual_observation"].to_numpy(dtype=float)
    observation_dates = pd.to_datetime(cache.observations["target_date"]).to_numpy()
    output_rows: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        payload = row._asdict()
        target = pd.Timestamp(payload["target_date"]).normalize()
        origin = pd.Timestamp(payload["forecast_origin"]).normalize()
        point = float(payload["point_prediction"])
        raw_quantile = float(payload["raw_quantile_prediction"])
        fallback = "not_applicable_raw"
        group = "raw_quantile_reference"
        history_count = 0
        cutoff = pd.NaT
        quantile_index: int | None = None

        if policy.raw_quantile_reference:
            buffer = raw_quantile - point
            upper = raw_quantile
        else:
            available = cache.histories_by_row_id[int(payload["calibration_row_id"])]
            if policy.grouping == "day_type":
                primary = available.daytype_indices
                group = str(payload["day_type"])
            elif policy.grouping == "pooled":
                primary = available.pooled_indices
                group = "pooled"
            else:
                raise AssertionError(f"Unsupported grouping: {policy.grouping}")
            if policy.maximum_history_dates is not None:
                primary = primary[-policy.maximum_history_dates :]
            used = primary
            fallback = "none"
            if len(used) < minimum:
                pooled = available.pooled_indices
                if policy.grouping == "day_type" and len(pooled) >= minimum:
                    used = pooled
                    fallback = "pooled_expanding"
                    group = "pooled_fallback"
                else:
                    used = np.array([], dtype=int)
                    fallback = "insufficient_no_valid_fallback"
            history_count = int(len(used))
            if history_count:
                buffer, quantile_index = conformal_empirical_quantile(
                    observation_values[used], target_coverage
                )
                cutoff = pd.Timestamp(observation_dates[used[-1]]).normalize()
                upper = point + buffer
            else:
                buffer = np.nan
                upper = np.nan

        recommendation = float(np.ceil(upper)) if np.isfinite(upper) else np.nan
        covered: Any = (
            bool(float(payload["actual"]) <= upper) if np.isfinite(upper) else pd.NA
        )
        under = (
            max(float(payload["actual"]) - recommendation, 0.0)
            if np.isfinite(recommendation)
            else np.nan
        )
        over = (
            max(recommendation - float(payload["actual"]), 0.0)
            if np.isfinite(recommendation)
            else np.nan
        )
        provenance_valid = bool(
            pd.isna(cutoff) or (cutoff < target and cutoff <= origin)
        )
        if not provenance_valid:
            raise AssertionError("Calibration provenance cutoff is not origin-valid")
        output_rows.append(
            {
                **payload,
                "calibration_policy_id": policy.calibration_policy_id,
                "coverage_target": float(target_coverage),
                "calibration_buffer": buffer,
                "uncalibrated_upper_prediction": upper,
                "unrounded_upper_recommendation": upper,
                "recommended_meals": recommendation,
                "covered": covered,
                "underprepared_meals": under,
                "overprepared_meals": over,
                "calibration_history_count": history_count,
                "calibration_group": group,
                "fallback_used": fallback,
                "calibration_cutoff_date": cutoff,
                "conformal_quantile_one_based_index": quantile_index,
                "minimum_calibration_history": minimum,
                "provenance_valid": provenance_valid,
                "legacy_percentage_buffer_added": False,
                "saved_residual_buffer_added": False,
            }
        )
    out = pd.DataFrame(output_rows)
    if not np.allclose(
        out["point_prediction"].to_numpy(),
        frame["point_prediction"].to_numpy(),
        rtol=0,
        atol=0,
    ):
        raise AssertionError("Calibration changed frozen point predictions")
    return out


DEVELOPMENT_SELECTION_COLUMNS = [
    "calibration_policy_id",
    "coverage_target",
    "empirical_coverage",
    "mean_over_preparation",
    "mean_under_preparation",
    "sunday_coverage",
    "s2_coverage",
]


def select_development_calibration(
    development_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame, str]:
    """Select at 80% from development aggregates only."""

    missing = set(DEVELOPMENT_SELECTION_COLUMNS).difference(development_metrics.columns)
    if missing:
        raise ValueError(f"Development metrics missing columns: {sorted(missing)}")
    table = development_metrics[DEVELOPMENT_SELECTION_COLUMNS].copy()
    table = table[np.isclose(table["coverage_target"], 0.8)].copy()
    expected = {item.calibration_policy_id for item in CALIBRATION_POLICIES}
    if set(table["calibration_policy_id"]) != expected or len(table) != len(expected):
        raise ValueError("Development table must contain C0-C4 exactly once at 0.80")
    raw = table[table["calibration_policy_id"] == C0.calibration_policy_id].iloc[0]
    table["conservatism_rank"] = table["calibration_policy_id"].map(
        {C0.calibration_policy_id: -1, **CONSERVATISM_RANK}
    )
    primary = table[table["calibration_policy_id"] != C0.calibration_policy_id].copy()
    primary["coverage_distance_from_80"] = (
        primary["empirical_coverage"] - 0.8
    ).abs()
    primary["passes_coverage_band"] = primary["empirical_coverage"].between(
        0.77 - DECISION_EPSILON, 0.84 + DECISION_EPSILON, inclusive="both"
    )
    primary["passes_sunday_guardrail"] = primary["sunday_coverage"].ge(
        0.74 - DECISION_EPSILON
    )
    primary["passes_s2_guardrail"] = primary["s2_coverage"].ge(
        0.74 - DECISION_EPSILON
    )
    primary["passes_underpreparation_guardrail"] = primary[
        "mean_under_preparation"
    ].le(float(raw["mean_under_preparation"]) + DECISION_EPSILON)
    primary["development_eligible"] = primary[
        [
            "passes_coverage_band",
            "passes_sunday_guardrail",
            "passes_s2_guardrail",
            "passes_underpreparation_guardrail",
        ]
    ].all(axis=1)
    eligible = primary[primary["development_eligible"]].copy()
    if not eligible.empty:
        best_over = float(eligible["mean_over_preparation"].min())
        near = eligible[
            eligible["mean_over_preparation"] <= best_over + 1.0 + DECISION_EPSILON
        ]
        selected = str(
            near.sort_values(
                ["conservatism_rank", "mean_over_preparation"], kind="stable"
            ).iloc[0]["calibration_policy_id"]
        )
        path = "coverage_band_then_overpreparation_with_one_meal_simplicity_tolerance"
    else:
        guarded = primary[
            primary["passes_sunday_guardrail"] & primary["passes_s2_guardrail"]
        ].copy()
        if guarded.empty:
            selected = C0.calibration_policy_id
            path = "no_primary_policy_met_sunday_and_s2_guardrails"
        else:
            selected = str(
                guarded.sort_values(
                    [
                        "coverage_distance_from_80",
                        "mean_over_preparation",
                        "conservatism_rank",
                    ],
                    kind="stable",
                ).iloc[0]["calibration_policy_id"]
            )
            path = "closest_to_80_fallback_subject_to_sunday_and_s2"
    audit = table.merge(
        primary[
            [
                "calibration_policy_id",
                "coverage_distance_from_80",
                "passes_coverage_band",
                "passes_sunday_guardrail",
                "passes_s2_guardrail",
                "passes_underpreparation_guardrail",
                "development_eligible",
            ]
        ],
        on="calibration_policy_id",
        how="left",
        validate="one_to_one",
    )
    audit["development_selected"] = audit["calibration_policy_id"].eq(selected)
    audit["selection_path"] = path
    return selected, audit, path


CONFIRMATION_COLUMNS = [
    "calibration_policy_id",
    "coverage_target",
    "empirical_coverage",
    "mean_over_preparation",
    "p90_over_preparation",
    "sunday_coverage",
    "s2_coverage",
]


def apply_confirmation_calibration_guardrails(
    development_locked_policy: str,
    confirmation_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Apply absolute/referenced guardrails without selecting another window."""

    locked = resolve_calibration_policy(development_locked_policy).calibration_policy_id
    missing = set(CONFIRMATION_COLUMNS).difference(confirmation_metrics.columns)
    if missing:
        raise ValueError(f"Confirmation metrics missing columns: {sorted(missing)}")
    table = confirmation_metrics[CONFIRMATION_COLUMNS].copy()
    table = table[np.isclose(table["coverage_target"], 0.8)].copy()
    expected = {item.calibration_policy_id for item in CALIBRATION_POLICIES}
    if set(table["calibration_policy_id"]) != expected or len(table) != len(expected):
        raise ValueError("Confirmation table must contain C0-C4 exactly once at 0.80")
    raw = table[table["calibration_policy_id"] == C0.calibration_policy_id].iloc[0]
    table["mean_over_vs_raw"] = (
        table["mean_over_preparation"] - float(raw["mean_over_preparation"])
    )
    table["p90_over_vs_raw"] = (
        table["p90_over_preparation"] - float(raw["p90_over_preparation"])
    )
    table["passes_overall_coverage"] = table["empirical_coverage"].ge(
        0.74 - DECISION_EPSILON
    )
    table["passes_sunday_coverage"] = table["sunday_coverage"].ge(
        0.70 - DECISION_EPSILON
    )
    table["passes_s2_coverage"] = table["s2_coverage"].ge(
        0.70 - DECISION_EPSILON
    )
    table["passes_mean_overpreparation"] = table["mean_over_vs_raw"].le(
        5.0 + DECISION_EPSILON
    )
    table["passes_p90_overpreparation"] = table["p90_over_vs_raw"].le(
        10.0 + DECISION_EPSILON
    )
    pass_columns = [
        "passes_overall_coverage",
        "passes_sunday_coverage",
        "passes_s2_coverage",
        "passes_mean_overpreparation",
        "passes_p90_overpreparation",
    ]
    table["passes_all_confirmation_guardrails"] = table[pass_columns].all(axis=1)
    if locked == C0.calibration_policy_id:
        final = C0.calibration_policy_id
        table["guardrail_applied_to_locked_policy"] = False
    else:
        locked_row = table[table["calibration_policy_id"] == locked].iloc[0]
        final = locked if bool(locked_row["passes_all_confirmation_guardrails"]) else C0.calibration_policy_id
        table["guardrail_applied_to_locked_policy"] = table[
            "calibration_policy_id"
        ].eq(locked)
    table["final_recommended_policy"] = table["calibration_policy_id"].eq(final)
    return final, table


def calibration_registry_rows() -> list[dict[str, Any]]:
    return [item.to_dict() for item in CALIBRATION_POLICIES]
