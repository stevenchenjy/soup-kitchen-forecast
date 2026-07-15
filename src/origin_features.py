"""Pure, origin-aware feature reconstruction for the Phase 1 backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL
from src.data_processing import add_basic_calendar_features


CALENDAR_FEATURES = [
    "year_num",
    "month_num",
    "day_num",
    "weekday_num",
    "weekofyear",
    "is_weekend",
    "month_sin",
    "month_cos",
    "slot_num",
    "is_sun",
]

ATTENDANCE_FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "rolling_mean_3",
    "rolling_std_3",
    "lag_same_daytype_1",
    "rolling_mean_daytype_3",
    "rolling_std_daytype_3",
    "lag_same_slot_1",
    "rolling_mean_slot_3",
    "rolling_std_slot_3",
]

WEATHER_FEATURES = [
    "temp_10_13",
    "apparent_temp_10_13",
    "humidity_10_13",
    "wind_10_13",
    "precip_10_13",
]

MODEL_FEATURES = CALENDAR_FEATURES + ATTENDANCE_FEATURES + WEATHER_FEATURES

LAST_OBSERVED_DAYTYPE_FEATURES = [
    "last_observed_daytype_1",
    "last_observed_daytype_2",
    "last_observed_daytype_3",
    "last_observed_daytype_4",
    "last_observed_daytype_6",
]

DAYTYPE_SUMMARY_FEATURES = [
    "daytype_mean_last_2",
    "daytype_mean_last_4",
    "daytype_median_last_4",
    "daytype_mean_last_6",
    "daytype_median_last_6",
    "daytype_mean_last_8",
    "daytype_std_last_4",
    "daytype_std_last_8",
    "daytype_min_last_4",
    "daytype_max_last_4",
    "daytype_recent_vs_previous_3",
    "daytype_mean2_minus_previous2",
]

DAYTYPE_SLOT_FEATURES = [
    "daytype_slot_last_observed",
    "daytype_slot_mean_last_2",
    "daytype_slot_median_last_3",
    "daytype_slot_match_count",
    "daytype_slot_days_since_latest",
]

HORIZON_AWARE_FEATURES = [
    "calendar_days_ahead",
    "service_horizon",
    "observed_daytype_count",
    "missing_last_observed_daytype_1",
    "missing_last_observed_daytype_2",
    "missing_last_observed_daytype_3",
    "missing_last_observed_daytype_4",
    "missing_last_observed_daytype_6",
    "daytype_slot_history_missing",
    "future_eligible_services_between",
    "days_since_last_observed_daytype",
]

REPAIRED_FEATURES = (
    CALENDAR_FEATURES
    + LAST_OBSERVED_DAYTYPE_FEATURES
    + DAYTYPE_SUMMARY_FEATURES
    + DAYTYPE_SLOT_FEATURES
    + HORIZON_AWARE_FEATURES
)

W0_NO_WEATHER = "W0_no_weather"
W1_OBSERVED_REPLAY = "W1_observed_weather_replay"
W2_ARCHIVED_FORECAST = "W2_archived_forecast"
WEATHER_POLICIES = (W0_NO_WEATHER, W1_OBSERVED_REPLAY, W2_ARCHIVED_FORECAST)

T0_LEGACY_ALL = "T0_legacy_all_records"
T1_VALID_WEEKENDS = "T1_valid_weekends"
WEEKDAY_POLICIES = (T0_LEGACY_ALL, T1_VALID_WEEKENDS)


@dataclass(frozen=True)
class OriginFeatureResult:
    """One raw feature row plus per-feature audit provenance."""

    features: pd.Series
    provenance: tuple[dict[str, Any], ...]


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _policy_history(attendance_history: pd.DataFrame, weekday_policy: str) -> pd.DataFrame:
    if weekday_policy not in WEEKDAY_POLICIES:
        raise ValueError(f"Unsupported weekday policy: {weekday_policy}")
    required = {DATE_COL, TARGET_COL}
    missing = required.difference(attendance_history.columns)
    if missing:
        raise ValueError(f"Attendance history is missing columns: {sorted(missing)}")

    history = attendance_history[[DATE_COL, TARGET_COL]].copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL], errors="raise").dt.normalize()
    history[TARGET_COL] = pd.to_numeric(history[TARGET_COL], errors="coerce")
    history = history.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    if history[DATE_COL].duplicated().any():
        raise ValueError("Attendance history contains duplicate service dates")
    if weekday_policy == T1_VALID_WEEKENDS:
        history = history[history[DATE_COL].dt.weekday.isin([5, 6])].reset_index(drop=True)

    calendar = add_basic_calendar_features(history)
    return calendar


def _attendance_provenance(
    feature: str,
    sources: pd.DataFrame,
    origin: pd.Timestamp,
    required_count: int,
    aggregation: str,
    semantic_note: str = "",
) -> tuple[float, dict[str, Any]]:
    sources = sources.tail(required_count).copy()
    source_dates = [_iso(value) for value in sources[DATE_COL]]
    available = sources[sources[DATE_COL] <= origin]
    withheld = sources[sources[DATE_COL] > origin]

    if len(sources) < required_count:
        status = "insufficient_history"
        reason = f"requires {required_count} conceptual source record(s); found {len(sources)}"
        value = np.nan
    elif not withheld.empty:
        status = "unavailable_at_origin"
        reason = "one or more target-relative source records occur after forecast origin"
        value = np.nan
    elif sources[TARGET_COL].isna().any():
        status = "missing_attendance"
        reason = "one or more origin-available source attendance values are missing"
        value = np.nan
    else:
        values = sources[TARGET_COL].astype(float)
        if aggregation == "last":
            value = float(values.iloc[-1])
        elif aggregation == "mean":
            value = float(values.mean())
        elif aggregation == "std":
            value = float(values.std(ddof=1))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        status = "available"
        reason = "all target-relative source records were observed by forecast origin"

    provenance = {
        "feature": feature,
        "status": status,
        "source_type": "attendance",
        "source_dates": source_dates,
        "available_source_dates": [_iso(value) for value in available[DATE_COL]],
        "available_source_values": [float(value) for value in available[TARGET_COL].dropna()],
        "withheld_source_dates": [_iso(value) for value in withheld[DATE_COL]],
        "missing_reason": reason,
        "aggregation": aggregation,
        "semantic_note": semantic_note,
        "weather_source_type": "",
        "weather_issue_date": "",
    }
    return value, provenance


def build_features_as_of(
    attendance_history: pd.DataFrame,
    target_date: Any,
    forecast_origin: Any,
    *,
    weather_policy: str = W0_NO_WEATHER,
    weather_df: pd.DataFrame | None = None,
    weekday_policy: str = T1_VALID_WEEKENDS,
    feature_cols: list[str] | tuple[str, ...] | None = None,
) -> OriginFeatureResult:
    """Build one deterministic raw feature row using information known at an origin.

    Target-relative attendance sources are identified before the origin cutoff is
    applied. If a conceptual source lies after the cutoff, the feature remains
    missing instead of substituting an older observation.
    """

    target = pd.Timestamp(target_date).normalize()
    origin = pd.Timestamp(forecast_origin).normalize()
    if origin >= target:
        raise ValueError("forecast_origin must be earlier than target_date")
    if weather_policy not in WEATHER_POLICIES:
        raise ValueError(f"Unsupported weather policy: {weather_policy}")
    if weather_policy == W2_ARCHIVED_FORECAST:
        raise ValueError("W2 archived forecast weather is unavailable in this repository")

    requested_features = list(feature_cols or MODEL_FEATURES)
    unknown = sorted(set(requested_features).difference(MODEL_FEATURES))
    if unknown:
        raise ValueError(f"Unsupported Phase 1 features: {unknown}")

    history = _policy_history(attendance_history, weekday_policy)
    prior = history[history[DATE_COL] < target].copy()
    target_calendar = add_basic_calendar_features(
        pd.DataFrame([{DATE_COL: target, TARGET_COL: np.nan}])
    ).iloc[0]
    target_is_sun = int(target_calendar["is_sun"])
    target_slot = int(target_calendar["slot_num"])

    values: dict[str, float] = {}
    provenance: list[dict[str, Any]] = []

    for feature in CALENDAR_FEATURES:
        values[feature] = float(target_calendar[feature])
        provenance.append(
            {
                "feature": feature,
                "status": "calendar_known",
                "source_type": "target_calendar",
                "source_dates": [_iso(target)],
                "available_source_dates": [_iso(target)],
                "available_source_values": [],
                "withheld_source_dates": [],
                "missing_reason": "",
                "aggregation": "deterministic_calendar",
                "semantic_note": "",
                "weather_source_type": "",
                "weather_issue_date": "",
            }
        )

    combined = prior
    same_daytype = prior[prior["is_sun"] == target_is_sun]
    same_slot = prior[prior["slot_num"] == target_slot]

    attendance_specs = [
        ("lag1", combined, 1, "last", ""),
        ("lag2", combined.iloc[:-1], 1, "last", ""),
        ("lag3", combined.iloc[:-2], 1, "last", ""),
        ("rolling_mean_3", combined, 3, "mean", ""),
        ("rolling_std_3", combined, 3, "std", ""),
        ("lag_same_daytype_1", same_daytype, 1, "last", ""),
        ("rolling_mean_daytype_3", same_daytype, 3, "mean", ""),
        ("rolling_std_daytype_3", same_daytype, 3, "std", ""),
        (
            "lag_same_slot_1",
            same_slot,
            1,
            "last",
            "preserves current slot_num-only grouping, which can mix day types",
        ),
        (
            "rolling_mean_slot_3",
            same_slot,
            3,
            "mean",
            "preserves current slot_num-only grouping, which can mix day types",
        ),
        (
            "rolling_std_slot_3",
            same_slot,
            3,
            "std",
            "preserves current slot_num-only grouping, which can mix day types",
        ),
    ]
    for feature, sources, count, aggregation, note in attendance_specs:
        value, item = _attendance_provenance(
            feature,
            sources,
            origin,
            required_count=count,
            aggregation=aggregation,
            semantic_note=note,
        )
        values[feature] = value
        provenance.append(item)

    weather_row = None
    if weather_policy == W1_OBSERVED_REPLAY and weather_df is not None and not weather_df.empty:
        weather = weather_df.copy()
        if "date" not in weather.columns:
            raise ValueError("weather_df must contain a date column")
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
        matches = weather[weather["date"] == target]
        if not matches.empty:
            weather_row = matches.iloc[-1]

    for feature in WEATHER_FEATURES:
        if weather_policy == W0_NO_WEATHER:
            value = np.nan
            status = "weather_disabled"
            reason = "W0 intentionally disables weather"
            source_type = "none"
        elif weather_row is None or feature not in weather_row.index or pd.isna(weather_row[feature]):
            value = np.nan
            status = "weather_missing"
            reason = "target-date realized weather is absent from the local cache"
            source_type = "observed_weather_replay"
        else:
            value = float(weather_row[feature])
            status = "available"
            reason = "realized target-date weather replay; not origin-valid forecast weather"
            source_type = "observed_weather_replay"
        values[feature] = value
        provenance.append(
            {
                "feature": feature,
                "status": status,
                "source_type": "weather",
                "source_dates": [_iso(target)] if weather_row is not None else [],
                "available_source_dates": [_iso(target)] if status == "available" else [],
                "available_source_values": [value] if status == "available" else [],
                "withheld_source_dates": [],
                "missing_reason": reason,
                "aggregation": "target_date_join",
                "semantic_note": "observed replay is not an archived origin-time forecast",
                "weather_source_type": source_type,
                "weather_issue_date": "unavailable",
            }
        )

    output = pd.Series({feature: values.get(feature, np.nan) for feature in requested_features}, dtype=float)
    output = output.replace([np.inf, -np.inf], np.nan)
    requested = set(requested_features)
    return OriginFeatureResult(
        features=output,
        provenance=tuple(item for item in provenance if item["feature"] in requested),
    )


def _repaired_provenance(
    *,
    feature: str,
    target: pd.Timestamp,
    origin: pd.Timestamp,
    target_day_type: str,
    source: pd.DataFrame,
    required_count: int,
    aggregation: str,
    value: float,
    missing_reason: str = "",
    rank_or_window: str = "",
    source_type: str = "attendance",
    production_available: bool = True,
) -> dict[str, Any]:
    """Return the common auditable provenance record for a repaired feature."""

    source_dates = [_iso(item) for item in source[DATE_COL]] if DATE_COL in source else []
    source_values = (
        [float(item) for item in source[TARGET_COL].dropna()]
        if TARGET_COL in source
        else []
    )
    origin_valid = all(pd.Timestamp(item).normalize() <= origin for item in source_dates)
    if pd.isna(value):
        status = "insufficient_history" if missing_reason else "missing"
    elif source_type == "attendance":
        status = "available"
    else:
        status = "deterministic"
    return {
        "feature": feature,
        "status": status,
        "source_type": source_type,
        "source_dates": source_dates,
        "available_source_dates": source_dates,
        "available_source_values": source_values,
        "withheld_source_dates": [],
        "missing_reason": missing_reason,
        "aggregation": aggregation,
        "semantic_note": "feature meaning is indexed by information available at forecast origin",
        "weather_source_type": "",
        "weather_issue_date": "",
        "forecast_origin": _iso(origin),
        "target_date": _iso(target),
        "target_day_type": target_day_type,
        "rank_or_window": rank_or_window,
        "required_source_count": int(required_count),
        "used_source_count": int(len(source)),
        "origin_valid": bool(origin_valid),
        "production_available": bool(production_available),
    }


def build_repaired_features_as_of(
    attendance_history: pd.DataFrame,
    target_date: Any,
    forecast_origin: Any,
    *,
    calendar_days_ahead: int,
    service_horizon: int,
    weekday_policy: str = T1_VALID_WEEKENDS,
    feature_cols: list[str] | tuple[str, ...],
) -> OriginFeatureResult:
    """Build origin-valid repaired features with value-level attendance provenance.

    Repaired attendance features always select from matching-day-type records
    observed on or before ``forecast_origin``. No target-relative future record is
    ever a conceptual source, so each feature has the same meaning across S1-S5.
    """

    target = pd.Timestamp(target_date).normalize()
    origin = pd.Timestamp(forecast_origin).normalize()
    if origin >= target:
        raise ValueError("forecast_origin must be earlier than target_date")
    if int(calendar_days_ahead) != int((target - origin).days):
        raise ValueError("calendar_days_ahead does not match target minus origin")
    if int(service_horizon) < 1:
        raise ValueError("service_horizon must be positive")

    requested_features = list(feature_cols)
    unknown = sorted(set(requested_features).difference(REPAIRED_FEATURES))
    if unknown:
        raise ValueError(f"Unsupported repaired features: {unknown}")
    if len(requested_features) != len(set(requested_features)):
        raise ValueError("feature_cols contains duplicate names")

    history = _policy_history(attendance_history, weekday_policy)
    target_calendar = add_basic_calendar_features(
        pd.DataFrame([{DATE_COL: target, TARGET_COL: np.nan}])
    ).iloc[0]
    target_is_sun = int(target_calendar["is_sun"])
    target_day_type = "Sunday" if target_is_sun else "Saturday"
    target_slot = int(target_calendar["slot_num"])
    observed = history[
        (history[DATE_COL] <= origin)
        & (history[DATE_COL] < target)
        & (history[TARGET_COL].notna())
    ].copy()
    same_day = observed[observed["is_sun"] == target_is_sun].sort_values(
        DATE_COL, kind="stable"
    )
    same_slot = same_day[same_day["slot_num"] == target_slot].sort_values(
        DATE_COL, kind="stable"
    )

    values: dict[str, float] = {}
    provenance: list[dict[str, Any]] = []
    empty_source = history.iloc[0:0][[DATE_COL, TARGET_COL]]

    for feature in CALENDAR_FEATURES:
        values[feature] = float(target_calendar[feature])
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=empty_source,
                required_count=0,
                aggregation="deterministic_target_calendar",
                value=values[feature],
                source_type="target_calendar",
            )
        )

    ranks = [1, 2, 3, 4, 6]
    for rank in ranks:
        feature = f"last_observed_daytype_{rank}"
        if len(same_day) >= rank:
            source = same_day.iloc[[-rank]][[DATE_COL, TARGET_COL]]
            value = float(source[TARGET_COL].iloc[0])
            reason = ""
        else:
            source = same_day.iloc[0:0][[DATE_COL, TARGET_COL]]
            value = np.nan
            reason = f"requires matching-day-type rank {rank}; found {len(same_day)}"
        values[feature] = value
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=source,
                required_count=rank,
                aggregation="ranked_last_observed",
                value=value,
                missing_reason=reason,
                rank_or_window=f"rank_{rank}",
            )
        )

    summary_specs: list[tuple[str, int, str]] = [
        ("daytype_mean_last_2", 2, "mean"),
        ("daytype_mean_last_4", 4, "mean"),
        ("daytype_median_last_4", 4, "median"),
        ("daytype_mean_last_6", 6, "mean"),
        ("daytype_median_last_6", 6, "median"),
        ("daytype_mean_last_8", 8, "mean"),
        ("daytype_std_last_4", 4, "std"),
        ("daytype_std_last_8", 8, "std"),
        ("daytype_min_last_4", 4, "min"),
        ("daytype_max_last_4", 4, "max"),
    ]
    for feature, window, aggregation in summary_specs:
        source = same_day.tail(window)[[DATE_COL, TARGET_COL]]
        if len(source) < window:
            value = np.nan
            reason = f"requires {window} matching-day-type observations; found {len(source)}"
        else:
            series = source[TARGET_COL].astype(float)
            if aggregation == "mean":
                value = float(series.mean())
            elif aggregation == "median":
                value = float(series.median())
            elif aggregation == "std":
                value = float(series.std(ddof=1))
            elif aggregation == "min":
                value = float(series.min())
            elif aggregation == "max":
                value = float(series.max())
            else:  # pragma: no cover - the static table above controls this
                raise AssertionError(aggregation)
            reason = ""
        values[feature] = value
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=source,
                required_count=window,
                aggregation=aggregation,
                value=value,
                missing_reason=reason,
                rank_or_window=f"last_{window}",
            )
        )

    contrast_specs = [
        ("daytype_recent_vs_previous_3", "recent_minus_previous3_mean"),
        ("daytype_mean2_minus_previous2", "last2_mean_minus_previous2_mean"),
    ]
    for feature, aggregation in contrast_specs:
        source = same_day.tail(4)[[DATE_COL, TARGET_COL]]
        if len(source) < 4:
            value = np.nan
            reason = f"requires 4 matching-day-type observations; found {len(source)}"
        else:
            series = source[TARGET_COL].astype(float).reset_index(drop=True)
            if feature == "daytype_recent_vs_previous_3":
                value = float(series.iloc[3] - series.iloc[:3].mean())
            else:
                value = float(series.iloc[2:].mean() - series.iloc[:2].mean())
            reason = ""
        values[feature] = value
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=source,
                required_count=4,
                aggregation=aggregation,
                value=value,
                missing_reason=reason,
                rank_or_window="last_4",
            )
        )

    slot_specs: list[tuple[str, int, str]] = [
        ("daytype_slot_last_observed", 1, "last"),
        ("daytype_slot_mean_last_2", 2, "mean"),
        ("daytype_slot_median_last_3", 3, "median"),
    ]
    for feature, window, aggregation in slot_specs:
        source = same_slot.tail(window)[[DATE_COL, TARGET_COL]]
        if len(source) < window:
            value = np.nan
            reason = (
                f"requires {window} matching day-type/slot observations; "
                f"found {len(source)}"
            )
        else:
            series = source[TARGET_COL].astype(float)
            value = float(
                series.iloc[-1]
                if aggregation == "last"
                else series.mean()
                if aggregation == "mean"
                else series.median()
            )
            reason = ""
        values[feature] = value
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=source,
                required_count=window,
                aggregation=aggregation,
                value=value,
                missing_reason=reason,
                rank_or_window=f"same_daytype_slot_last_{window}",
            )
        )

    slot_all = same_slot[[DATE_COL, TARGET_COL]]
    values["daytype_slot_match_count"] = float(len(slot_all))
    values["daytype_slot_days_since_latest"] = (
        float((origin - pd.Timestamp(slot_all[DATE_COL].iloc[-1])).days)
        if not slot_all.empty
        else np.nan
    )
    for feature, value, aggregation in [
        ("daytype_slot_match_count", values["daytype_slot_match_count"], "count"),
        (
            "daytype_slot_days_since_latest",
            values["daytype_slot_days_since_latest"],
            "origin_minus_latest_source_date",
        ),
    ]:
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=slot_all,
                required_count=0 if feature.endswith("match_count") else 1,
                aggregation=aggregation,
                value=value,
                missing_reason="no matching day-type/slot history" if pd.isna(value) else "",
                rank_or_window="all_observed_matches",
            )
        )

    latest_day_source = same_day.tail(1)[[DATE_COL, TARGET_COL]]
    horizon_values = {
        "calendar_days_ahead": float(calendar_days_ahead),
        "service_horizon": float(service_horizon),
        "observed_daytype_count": float(len(same_day)),
        "future_eligible_services_between": float(service_horizon - 1),
        "days_since_last_observed_daytype": (
            float((origin - pd.Timestamp(latest_day_source[DATE_COL].iloc[-1])).days)
            if not latest_day_source.empty
            else np.nan
        ),
        "daytype_slot_history_missing": float(slot_all.empty),
    }
    for rank in ranks:
        horizon_values[f"missing_last_observed_daytype_{rank}"] = float(
            pd.isna(values[f"last_observed_daytype_{rank}"])
        )
    values.update(horizon_values)

    for feature in HORIZON_AWARE_FEATURES:
        if feature.startswith("missing_last_observed"):
            referenced = feature.removeprefix("missing_")
            source = next(
                item for item in provenance if item["feature"] == referenced
            )
            source_frame = same_day.tail(source["used_source_count"])[[DATE_COL, TARGET_COL]]
            aggregation = f"is_missing({referenced})"
        elif feature in {"observed_daytype_count", "days_since_last_observed_daytype"}:
            source_frame = same_day[[DATE_COL, TARGET_COL]]
            aggregation = feature
        elif feature == "daytype_slot_history_missing":
            source_frame = slot_all
            aggregation = "no_matching_daytype_slot_history"
        else:
            source_frame = empty_source
            aggregation = "deterministic_origin_target_horizon"
        provenance.append(
            _repaired_provenance(
                feature=feature,
                target=target,
                origin=origin,
                target_day_type=target_day_type,
                source=source_frame,
                required_count=0,
                aggregation=aggregation,
                value=values[feature],
                missing_reason=(
                    "no observed matching day type"
                    if feature == "days_since_last_observed_daytype"
                    and pd.isna(values[feature])
                    else ""
                ),
                source_type=(
                    "attendance"
                    if feature
                    in {
                        "observed_daytype_count",
                        "days_since_last_observed_daytype",
                        "daytype_slot_history_missing",
                    }
                    or feature.startswith("missing_last_observed")
                    else "origin_target_metadata"
                ),
            )
        )

    output = pd.Series(
        {feature: values.get(feature, np.nan) for feature in requested_features}, dtype=float
    ).replace([np.inf, -np.inf], np.nan)
    requested = set(requested_features)
    return OriginFeatureResult(
        features=output,
        provenance=tuple(item for item in provenance if item["feature"] in requested),
    )


def origin_available_baselines(
    attendance_history: pd.DataFrame,
    target_date: Any,
    forecast_origin: Any,
    *,
    weekday_policy: str = T1_VALID_WEEKENDS,
) -> dict[str, float]:
    """Return fixed simple baselines using attendance observed by the origin."""

    target = pd.Timestamp(target_date).normalize()
    origin = pd.Timestamp(forecast_origin).normalize()
    history = _policy_history(attendance_history, weekday_policy)
    observed = history[(history[DATE_COL] <= origin) & (history[DATE_COL] < target)].copy()
    target_calendar = add_basic_calendar_features(
        pd.DataFrame([{DATE_COL: target, TARGET_COL: np.nan}])
    ).iloc[0]
    same_day = observed[observed["is_sun"] == int(target_calendar["is_sun"])]
    last_four = same_day.tail(4)[TARGET_COL].dropna().astype(float)
    same_slot = same_day[same_day["slot_num"] == int(target_calendar["slot_num"])]

    def last_or_nan(frame: pd.DataFrame) -> float:
        valid = frame[TARGET_COL].dropna()
        return float(valid.iloc[-1]) if not valid.empty else np.nan

    return {
        "previous_same_daytype": last_or_nan(same_day),
        "mean_last4_same_daytype": float(last_four.mean()) if not last_four.empty else np.nan,
        "median_last4_same_daytype": float(last_four.median()) if not last_four.empty else np.nan,
        "expanding_same_daytype_mean": float(same_day[TARGET_COL].mean()) if not same_day.empty else np.nan,
        "seasonal_same_slot": last_or_nan(same_slot),
    }


def apply_calibration_placeholder(
    prediction: float,
    *,
    calibration: Any | None = None,
) -> float:
    """Phase 2 hook; Phase 1 deliberately performs no calibration."""

    if calibration is not None:
        raise ValueError("Calibration optimization is outside Phase 1")
    return float(prediction)
