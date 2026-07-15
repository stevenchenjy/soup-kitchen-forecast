"""Origin-aware, horizon-indexed evaluation for the unchanged current models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import DATE_COL, TARGET_COL
from src.modeling import make_point_model, make_quantile_model
from src.origin_features import (
    ATTENDANCE_FEATURES,
    MODEL_FEATURES,
    T0_LEGACY_ALL,
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
    W1_OBSERVED_REPLAY,
    WEATHER_FEATURES,
    WEEKDAY_POLICIES,
    apply_calibration_placeholder,
    build_features_as_of,
    origin_available_baselines,
)
from src.training_windows import (
    TrainingWindowDefinition,
    apply_training_window,
    resolve_training_window,
)


SCENARIO_DEFINITIONS = {
    "S1_same_weekend_saturday": "Friday before the target Saturday",
    "S2_same_weekend_sunday": "Friday before the target weekend; Saturday actual withheld",
    "S3_next_service": "end of the immediately preceding policy-valid recorded service",
    "S4_two_service_ahead": "end of the same weekday seven calendar days earlier",
    "S5_longest_supported": "end of target minus 15 days; application maximum",
}

BASELINE_COLUMNS = [
    "previous_same_daytype",
    "mean_last4_same_daytype",
    "median_last4_same_daytype",
    "expanding_same_daytype_mean",
    "seasonal_same_slot",
]


@dataclass(frozen=True)
class ForecastScenario:
    scenario: str
    forecast_origin: pd.Timestamp
    target_date: pd.Timestamp
    calendar_days_ahead: int
    service_horizon: int


@dataclass
class OriginBacktestResult:
    predictions: pd.DataFrame
    preprocessing_diagnostics: pd.DataFrame
    feature_provenance: pd.DataFrame
    skipped_folds: pd.DataFrame


@dataclass(frozen=True)
class CachedPointTrainingFold:
    """One immutable expanding-history training context for point-model reuse."""

    cache_key: tuple[str, str, str, int, pd.Timestamp]
    segment: str
    training_row_indices: tuple[int, ...]
    training_dates: tuple[pd.Timestamp, ...]
    x_train: np.ndarray
    y_train: np.ndarray
    imputer_statistics: np.ndarray
    preprocessing_id: str
    training_start_date: pd.Timestamp
    training_end_date: pd.Timestamp


@dataclass(frozen=True)
class CachedPointPredictionFold:
    """One prediction key referencing a shared immutable training context."""

    training_cache_key: tuple[str, str, str, int, pd.Timestamp]
    x_test: np.ndarray
    metadata: dict[str, Any]


@dataclass
class CachedPointFoldSet:
    training_folds: dict[
        tuple[str, str, str, int, pd.Timestamp], CachedPointTrainingFold
    ]
    prediction_folds: list[CachedPointPredictionFold]
    skipped_folds: pd.DataFrame
    training_frame_build_count: int
    prediction_feature_build_count: int


def _normalized_history(history: pd.DataFrame, weekday_policy: str) -> pd.DataFrame:
    if weekday_policy not in WEEKDAY_POLICIES:
        raise ValueError(f"Unsupported weekday policy: {weekday_policy}")
    out = history[[DATE_COL, TARGET_COL]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="raise").dt.normalize()
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="raise").astype(float)
    out = out.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    if out[DATE_COL].duplicated().any():
        raise ValueError("Attendance history contains duplicate service dates")
    if weekday_policy == T1_VALID_WEEKENDS:
        out = out[out[DATE_COL].dt.weekday.isin([5, 6])].reset_index(drop=True)
    return out


def _count_weekend_services(origin: pd.Timestamp, target: pd.Timestamp) -> int:
    dates = pd.date_range(origin + pd.Timedelta(days=1), target, freq="D")
    return int(sum(value.weekday() in {5, 6} for value in dates))


def generate_forecast_scenarios(
    history: pd.DataFrame,
    target_date: Any,
    *,
    weekday_policy: str,
) -> list[ForecastScenario]:
    """Generate deterministic operational origins for one recorded target."""

    target = pd.Timestamp(target_date).normalize()
    records = _normalized_history(history, weekday_policy)
    if weekday_policy == T1_VALID_WEEKENDS and target.weekday() not in {5, 6}:
        return []

    scenarios: list[ForecastScenario] = []

    def add(name: str, origin: pd.Timestamp, service_horizon: int | None = None) -> None:
        if origin >= target:
            return
        calendar_days = int((target - origin).days)
        horizon = _count_weekend_services(origin, target) if service_horizon is None else service_horizon
        if horizon < 1:
            return
        scenarios.append(
            ForecastScenario(
                scenario=name,
                forecast_origin=origin,
                target_date=target,
                calendar_days_ahead=calendar_days,
                service_horizon=int(horizon),
            )
        )

    if target.weekday() == 5:
        add("S1_same_weekend_saturday", target - pd.Timedelta(days=1), 1)
    if target.weekday() == 6:
        add("S2_same_weekend_sunday", target - pd.Timedelta(days=2), 2)

    prior = records[records[DATE_COL] < target]
    if not prior.empty:
        add("S3_next_service", pd.Timestamp(prior[DATE_COL].iloc[-1]), 1)

    if target.weekday() in {5, 6}:
        add("S4_two_service_ahead", target - pd.Timedelta(days=7), 2)
        add("S5_longest_supported", target - pd.Timedelta(days=15), 5)

    return scenarios


def calendar_days_bucket(days: int) -> str:
    if days == 1:
        return "1 day"
    if days == 2:
        return "2 days"
    if days <= 7:
        return "3-7 days"
    return "8-15 days"


def calculate_metrics(
    frame: pd.DataFrame,
    *,
    prediction_col: str = "point_prediction",
    quantile_col: str | None = "quantile_prediction",
) -> dict[str, float | int]:
    """Calculate the fixed Phase 1 metric set for one aligned frame."""

    usable = frame.dropna(subset=["actual", prediction_col]).copy()
    if usable.empty:
        return {"row_count": 0}
    actual = usable["actual"].astype(float).to_numpy()
    prediction = usable[prediction_col].astype(float).to_numpy()
    error = prediction - actual
    absolute = np.abs(error)
    result: dict[str, float | int] = {
        "row_count": int(len(usable)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
        "mean_signed_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute)),
        "p75_absolute_error": float(np.quantile(absolute, 0.75)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "max_absolute_error": float(np.max(absolute)),
        "underprediction_frequency": float(np.mean(error < 0)),
        "overprediction_frequency": float(np.mean(error > 0)),
    }
    if quantile_col and quantile_col in usable.columns:
        q_usable = usable.dropna(subset=[quantile_col])
        if not q_usable.empty:
            q_actual = q_usable["actual"].astype(float).to_numpy()
            quantile = q_usable[quantile_col].astype(float).to_numpy()
            residual = q_actual - quantile
            pinball = np.maximum(0.8 * residual, (0.8 - 1.0) * residual)
            result.update(
                {
                    "quantile_row_count": int(len(q_usable)),
                    "raw_quantile_coverage": float(np.mean(q_actual <= quantile)),
                    "mean_quantile_excess_shortfall": float(np.mean(quantile - q_actual)),
                    "mean_quantile_pinball_loss": float(np.mean(pinball)),
                }
            )
    return result


class OriginAwareBacktester:
    """Origin-aware evaluator with optional segment-level training windows."""

    def __init__(
        self,
        attendance_history: pd.DataFrame,
        *,
        weather_df: pd.DataFrame | None,
        feature_cols: Iterable[str] = MODEL_FEATURES,
        residual_buffer_by_day: dict[str, float] | None = None,
        default_meal_buffer_pct: float = 0.08,
        min_train_size: int = 18,
        quantile: float = 0.8,
        feature_set_id: str = "F0_CURRENT_ORIGIN",
        feature_builder: Callable[..., Any] | None = None,
        attendance_feature_cols: Iterable[str] | None = None,
        random_seed: int = 42,
        training_window: TrainingWindowDefinition | str | None = None,
    ) -> None:
        self.history = _normalized_history(attendance_history, T0_LEGACY_ALL)
        self.weather_df = None if weather_df is None else weather_df.copy(deep=True)
        self.feature_cols = list(feature_cols)
        self.feature_builder = feature_builder
        if self.feature_builder is None and self.feature_cols != list(MODEL_FEATURES):
            raise ValueError("Phase 1 requires the unchanged ordered 26-feature model contract")
        self.feature_set_id = str(feature_set_id)
        self.attendance_feature_cols = list(
            ATTENDANCE_FEATURES if attendance_feature_cols is None else attendance_feature_cols
        )
        unknown_attendance = set(self.attendance_feature_cols).difference(self.feature_cols)
        if unknown_attendance:
            raise ValueError(
                f"attendance_feature_cols are absent from feature_cols: {sorted(unknown_attendance)}"
            )
        self.random_seed = int(random_seed)
        self.training_window = resolve_training_window(training_window)
        self.residual_buffer_by_day = residual_buffer_by_day or {"sat": 0.0, "sun": 0.0}
        self.default_meal_buffer_pct = float(default_meal_buffer_pct)
        self.min_train_size = int(min_train_size)
        self.quantile = float(quantile)
        self._training_frames: dict[tuple[str, str], pd.DataFrame] = {}
        self._fit_cache: dict[tuple[str, str, str, str, int, pd.Timestamp], dict[str, Any]] = {}
        self._preprocessing_rows: list[dict[str, Any]] = []

    @staticmethod
    def segment_for_target(target: pd.Timestamp) -> str:
        return "sun" if pd.Timestamp(target).weekday() == 6 else "sat"

    def build_target_features(
        self,
        target_date: Any,
        forecast_origin: Any,
        *,
        weather_policy: str,
        weekday_policy: str,
        calendar_days_ahead: int | None = None,
        service_horizon: int = 1,
    ):
        target = pd.Timestamp(target_date).normalize()
        origin = pd.Timestamp(forecast_origin).normalize()
        days_ahead = (
            int((target - origin).days)
            if calendar_days_ahead is None
            else int(calendar_days_ahead)
        )
        if self.feature_builder is not None:
            return self.feature_builder(
                self.history,
                target,
                origin,
                weather_policy=weather_policy,
                weather_df=self.weather_df,
                weekday_policy=weekday_policy,
                feature_cols=self.feature_cols,
                calendar_days_ahead=days_ahead,
                service_horizon=int(service_horizon),
            )
        return build_features_as_of(
            self.history,
            target,
            origin,
            weather_policy=weather_policy,
            weather_df=self.weather_df,
            weekday_policy=weekday_policy,
            feature_cols=self.feature_cols,
        )

    def _build_training_frame(self, weekday_policy: str, weather_policy: str) -> pd.DataFrame:
        key = (weekday_policy, weather_policy)
        if key in self._training_frames:
            return self._training_frames[key]
        records = _normalized_history(self.history, weekday_policy)
        rows: list[dict[str, Any]] = []
        for index in range(1, len(records)):
            target = pd.Timestamp(records.loc[index, DATE_COL])
            origin = pd.Timestamp(records.loc[index - 1, DATE_COL])
            built = self.build_target_features(
                target,
                origin,
                weather_policy=weather_policy,
                weekday_policy=weekday_policy,
                calendar_days_ahead=int((target - origin).days),
                service_horizon=1,
            )
            row: dict[str, Any] = built.features.to_dict()
            row.update(
                {
                    DATE_COL: target,
                    TARGET_COL: float(records.loc[index, TARGET_COL]),
                    "training_origin": origin,
                    "segment": self.segment_for_target(target),
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
        self._training_frames[key] = frame
        return frame

    def _fit_fold(
        self,
        *,
        forecast_origin: pd.Timestamp,
        segment: str,
        weather_policy: str,
        weekday_policy: str,
    ) -> dict[str, Any] | None:
        full = self._build_training_frame(weekday_policy, weather_policy)
        available = full[full[DATE_COL] <= forecast_origin].copy()
        available_segment = available[available["segment"] == segment].copy()
        if len(available_segment) < self.min_train_size:
            return None
        windowed = apply_training_window(available_segment, self.training_window)
        segment_train = windowed.frame
        if len(segment_train) < self.min_train_size:
            raise AssertionError("Training-window retention violated the unchanged minimum fit size")
        training_end = pd.Timestamp(segment_train[DATE_COL].max())
        cache_key = (
            weekday_policy,
            weather_policy,
            segment,
            self.training_window.training_window_id,
            windowed.available_segment_training_rows,
            training_end,
        )
        if cache_key in self._fit_cache:
            return {**self._fit_cache[cache_key], "training_row_count": int(len(available))}

        if training_end > forecast_origin:
            raise AssertionError("Fold training end exceeds forecast origin")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(segment_train[self.feature_cols])
        if x_train.shape[1] != len(self.feature_cols):
            raise AssertionError("Fold-local imputation changed the feature count")
        y_train = segment_train[TARGET_COL].astype(float)
        point_model = make_point_model()
        quantile_model = make_quantile_model(quantile=self.quantile)
        point_model.fit(x_train, y_train)
        quantile_model.fit(x_train, y_train)

        preprocessing_id = (
            f"{self.feature_set_id}|{weekday_policy}|{weather_policy}|{segment}|"
            f"{self.training_window.training_window_id}|"
            f"available={windowed.available_segment_training_rows}|"
            f"retained={windowed.retained_segment_training_rows}|"
            f"end={training_end:%Y-%m-%d}"
        )
        missing_counts = segment_train[self.feature_cols].isna().sum()
        for index, feature in enumerate(self.feature_cols):
            self._preprocessing_rows.append(
                {
                    "feature_set_id": self.feature_set_id,
                    "preprocessing_id": preprocessing_id,
                    "weekday_policy": weekday_policy,
                    "weather_policy": weather_policy,
                    "segment": segment,
                    "training_window_id": self.training_window.training_window_id,
                    "configured_window_rows": self.training_window.configured_window_rows,
                    "available_segment_training_rows": windowed.available_segment_training_rows,
                    "retained_segment_training_rows": windowed.retained_segment_training_rows,
                    "window_constrained": windowed.window_constrained,
                    "effective_window_days": windowed.effective_window_days,
                    "effective_window_years": windowed.effective_window_years,
                    "feature": feature,
                    "training_start_date": pd.Timestamp(segment_train[DATE_COL].min()),
                    "training_end_date": training_end,
                    "segment_training_row_count": int(len(segment_train)),
                    "training_missing_count": int(missing_counts[feature]),
                    "training_missing_rate": float(missing_counts[feature] / len(segment_train)),
                    "all_training_values_missing": bool(missing_counts[feature] == len(segment_train)),
                    "imputer_strategy": "median",
                    "imputer_statistic": float(imputer.statistics_[index]),
                    "fit_includes_test_or_future": False,
                }
            )

        fit = {
            "imputer": imputer,
            "point_model": point_model,
            "quantile_model": quantile_model,
            "training_end_date": training_end,
            "training_row_count": int(len(available)),
            "segment_training_row_count": int(len(segment_train)),
            "training_window_id": self.training_window.training_window_id,
            "configured_window_rows": self.training_window.configured_window_rows,
            "available_segment_training_rows": windowed.available_segment_training_rows,
            "retained_segment_training_rows": windowed.retained_segment_training_rows,
            "window_constrained": windowed.window_constrained,
            "effective_window_days": windowed.effective_window_days,
            "effective_window_years": windowed.effective_window_years,
            "retained_training_start_date": windowed.retained_training_start_date,
            "retained_training_end_date": windowed.retained_training_end_date,
            "preprocessing_id": preprocessing_id,
        }
        self._fit_cache[cache_key] = fit
        return {**fit, "training_row_count": int(len(available))}

    def prepare_point_fold_cache(
        self,
        *,
        weather_policy: str = W0_NO_WEATHER,
        weekday_policy: str = T1_VALID_WEEKENDS,
        target_dates: Iterable[Any] | None = None,
        period_role_by_target: dict[pd.Timestamp, str] | None = None,
    ) -> CachedPointFoldSet:
        """Build origin-aware point-model arrays once for reuse across fit policies.

        Fold-local median imputation is deliberately unweighted. This method never
        constructs or fits a point or quantile model.
        """

        if self.training_window.training_window_id != "TW_EXPANDING":
            raise ValueError("Cached sample-weight screening requires TW_EXPANDING")
        records = _normalized_history(self.history, weekday_policy)
        requested_dates = (
            None
            if target_dates is None
            else {pd.Timestamp(item).normalize() for item in target_dates}
        )
        if requested_dates is not None:
            records = records[records[DATE_COL].isin(requested_dates)].reset_index(drop=True)
        role_map = {
            pd.Timestamp(key).normalize(): value
            for key, value in (period_role_by_target or {}).items()
        }
        full = self._build_training_frame(weekday_policy, weather_policy)
        training_folds: dict[
            tuple[str, str, str, int, pd.Timestamp], CachedPointTrainingFold
        ] = {}
        prediction_folds: list[CachedPointPredictionFold] = []
        skipped_rows: list[dict[str, Any]] = []
        prediction_feature_build_count = 0
        prediction_keys: set[tuple[Any, ...]] = set()

        for record in records.itertuples(index=False):
            target = pd.Timestamp(getattr(record, DATE_COL)).normalize()
            actual = float(getattr(record, TARGET_COL))
            for scenario in generate_forecast_scenarios(
                self.history, target, weekday_policy=weekday_policy
            ):
                segment = self.segment_for_target(target)
                available = full[full[DATE_COL] <= scenario.forecast_origin]
                segment_train = available[available["segment"] == segment].copy()
                if len(segment_train) < self.min_train_size:
                    skipped_rows.append(
                        {
                            "target_date": target,
                            "forecast_origin": scenario.forecast_origin,
                            "scenario": scenario.scenario,
                            "weather_policy": weather_policy,
                            "weekday_policy": weekday_policy,
                            "reason": f"fewer than {self.min_train_size} segment training rows",
                        }
                    )
                    continue
                segment_train = segment_train.sort_values(DATE_COL, kind="stable")
                training_end = pd.Timestamp(segment_train[DATE_COL].iloc[-1]).normalize()
                cache_key = (
                    weekday_policy,
                    weather_policy,
                    segment,
                    int(len(segment_train)),
                    training_end,
                )
                if cache_key not in training_folds:
                    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                    x_train = imputer.fit_transform(segment_train[self.feature_cols])
                    if x_train.shape[1] != len(self.feature_cols):
                        raise AssertionError("Fold-local imputation changed feature count")
                    training_dates = tuple(
                        pd.Timestamp(value).normalize()
                        for value in segment_train[DATE_COL].tolist()
                    )
                    if training_dates[-1] > scenario.forecast_origin:
                        raise AssertionError("Cached fold includes a post-origin target")
                    preprocessing_id = (
                        f"{self.feature_set_id}|{weekday_policy}|{weather_policy}|"
                        f"{segment}|TW_EXPANDING|n={len(segment_train)}|"
                        f"end={training_end:%Y-%m-%d}|unweighted_imputer"
                    )
                    x_train.setflags(write=False)
                    y_train = segment_train[TARGET_COL].astype(float).to_numpy(copy=True)
                    y_train.setflags(write=False)
                    statistics = np.asarray(imputer.statistics_, dtype=float).copy()
                    statistics.setflags(write=False)
                    training_folds[cache_key] = CachedPointTrainingFold(
                        cache_key=cache_key,
                        segment=segment,
                        training_row_indices=tuple(int(value) for value in segment_train.index),
                        training_dates=training_dates,
                        x_train=x_train,
                        y_train=y_train,
                        imputer_statistics=statistics,
                        preprocessing_id=preprocessing_id,
                        training_start_date=training_dates[0],
                        training_end_date=training_dates[-1],
                    )
                cached_training = training_folds[cache_key]
                built = self.build_target_features(
                    target,
                    scenario.forecast_origin,
                    weather_policy=weather_policy,
                    weekday_policy=weekday_policy,
                    calendar_days_ahead=scenario.calendar_days_ahead,
                    service_horizon=scenario.service_horizon,
                )
                prediction_feature_build_count += 1
                provenance_valid = bool(
                    all(item.get("origin_valid", True) for item in built.provenance)
                )
                for item in built.provenance:
                    if item.get("source_type") != "attendance":
                        continue
                    if any(
                        pd.Timestamp(value).normalize() > scenario.forecast_origin
                        for value in item.get("available_source_dates", [])
                    ):
                        provenance_valid = False
                        break
                if not provenance_valid:
                    raise AssertionError("Cached point fold has post-origin feature provenance")
                raw = built.features.to_frame().T
                # Reconstruct the fold-local transform from the cached unweighted
                # statistics so no candidate-specific preprocessing object exists.
                raw_values = raw[self.feature_cols].to_numpy(dtype=float)
                x_test = np.where(
                    np.isnan(raw_values), cached_training.imputer_statistics, raw_values
                )
                if not np.isfinite(x_test).all():
                    raise AssertionError("Cached transformed point feature row is non-finite")
                x_test.setflags(write=False)
                feature_missing = built.features.isna()
                metadata = {
                    "feature_set_id": self.feature_set_id,
                    "period_role": role_map.get(target, "unassigned"),
                    "forecast_origin": scenario.forecast_origin,
                    "target_date": target,
                    "scenario": scenario.scenario,
                    "calendar_days_ahead": scenario.calendar_days_ahead,
                    "service_horizon": scenario.service_horizon,
                    "day_type": "Sunday" if target.weekday() == 6 else "Saturday",
                    "model_segment": segment,
                    "actual": actual,
                    "weather_policy": weather_policy,
                    "weekday_policy": weekday_policy,
                    "training_end_date": cached_training.training_end_date,
                    "available_segment_training_rows": int(len(segment_train)),
                    "retained_segment_training_rows": int(len(segment_train)),
                    "training_start_date": cached_training.training_start_date,
                    "feature_count": int(len(self.feature_cols)),
                    "feature_missing_count": int(feature_missing.sum()),
                    "feature_provenance_valid": True,
                    "preprocessing_id": cached_training.preprocessing_id,
                    "preprocessing_weighted": False,
                    "random_seed": self.random_seed,
                }
                prediction_key = (
                    self.feature_set_id,
                    target,
                    scenario.scenario,
                    weather_policy,
                    weekday_policy,
                )
                if prediction_key in prediction_keys:
                    raise AssertionError(f"Duplicate cached prediction key: {prediction_key}")
                prediction_keys.add(prediction_key)
                prediction_folds.append(
                    CachedPointPredictionFold(
                        training_cache_key=cache_key,
                        x_test=x_test,
                        metadata=metadata,
                    )
                )
        if not prediction_folds:
            raise ValueError("Point-fold cache contains no eligible predictions")
        return CachedPointFoldSet(
            training_folds=training_folds,
            prediction_folds=prediction_folds,
            skipped_folds=pd.DataFrame(skipped_rows),
            training_frame_build_count=1,
            prediction_feature_build_count=prediction_feature_build_count,
        )

    def run(
        self,
        *,
        weather_policies: Iterable[str] = (W0_NO_WEATHER, W1_OBSERVED_REPLAY),
        weekday_policies: Iterable[str] = (T0_LEGACY_ALL, T1_VALID_WEEKENDS),
        legacy_predictions: pd.DataFrame | None = None,
        target_dates: Iterable[Any] | None = None,
        period_role_by_target: dict[pd.Timestamp, str] | None = None,
    ) -> OriginBacktestResult:
        legacy: dict[pd.Timestamp, tuple[float, float]] = {}
        if legacy_predictions is not None and not legacy_predictions.empty:
            for item in legacy_predictions.itertuples(index=False):
                date = pd.Timestamp(getattr(item, DATE_COL)).normalize()
                legacy[date] = (float(item.pred), float(item.pred_q))

        prediction_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        requested_dates = (
            None
            if target_dates is None
            else {pd.Timestamp(item).normalize() for item in target_dates}
        )
        role_map = {
            pd.Timestamp(key).normalize(): value
            for key, value in (period_role_by_target or {}).items()
        }
        for weekday_policy in weekday_policies:
            records = _normalized_history(self.history, weekday_policy)
            if requested_dates is not None:
                records = records[records[DATE_COL].isin(requested_dates)].reset_index(drop=True)
            for weather_policy in weather_policies:
                for record in records.itertuples(index=False):
                    target = pd.Timestamp(getattr(record, DATE_COL)).normalize()
                    actual = float(getattr(record, TARGET_COL))
                    scenarios = generate_forecast_scenarios(
                        self.history,
                        target,
                        weekday_policy=weekday_policy,
                    )
                    for scenario in scenarios:
                        segment = self.segment_for_target(target)
                        fit = self._fit_fold(
                            forecast_origin=scenario.forecast_origin,
                            segment=segment,
                            weather_policy=weather_policy,
                            weekday_policy=weekday_policy,
                        )
                        if fit is None:
                            skipped_rows.append(
                                {
                                    "target_date": target,
                                    "forecast_origin": scenario.forecast_origin,
                                    "scenario": scenario.scenario,
                                    "weather_policy": weather_policy,
                                    "weekday_policy": weekday_policy,
                                    "reason": f"fewer than {self.min_train_size} segment training rows",
                                }
                            )
                            continue

                        built = self.build_target_features(
                            target,
                            scenario.forecast_origin,
                            weather_policy=weather_policy,
                            weekday_policy=weekday_policy,
                            calendar_days_ahead=scenario.calendar_days_ahead,
                            service_horizon=scenario.service_horizon,
                        )
                        raw = built.features.to_frame().T
                        transformed = fit["imputer"].transform(raw[self.feature_cols])
                        point = apply_calibration_placeholder(
                            float(fit["point_model"].predict(transformed)[0])
                        )
                        quantile = float(fit["quantile_model"].predict(transformed)[0])
                        if not np.isfinite(point) or not np.isfinite(quantile):
                            raise AssertionError("Non-finite fitted prediction")
                        residual = float(self.residual_buffer_by_day.get(segment, 0.0))
                        replay = int(
                            np.ceil(
                                max(
                                    point * (1.0 + self.default_meal_buffer_pct),
                                    quantile,
                                    point + residual,
                                )
                            )
                        )
                        baselines = origin_available_baselines(
                            self.history,
                            target,
                            scenario.forecast_origin,
                            weekday_policy=weekday_policy,
                        )
                        feature_missing = built.features.isna()
                        attendance_missing = feature_missing.reindex(
                            self.attendance_feature_cols, fill_value=False
                        )
                        weather_missing = feature_missing.reindex(WEATHER_FEATURES).fillna(True)
                        legacy_values = legacy.get(target, (np.nan, np.nan))
                        point_error = point - actual
                        row = {
                            "feature_set_id": self.feature_set_id,
                            "period_role": role_map.get(target, "unassigned"),
                            "forecast_origin": scenario.forecast_origin,
                            "target_date": target,
                            "scenario": scenario.scenario,
                            "calendar_days_ahead": scenario.calendar_days_ahead,
                            "calendar_days_bucket": calendar_days_bucket(scenario.calendar_days_ahead),
                            "service_horizon": scenario.service_horizon,
                            "day_type": "Sunday" if target.weekday() == 6 else "Saturday",
                            "actual_weekday": target.day_name(),
                            "model_segment": segment,
                            "actual": actual,
                            "point_prediction": point,
                            "quantile_prediction": quantile,
                            "weather_policy": weather_policy,
                            "weekday_policy": weekday_policy,
                            "training_end_date": fit["training_end_date"],
                            "training_row_count": fit["training_row_count"],
                            "segment_training_row_count": fit["segment_training_row_count"],
                            "training_window_id": fit["training_window_id"],
                            "configured_window_rows": fit["configured_window_rows"],
                            "available_segment_training_rows": fit[
                                "available_segment_training_rows"
                            ],
                            "retained_segment_training_rows": fit[
                                "retained_segment_training_rows"
                            ],
                            "window_constrained": fit["window_constrained"],
                            "effective_window_days": fit["effective_window_days"],
                            "effective_window_years": fit["effective_window_years"],
                            "retained_training_start_date": fit[
                                "retained_training_start_date"
                            ],
                            "retained_training_end_date": fit["retained_training_end_date"],
                            "point_error": point_error,
                            "absolute_error": abs(point_error),
                            "squared_error": point_error**2,
                            "absolute_percentage_error": abs(point_error / actual),
                            "quantile_covers": bool(actual <= quantile),
                            "quantile_excess_shortfall": quantile - actual,
                            "point_model_name": type(fit["point_model"]).__name__,
                            "quantile_model_name": type(fit["quantile_model"]).__name__,
                            "feature_count": int(len(self.feature_cols)),
                            "feature_missing_count": int(feature_missing.sum()),
                            "attendance_feature_missing_count": int(attendance_missing.sum()),
                            "feature_provenance_valid": bool(
                                all(item.get("origin_valid", True) for item in built.provenance)
                            ),
                            "random_seed": self.random_seed,
                            "weather_missing_count": int(weather_missing.sum()),
                            "weather_missing_rate": float(weather_missing.mean()),
                            "weather_imputed": bool(weather_missing.any()),
                            "preprocessing_id": fit["preprocessing_id"],
                            "production_rule_replay_meals": replay,
                            "production_rule_replay_covers": bool(actual <= replay),
                            "production_rule_replay_excess_shortfall": replay - actual,
                            "residual_buffer_replay": residual,
                            "meal_buffer_pct_replay": self.default_meal_buffer_pct,
                            "legacy_point_prediction": legacy_values[0],
                            "legacy_quantile_prediction": legacy_values[1],
                            **baselines,
                        }
                        prediction_rows.append(row)

                        for item in built.provenance:
                            if item["feature"] not in self.feature_cols:
                                continue
                            item_payload = {
                                key: value
                                for key, value in item.items()
                                if key
                                not in {
                                    "forecast_origin",
                                    "target_date",
                                    "target_day_type",
                                }
                            }
                            provenance_rows.append(
                                {
                                    "feature_set_id": self.feature_set_id,
                                    "forecast_origin": scenario.forecast_origin,
                                    "target_date": target,
                                    "scenario": scenario.scenario,
                                    "weather_policy": weather_policy,
                                    "weekday_policy": weekday_policy,
                                    "day_type": row["day_type"],
                                    "raw_feature_value": built.features[item["feature"]],
                                    "imputed": bool(pd.isna(built.features[item["feature"]])),
                                    **item_payload,
                                }
                            )

        predictions = pd.DataFrame(prediction_rows)
        if predictions.empty:
            raise ValueError("Origin-aware backtest produced no predictions")
        key_cols = [
            "feature_set_id",
            "target_date",
            "scenario",
            "weather_policy",
            "weekday_policy",
        ]
        if predictions.duplicated(key_cols).any():
            duplicates = predictions.loc[predictions.duplicated(key_cols, keep=False), key_cols]
            raise AssertionError(f"Duplicate prediction keys:\n{duplicates.to_string(index=False)}")
        if (predictions["forecast_origin"] >= predictions["target_date"]).any():
            raise AssertionError("Every forecast origin must precede its target")
        if (predictions["training_end_date"] > predictions["forecast_origin"]).any():
            raise AssertionError("A preprocessing/model fold includes targets after the forecast origin")

        provenance = pd.DataFrame(provenance_rows)
        if not provenance.empty:
            provenance["source_dates"] = provenance["source_dates"].map(json.dumps)
            provenance["available_source_dates"] = provenance["available_source_dates"].map(json.dumps)
            provenance["available_source_values"] = provenance["available_source_values"].map(json.dumps)
            provenance["withheld_source_dates"] = provenance["withheld_source_dates"].map(json.dumps)
        diagnostics = pd.DataFrame(self._preprocessing_rows).drop_duplicates("preprocessing_id feature".split())
        return OriginBacktestResult(
            predictions=predictions.sort_values(key_cols, kind="stable").reset_index(drop=True),
            preprocessing_diagnostics=diagnostics.sort_values(
                ["weekday_policy", "weather_policy", "segment", "training_end_date", "feature"],
                kind="stable",
            ).reset_index(drop=True),
            feature_provenance=provenance,
            skipped_folds=pd.DataFrame(skipped_rows),
        )

    def point_feature_importance_rows(self) -> pd.DataFrame:
        """Return Random Forest impurity importances from cached expanding folds."""

        rows: list[dict[str, Any]] = []
        for cache_key, fit in self._fit_cache.items():
            weekday_policy, weather_policy, segment = cache_key[:3]
            segment_count = int(fit["retained_segment_training_rows"])
            importances = getattr(fit["point_model"], "feature_importances_", None)
            if importances is None:
                continue
            for feature, importance in zip(self.feature_cols, importances, strict=True):
                rows.append(
                    {
                        "feature_set_id": self.feature_set_id,
                        "weekday_policy": weekday_policy,
                        "weather_policy": weather_policy,
                        "segment": segment,
                        "training_window_id": fit["training_window_id"],
                        "segment_training_row_count": int(segment_count),
                        "training_end_date": fit["training_end_date"],
                        "feature": feature,
                        "rf_impurity_importance": float(importance),
                    }
                )
        return pd.DataFrame(rows)


def metric_breakdowns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form metrics table across all required diagnostic slices."""

    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    valid_actuals = frame.drop_duplicates("target_date").sort_values("target_date")
    valid_actuals = valid_actuals[valid_actuals["actual_weekday"].isin(["Saturday", "Sunday"])]
    recent_dates = set(valid_actuals.tail(52)["target_date"])
    frame["period"] = np.where(frame["target_date"].isin(recent_dates), "Recent 52", "Earlier")
    frame["year"] = frame["target_date"].dt.year.astype(str)
    frame["quarter"] = frame["target_date"].dt.to_period("Q").astype(str)
    quartile_source = valid_actuals["actual"]
    boundaries = quartile_source.quantile([0.25, 0.5, 0.75]).to_list()
    frame["attendance_quartile"] = pd.cut(
        frame["actual"],
        bins=[-np.inf, *boundaries, np.inf],
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        include_lowest=True,
    ).astype(str)

    dimensions = [
        ("overall", None),
        ("day_type", "day_type"),
        ("scenario", "scenario"),
        ("service_horizon", "service_horizon"),
        ("calendar_days_bucket", "calendar_days_bucket"),
        ("year", "year"),
        ("quarter", "quarter"),
        ("period", "period"),
        ("attendance_quartile", "attendance_quartile"),
    ]
    rows: list[dict[str, Any]] = []
    policy_groups = frame.groupby(["weekday_policy", "weather_policy"], dropna=False, sort=True)
    for (weekday_policy, weather_policy), policy_frame in policy_groups:
        for breakdown, column in dimensions:
            groups = [("All", policy_frame)] if column is None else policy_frame.groupby(column, dropna=False, sort=True)
            for value, part in groups:
                metrics = calculate_metrics(part)
                rows.append(
                    {
                        "model": "origin_aware_current_model",
                        "weekday_policy": weekday_policy,
                        "weather_policy": weather_policy,
                        "breakdown": breakdown,
                        "breakdown_value": str(value),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)
