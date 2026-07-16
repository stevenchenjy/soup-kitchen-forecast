from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import src.origin_backtest as origin_backtest
from src.feature_sets import (
    F0,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    FEATURE_SET_IDS,
    build_feature_set_registry,
    make_repaired_feature_builder,
    select_compact_f6,
    select_f5_parent,
)
from src.origin_backtest import OriginAwareBacktester, calculate_metrics
from src.origin_features import (
    CALENDAR_FEATURES,
    DAYTYPE_SLOT_FEATURES,
    DAYTYPE_SUMMARY_FEATURES,
    HORIZON_AWARE_FEATURES,
    LAST_OBSERVED_DAYTYPE_FEATURES,
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
    build_repaired_features_as_of,
)


ROOT = Path(__file__).resolve().parents[1]


def history_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2024-06-30", freq="D")
    dates = dates[dates.weekday.isin([5, 6])]
    values = 80.0 + np.arange(len(dates))
    return pd.DataFrame({"service_date": dates, "visitors": values})


def built(target: str, origin: str, features: list[str] | None = None):
    feature_list = features or (
        CALENDAR_FEATURES
        + LAST_OBSERVED_DAYTYPE_FEATURES
        + DAYTYPE_SUMMARY_FEATURES
        + DAYTYPE_SLOT_FEATURES
        + HORIZON_AWARE_FEATURES
    )
    target_ts = pd.Timestamp(target)
    origin_ts = pd.Timestamp(origin)
    service_horizon = sum(
        day.weekday() in {5, 6}
        for day in pd.date_range(origin_ts + pd.Timedelta(days=1), target_ts)
    )
    return build_repaired_features_as_of(
        history_frame(),
        target_ts,
        origin_ts,
        calendar_days_ahead=(target_ts - origin_ts).days,
        service_horizon=service_horizon,
        feature_cols=feature_list,
    )


def provenance(result, feature: str) -> dict:
    return next(item for item in result.provenance if item["feature"] == feature)


def test_last_observed_lags_select_only_matching_daytype() -> None:
    result = built("2024-06-30", "2024-06-28")
    source = provenance(result, "last_observed_daytype_1")
    assert pd.Timestamp(source["source_dates"][0]).weekday() == 6
    assert result.features["last_observed_daytype_1"] == history_frame().loc[
        history_frame().service_date == pd.Timestamp("2024-06-23"), "visitors"
    ].iloc[0]


def test_lag_sources_are_on_or_before_origin() -> None:
    result = built("2024-06-30", "2024-06-28")
    for item in result.provenance:
        if item["source_type"] == "attendance":
            assert all(pd.Timestamp(date) <= pd.Timestamp("2024-06-28") for date in item["source_dates"])
            assert item["origin_valid"] is True


def test_s2_sunday_excludes_target_weekend_saturday() -> None:
    result = built("2024-06-30", "2024-06-28")
    all_sources = [date for item in result.provenance for date in item["source_dates"]]
    assert "2024-06-29" not in all_sources


@pytest.mark.parametrize(
    ("target", "origin", "forbidden_start"),
    [
        ("2024-06-29", "2024-06-22", "2024-06-23"),
        ("2024-06-29", "2024-06-14", "2024-06-15"),
    ],
)
def test_s4_and_s5_exclude_intervening_future_services(
    target: str, origin: str, forbidden_start: str
) -> None:
    result = built(target, origin)
    all_sources = [pd.Timestamp(date) for item in result.provenance for date in item["source_dates"]]
    assert all(date < pd.Timestamp(forbidden_start) for date in all_sources)


def test_last_observed_lag_meaning_is_scenario_invariant() -> None:
    friday = built("2024-06-30", "2024-06-28")
    earlier = built("2024-06-30", "2024-06-22")
    for result, origin in [(friday, "2024-06-28"), (earlier, "2024-06-22")]:
        item = provenance(result, "last_observed_daytype_1")
        assert item["aggregation"] == "ranked_last_observed"
        assert item["rank_or_window"] == "rank_1"
        assert pd.Timestamp(item["source_dates"][0]) <= pd.Timestamp(origin)


def test_last_four_median_and_mean_match_manual_calculation() -> None:
    result = built("2024-06-30", "2024-06-28")
    history = history_frame()
    expected = history[
        (history.service_date.dt.weekday == 6)
        & (history.service_date <= pd.Timestamp("2024-06-28"))
    ].tail(4).visitors
    assert result.features["daytype_median_last_4"] == expected.median()
    assert result.features["daytype_mean_last_4"] == expected.mean()


def test_insufficient_history_is_missing_with_reason() -> None:
    history = history_frame().head(2)
    result = build_repaired_features_as_of(
        history,
        "2023-01-08",
        "2023-01-06",
        calendar_days_ahead=2,
        service_horizon=2,
        feature_cols=["last_observed_daytype_6"],
    )
    assert pd.isna(result.features.iloc[0])
    assert "found 1" in provenance(result, "last_observed_daytype_6")["missing_reason"]


def test_same_slot_features_match_daytype_and_slot() -> None:
    result = built("2024-06-30", "2024-06-28")
    for date in provenance(result, "daytype_slot_last_observed")["source_dates"]:
        date = pd.Timestamp(date)
        assert date.weekday() == 6
        assert ((date.day - 1) // 7 + 1) == 5


def test_same_slot_provenance_contains_only_valid_matches() -> None:
    result = built("2024-06-30", "2024-06-28")
    item = provenance(result, "daytype_slot_median_last_3")
    assert item["origin_valid"] is True
    assert item["used_source_count"] <= 3
    assert all(pd.Timestamp(date) <= pd.Timestamp("2024-06-28") for date in item["source_dates"])


def test_horizon_features_match_supplied_phase1_definition() -> None:
    result = built("2024-06-30", "2024-06-28")
    assert result.features["calendar_days_ahead"] == 2
    assert result.features["service_horizon"] == 2
    assert result.features["future_eligible_services_between"] == 1


def test_missing_indicators_match_raw_lags() -> None:
    result = built("2023-01-08", "2023-01-06")
    for rank in [1, 2, 3, 4, 6]:
        raw = result.features[f"last_observed_daytype_{rank}"]
        indicator = result.features[f"missing_last_observed_daytype_{rank}"]
        assert indicator == float(pd.isna(raw))


def test_calendar_only_has_no_attendance_features_or_sources() -> None:
    result = built("2024-06-30", "2024-06-28", list(CALENDAR_FEATURES))
    assert result.features.index.tolist() == CALENDAR_FEATURES
    assert all(item["source_type"] == "target_calendar" for item in result.provenance)


def test_registry_ids_unique_and_stable() -> None:
    registry = build_feature_set_registry()
    assert tuple(registry) == FEATURE_SET_IDS
    assert len(registry) == len(set(registry)) == 7


def test_registry_parent_child_differences_are_exact() -> None:
    registry = build_feature_set_registry(f5_parent_id=F3)
    assert set(registry[F2].feature_list) - set(registry[F1].feature_list) == set(
        LAST_OBSERVED_DAYTYPE_FEATURES
    )
    assert set(registry[F3].feature_list) - set(registry[F2].feature_list) == set(
        DAYTYPE_SUMMARY_FEATURES
    )
    assert set(registry[F4].feature_list) - set(registry[F3].feature_list) == set(
        DAYTYPE_SLOT_FEATURES
    )
    assert set(registry[F5].feature_list) - set(registry[F3].feature_list) == set(
        HORIZON_AWARE_FEATURES
    )


def test_every_registered_repaired_feature_is_generated() -> None:
    registry = build_feature_set_registry()
    for feature_set_id in [F1, F2, F3, F4, F5, F6]:
        definition = registry[feature_set_id]
        result = built("2024-06-30", "2024-06-28", list(definition.feature_list))
        assert result.features.index.tolist() == list(definition.feature_list)


def test_f5_parent_selection_uses_macro_then_s2_then_simplicity() -> None:
    frame = pd.DataFrame(
        [
            {"feature_set_id": F2, "development_macro_mae": 10.0, "development_s2_mae": 11.0, "feature_count": 15},
            {"feature_set_id": F3, "development_macro_mae": 10.0, "development_s2_mae": 10.5, "feature_count": 27},
            {"feature_set_id": F4, "development_macro_mae": 10.2, "development_s2_mae": 9.0, "feature_count": 32},
        ]
    )
    assert select_f5_parent(frame) == F3


def test_compact_selection_uses_only_predeclared_group_thresholds() -> None:
    frame = pd.DataFrame(
        [
            {"feature_set_id": F1, "development_macro_mae": 16.0, "development_s2_mae": 18.0},
            {"feature_set_id": F2, "development_macro_mae": 15.8, "development_s2_mae": 18.1},
            {"feature_set_id": F3, "development_macro_mae": 15.79, "development_s2_mae": 18.0},
            {"feature_set_id": F4, "development_macro_mae": 15.6, "development_s2_mae": 17.9},
            {"feature_set_id": F5, "development_macro_mae": 15.4, "development_s2_mae": 18.0},
        ]
    )
    features, groups, decisions = select_compact_f6(frame, f5_parent_id=F4)
    assert "last_observed_daytype" in groups
    assert "daytype_summaries" not in groups
    assert "daytype_slot" in groups
    assert "horizon_availability" in groups
    assert all("supported" in item for item in decisions)
    assert len(features) == len(set(features))


def test_generic_backtester_keeps_fold_preprocessing_isolated(monkeypatch) -> None:
    monkeypatch.setattr(origin_backtest, "make_point_model", lambda: DummyRegressor(strategy="mean"))
    monkeypatch.setattr(
        origin_backtest,
        "make_quantile_model",
        lambda quantile: DummyRegressor(strategy="quantile", quantile=quantile),
    )
    definition = build_feature_set_registry()[F2]
    evaluator = OriginAwareBacktester(
        history_frame(),
        weather_df=None,
        feature_cols=definition.feature_list,
        feature_set_id=F2,
        feature_builder=make_repaired_feature_builder(definition),
        attendance_feature_cols=LAST_OBSERVED_DAYTYPE_FEATURES,
        min_train_size=3,
    )
    fit = evaluator._fit_fold(
        forecast_origin=pd.Timestamp("2023-04-01"),
        segment="sat",
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
    )
    assert fit is not None
    assert fit["training_end_date"] <= pd.Timestamp("2023-04-01")
    assert all(row["fit_includes_test_or_future"] is False for row in evaluator._preprocessing_rows)


def test_generic_prediction_keys_are_unique_and_provenance_is_origin_safe(monkeypatch) -> None:
    monkeypatch.setattr(origin_backtest, "make_point_model", lambda: DummyRegressor(strategy="mean"))
    monkeypatch.setattr(
        origin_backtest,
        "make_quantile_model",
        lambda quantile: DummyRegressor(strategy="quantile", quantile=quantile),
    )
    definition = build_feature_set_registry()[F2]
    evaluator = OriginAwareBacktester(
        history_frame(),
        weather_df=None,
        feature_cols=definition.feature_list,
        feature_set_id=F2,
        feature_builder=make_repaired_feature_builder(definition),
        attendance_feature_cols=LAST_OBSERVED_DAYTYPE_FEATURES,
        min_train_size=3,
    )
    result = evaluator.run(
        weather_policies=[W0_NO_WEATHER], weekday_policies=[T1_VALID_WEEKENDS]
    )
    keys = ["feature_set_id", "forecast_origin", "target_date", "scenario", "weather_policy", "weekday_policy"]
    assert not result.predictions.duplicated(keys).any()
    assert result.predictions.feature_provenance_valid.all()
    for row in result.feature_provenance.itertuples(index=False):
        for source_date in json.loads(row.source_dates):
            assert pd.Timestamp(source_date) <= pd.Timestamp(row.forecast_origin)


@pytest.mark.skipif(
    not (
        ROOT
        / "artifacts/ny_12550/model_optimization/phase1_origin_backtest"
        / "05_origin_aware_predictions.csv"
    ).is_file(),
    reason="The F0 reproduction gate requires ignored Phase 1 artifacts",
)
def test_phase1_reference_artifact_reproduces_required_f0_gate() -> None:
    predictions = pd.read_csv(
        ROOT / "artifacts/ny_12550/model_optimization/phase1_origin_backtest/05_origin_aware_predictions.csv"
    )
    preferred = predictions[
        (predictions.weekday_policy == "T1_valid_weekends")
        & (predictions.weather_policy == "W0_no_weather")
    ]
    metrics = calculate_metrics(preferred)
    assert metrics["row_count"] == 1256
    assert metrics["mae"] == pytest.approx(16.277079645681486, abs=1e-12)
    assert metrics["rmse"] == pytest.approx(20.547669609537795, abs=1e-12)
    assert metrics["mean_signed_error"] == pytest.approx(-3.7132279301892788, abs=1e-12)
    assert metrics["raw_quantile_coverage"] == pytest.approx(0.6138535031847133, abs=1e-12)
