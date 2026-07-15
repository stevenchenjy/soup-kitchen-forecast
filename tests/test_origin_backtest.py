from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import src.origin_backtest as origin_backtest
from src.origin_backtest import OriginAwareBacktester, generate_forecast_scenarios
from src.origin_features import T0_LEGACY_ALL, T1_VALID_WEEKENDS, W0_NO_WEATHER


def long_history() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2023-08-31", freq="D")
    dates = dates[dates.weekday.isin([5, 6])]
    visitors = 90 + np.arange(len(dates), dtype=float)
    return pd.DataFrame({"service_date": dates, "visitors": visitors})


def patch_fast_models(monkeypatch) -> None:
    monkeypatch.setattr(origin_backtest, "make_point_model", lambda: DummyRegressor(strategy="mean"))
    monkeypatch.setattr(
        origin_backtest,
        "make_quantile_model",
        lambda quantile: DummyRegressor(strategy="quantile", quantile=quantile),
    )


def test_scenario_origins_and_horizons_are_deterministic() -> None:
    history = long_history()
    saturday = {
        item.scenario: item
        for item in generate_forecast_scenarios(
            history, "2023-08-19", weekday_policy=T1_VALID_WEEKENDS
        )
    }
    sunday = {
        item.scenario: item
        for item in generate_forecast_scenarios(
            history, "2023-08-20", weekday_policy=T1_VALID_WEEKENDS
        )
    }
    assert saturday["S1_same_weekend_saturday"].forecast_origin == pd.Timestamp("2023-08-18")
    assert sunday["S2_same_weekend_sunday"].forecast_origin == pd.Timestamp("2023-08-18")
    assert sunday["S2_same_weekend_sunday"].service_horizon == 2
    assert saturday["S4_two_service_ahead"].service_horizon == 2
    assert saturday["S5_longest_supported"].calendar_days_ahead == 15
    assert saturday["S5_longest_supported"].service_horizon == 5


def test_tuesday_gets_t0_next_service_only_and_no_t1_scenario() -> None:
    history = long_history()
    history.loc[len(history)] = [pd.Timestamp("2023-09-05"), 141.0]
    t0 = generate_forecast_scenarios(history, "2023-09-05", weekday_policy=T0_LEGACY_ALL)
    t1 = generate_forecast_scenarios(history, "2023-09-05", weekday_policy=T1_VALID_WEEKENDS)
    assert [item.scenario for item in t0] == ["S3_next_service"]
    assert t1 == []


def test_fold_imputer_is_training_only(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    history = long_history()
    backtester = OriginAwareBacktester(history, weather_df=None, min_train_size=3)
    origin = pd.Timestamp("2023-02-12")
    fit = backtester._fit_fold(
        forecast_origin=origin,
        segment="sat",
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
    )
    assert fit is not None
    training = backtester._build_training_frame(T1_VALID_WEEKENDS, W0_NO_WEATHER)
    training = training[(training["service_date"] <= origin) & (training["segment"] == "sat")]
    expected = training["lag_same_daytype_1"].median()
    diagnostic = next(
        row
        for row in backtester._preprocessing_rows
        if row["preprocessing_id"] == fit["preprocessing_id"]
        and row["feature"] == "lag_same_daytype_1"
    )
    assert diagnostic["imputer_statistic"] == expected
    assert diagnostic["training_end_date"] <= origin
    assert diagnostic["fit_includes_test_or_future"] is False


def test_direct_and_backtest_feature_paths_are_identical(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    history = long_history()
    backtester = OriginAwareBacktester(history, weather_df=None, min_train_size=3)
    direct = origin_backtest.build_features_as_of(
        history,
        "2023-08-20",
        "2023-08-18",
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
    )
    through_backtester = backtester.build_target_features(
        "2023-08-20",
        "2023-08-18",
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
    )
    pd.testing.assert_series_equal(direct.features, through_backtester.features)
    assert direct.provenance == through_backtester.provenance


def test_backtest_keys_are_unique_and_boundaries_are_origin_safe(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    history = long_history()
    backtester = OriginAwareBacktester(history, weather_df=None, min_train_size=3)
    result = backtester.run(
        weather_policies=[W0_NO_WEATHER],
        weekday_policies=[T1_VALID_WEEKENDS],
    )
    keys = ["target_date", "scenario", "weather_policy", "weekday_policy"]
    assert not result.predictions.duplicated(keys).any()
    assert (result.predictions["forecast_origin"] < result.predictions["target_date"]).all()
    assert (result.predictions["training_end_date"] <= result.predictions["forecast_origin"]).all()
    source_date_columns = ["available_source_dates"]
    attendance = result.feature_provenance[result.feature_provenance["source_type"] == "attendance"]
    for row in attendance.itertuples(index=False):
        for source_date in __import__("json").loads(getattr(row, source_date_columns[0])):
            assert pd.Timestamp(source_date) <= pd.Timestamp(row.forecast_origin)
