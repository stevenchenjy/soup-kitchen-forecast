from __future__ import annotations

import pandas as pd
import pytest

from src.origin_features import (
    ATTENDANCE_FEATURES,
    MODEL_FEATURES,
    T0_LEGACY_ALL,
    T1_VALID_WEEKENDS,
    W0_NO_WEATHER,
    W1_OBSERVED_REPLAY,
    WEATHER_FEATURES,
    apply_calibration_placeholder,
    build_features_as_of,
    origin_available_baselines,
)


def history_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-06", periods=24, freq="D")
    dates = dates[dates.weekday.isin([5, 6])]
    return pd.DataFrame({"service_date": dates, "visitors": range(100, 100 + len(dates))})


def weather_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-21"],
            "temp_10_13": [11.0],
            "apparent_temp_10_13": [10.0],
            "humidity_10_13": [55.0],
            "wind_10_13": [7.0],
            "precip_10_13": [0.2],
        }
    )


def provenance_for(result, feature: str) -> dict:
    return next(item for item in result.provenance if item["feature"] == feature)


def test_sunday_before_weekend_excludes_saturday_and_post_origin_attendance() -> None:
    history = history_frame()
    result = build_features_as_of(
        history,
        "2024-01-21",
        "2024-01-19",
        weekday_policy=T1_VALID_WEEKENDS,
    )
    lag = provenance_for(result, "lag1")
    assert pd.isna(result.features["lag1"])
    assert lag["withheld_source_dates"] == ["2024-01-20"]
    assert lag["available_source_values"] == []
    assert lag["status"] == "unavailable_at_origin"


def test_target_attendance_is_never_used() -> None:
    history = history_frame()
    result = build_features_as_of(history, "2024-01-21", "2024-01-20")
    assert result.features["lag1"] == history.loc[
        history["service_date"] == pd.Timestamp("2024-01-20"), "visitors"
    ].iloc[0]
    assert "2024-01-21" not in provenance_for(result, "lag1")["source_dates"]


def test_two_service_ahead_excludes_intervening_attendance() -> None:
    history = history_frame()
    result = build_features_as_of(history, "2024-01-20", "2024-01-13")
    lag = provenance_for(result, "lag1")
    assert pd.isna(result.features["lag1"])
    assert lag["withheld_source_dates"] == ["2024-01-14"]


def test_output_is_deterministic_and_does_not_mutate_input() -> None:
    history = history_frame()
    before = history.copy(deep=True)
    first = build_features_as_of(history, "2024-01-21", "2024-01-19")
    second = build_features_as_of(history, "2024-01-21", "2024-01-19")
    pd.testing.assert_frame_equal(history, before)
    pd.testing.assert_series_equal(first.features, second.features)
    assert first.provenance == second.provenance


def test_weather_w0_is_missing_and_w1_replays_observation() -> None:
    history = history_frame()
    w0 = build_features_as_of(
        history,
        "2024-01-21",
        "2024-01-19",
        weather_policy=W0_NO_WEATHER,
        weather_df=weather_frame(),
    )
    w1 = build_features_as_of(
        history,
        "2024-01-21",
        "2024-01-19",
        weather_policy=W1_OBSERVED_REPLAY,
        weather_df=weather_frame(),
    )
    assert w0.features[WEATHER_FEATURES].isna().all()
    assert w1.features[WEATHER_FEATURES].tolist() == [11.0, 10.0, 55.0, 7.0, 0.2]
    assert provenance_for(w1, "temp_10_13")["weather_issue_date"] == "unavailable"


def test_all_current_features_are_returned_with_attendance_provenance() -> None:
    result = build_features_as_of(history_frame(), "2024-01-21", "2024-01-19")
    assert result.features.index.tolist() == MODEL_FEATURES
    assert {item["feature"] for item in result.provenance} == set(MODEL_FEATURES)
    for feature in ATTENDANCE_FEATURES:
        assert provenance_for(result, feature)["source_type"] == "attendance"


def test_baselines_use_only_origin_available_records() -> None:
    history = history_frame()
    values = origin_available_baselines(history, "2024-01-21", "2024-01-19")
    expected = history.loc[history["service_date"] == pd.Timestamp("2024-01-14"), "visitors"].iloc[0]
    assert values["previous_same_daytype"] == expected
    changed = history.copy()
    changed.loc[changed["service_date"] > pd.Timestamp("2024-01-19"), "visitors"] = 9999
    assert origin_available_baselines(changed, "2024-01-21", "2024-01-19") == values


def test_calibration_placeholder_is_identity_and_rejects_phase2_work() -> None:
    assert apply_calibration_placeholder(123.4) == 123.4
    with pytest.raises(ValueError, match="outside Phase 1"):
        apply_calibration_placeholder(123.4, calibration={"offset": 1})


def test_t0_allows_tuesday_history_while_t1_excludes_it() -> None:
    history = history_frame()
    history.loc[len(history)] = [pd.Timestamp("2024-01-16"), 999]
    t0 = build_features_as_of(
        history,
        "2024-01-20",
        "2024-01-16",
        weekday_policy=T0_LEGACY_ALL,
    )
    t1 = build_features_as_of(
        history,
        "2024-01-20",
        "2024-01-16",
        weekday_policy=T1_VALID_WEEKENDS,
    )
    assert t0.features["lag1"] == 999
    assert t1.features["lag1"] != 999
