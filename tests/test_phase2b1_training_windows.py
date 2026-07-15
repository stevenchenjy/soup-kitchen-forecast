from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import src.origin_backtest as origin_backtest
from scripts.run_phase2b1_training_windows import json_clean
from src.origin_backtest import OriginAwareBacktester
from src.origin_features import T1_VALID_WEEKENDS, W0_NO_WEATHER
from src.training_windows import TW_26, TW_EXPANDING


def history() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    dates = dates[dates.weekday.isin([5, 6])]
    return pd.DataFrame(
        {
            "service_date": dates,
            "visitors": 100 + np.arange(len(dates), dtype=float) % 60,
        }
    )


def patch_fast_models(monkeypatch) -> None:
    monkeypatch.setattr(origin_backtest, "make_point_model", lambda: DummyRegressor(strategy="mean"))
    monkeypatch.setattr(
        origin_backtest,
        "make_quantile_model",
        lambda quantile: DummyRegressor(strategy="quantile", quantile=quantile),
    )


def fit(backtester: OriginAwareBacktester):
    return backtester._fit_fold(
        forecast_origin=pd.Timestamp("2024-12-20"),
        segment="sat",
        weather_policy=W0_NO_WEATHER,
        weekday_policy=T1_VALID_WEEKENDS,
    )


def test_default_and_explicit_expanding_folds_are_identical(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    default = OriginAwareBacktester(history(), weather_df=None, min_train_size=18)
    explicit = OriginAwareBacktester(
        history(), weather_df=None, min_train_size=18, training_window=TW_EXPANDING
    )
    default_fit = fit(default)
    explicit_fit = fit(explicit)
    assert default_fit is not None and explicit_fit is not None
    for key in [
        "training_end_date",
        "training_row_count",
        "segment_training_row_count",
        "available_segment_training_rows",
        "retained_segment_training_rows",
        "effective_window_days",
    ]:
        assert default_fit[key] == explicit_fit[key]
    default_frame = default._build_training_frame(T1_VALID_WEEKENDS, W0_NO_WEATHER)
    row = default_frame[default_frame["service_date"] == pd.Timestamp("2024-12-21")].iloc[0]
    x = row[default.feature_cols].to_frame().T
    np.testing.assert_allclose(
        default_fit["point_model"].predict(default_fit["imputer"].transform(x)),
        explicit_fit["point_model"].predict(explicit_fit["imputer"].transform(x)),
    )


def test_finite_window_limits_only_fit_examples_and_reports_diagnostics(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    backtester = OriginAwareBacktester(
        history(), weather_df=None, min_train_size=18, training_window=TW_26
    )
    result = fit(backtester)
    assert result is not None
    assert result["available_segment_training_rows"] > 26
    assert result["retained_segment_training_rows"] == 26
    assert result["segment_training_row_count"] == 26
    assert result["window_constrained"] is True
    assert result["training_window_id"] == "TW_26"
    assert result["retained_training_end_date"] <= pd.Timestamp("2024-12-20")
    # The cached feature frame remains built for all historical targets; only fitting is sliced.
    full = backtester._build_training_frame(T1_VALID_WEEKENDS, W0_NO_WEATHER)
    assert len(full[full["segment"] == "sat"]) > result["retained_segment_training_rows"]
    diagnostic = backtester._preprocessing_rows[0]
    assert diagnostic["configured_window_rows"] == 26
    assert diagnostic["retained_segment_training_rows"] == 26


def test_finite_run_has_unique_keys_and_origin_safe_retained_dates(monkeypatch) -> None:
    patch_fast_models(monkeypatch)
    backtester = OriginAwareBacktester(
        history(), weather_df=None, min_train_size=18, training_window="TW_26"
    )
    targets = pd.date_range("2024-11-01", "2024-12-31", freq="D")
    targets = targets[targets.weekday.isin([5, 6])]
    result = backtester.run(
        weather_policies=[W0_NO_WEATHER],
        weekday_policies=[T1_VALID_WEEKENDS],
        target_dates=targets,
    )
    keys = ["target_date", "scenario", "weather_policy", "weekday_policy"]
    assert not result.predictions.duplicated(keys).any()
    assert set(result.predictions["training_window_id"]) == {"TW_26"}
    assert (result.predictions["retained_segment_training_rows"] <= 26).all()
    assert (
        result.predictions["retained_training_end_date"]
        <= result.predictions["forecast_origin"]
    ).all()


def test_machine_readable_lock_serialization_uses_json_null_for_expanding_rows() -> None:
    assert json_clean({"configured_window_rows": np.nan}) == {
        "configured_window_rows": None
    }
