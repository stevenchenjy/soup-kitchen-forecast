from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.config import DATE_COL, TARGET_COL
from src.production_features import LOCKED_F6_FEATURES, build_locked_f6_feature_row, build_locked_f6_training_frame
from src import weather_candidate as candidate
from scripts import train_weather_candidate as trainer
from src.weather_snapshots import HOURLY_UNITS


class SpyModel:
    def __init__(self, value=123.25):
        self.n_features_in_ = 39
        self.value = value
        self.seen = []

    def predict(self, values):
        self.seen.append(np.array(values, copy=True))
        return np.array([self.value])


def spy_fit(frame, feature_cols, quantile, return_preprocessors):
    assert feature_cols == list(candidate.WEATHER_CANDIDATE_FEATURES)
    assert quantile == 0.8 and return_preprocessors is True
    models, quantiles, preprocessors = {}, {}, {}
    for segment, is_sun in (("sat", 0), ("sun", 1)):
        models[segment] = SpyModel(100.25 if segment == "sat" else 200.25)
        quantiles[segment] = SpyModel(120.25 if segment == "sat" else 220.25)
        preprocessors[segment] = SimpleImputer(strategy="median", keep_empty_features=True).fit(
            frame.loc[frame["is_sun"] == is_sun, feature_cols]
        )
    return models, quantiles, preprocessors


@pytest.fixture
def history():
    dates = pd.date_range("2025-01-04", periods=180, freq="D")
    dates = dates[dates.weekday.isin([5, 6])][:48]
    return pd.DataFrame({DATE_COL: dates, TARGET_COL: np.arange(len(dates), dtype=float) + 100})


@pytest.fixture
def weather(history):
    frame = pd.DataFrame({DATE_COL: history[DATE_COL]})
    for name, value in zip(candidate.WEATHER_FEATURES, (1., 5., 2., 25., 0.5, 0.01)):
        frame[name] = value
    return frame


def build_package(history, weather):
    with patch.object(candidate, "fit_final_models_by_daytype", side_effect=spy_fit):
        return candidate.build_weather_candidate_package(
            location_id="ny_12550", attendance=history, weather_features=weather,
            package_id="ny_12550_weather_test_v1",
            baseline_source={"sha256": "a" * 64, "package_id": "f6_v1"},
            weather_input={"sha256": "b" * 64, "data_kind": candidate.TRAINING_WEATHER},
            weather_context={"zip_code": "12550", "country_code": "US", "timezone": "America/New_York",
                             "latitude": 41.5034, "longitude": -74.0104},
        )


@pytest.fixture
def package(history, weather):
    return build_package(history, weather)


def test_fit_keeps_locked_f6_training_rows_and_records_bootstrap_provenance(history, weather):
    original = history.copy(deep=True)
    expected = build_locked_f6_training_frame(history)
    def fit(frame, *args, **kwargs):
        pd.testing.assert_frame_equal(frame[expected.df.columns], expected.df)
        assert not frame[list(candidate.WEATHER_FEATURES)].isna().any().any()
        return spy_fit(frame, *args, **kwargs)
    started = datetime.now(timezone.utc)
    with patch.object(candidate, "fit_final_models_by_daytype", side_effect=fit):
        package = candidate.build_weather_candidate_package(
            location_id="ny_12550", attendance=history, weather_features=weather,
            package_id="weather_fit_v1", baseline_source={"sha256": "a" * 64},
            weather_input={"sha256": "b" * 64},
            weather_context={"zip_code": "12550", "country_code": "US", "timezone": "America/New_York",
                             "latitude": 41.5034, "longitude": -74.0104},
        )
    pd.testing.assert_frame_equal(history, original)
    assert started <= datetime.fromisoformat(package["created_at_utc"]) <= datetime.now(timezone.utc)
    assert package["training_weather"] == "realized_historical_bootstrap"
    assert package["evaluation"]["bootstrap_is_archived_forecast_validation"] is False
    assert package["model_package_schema_version"] == 0
    assert package["baseline_model_sha256"] == "a" * 64
    assert package["training_end_date"] == history[DATE_COL].max().date().isoformat()
    assert package["history"]["sha256"] == candidate.history_sha256(history)
    assert package["activation"] == {"active_model_changed": False, "automatic_activation_allowed": False}


@pytest.mark.parametrize("target,expected_point,expected_meals,segment", [
    ("2025-08-02", 100.25, 121, "sat"), ("2025-08-03", 200.25, 221, "sun"),
])
def test_shared_f6_row_is_preserved_and_correct_segment_predicts(package, weather, target, expected_point, expected_meals, segment):
    x_f6 = build_locked_f6_feature_row(package["history_df"], target, "2025-08-01")
    before = x_f6.copy(deep=True)
    features = weather.iloc[0].to_dict()
    combined = candidate.build_weather_candidate_feature_row(package["history_df"], target, "2025-08-01", features)
    pd.testing.assert_frame_equal(combined[list(LOCKED_F6_FEATURES)], x_f6.astype(float))
    result = candidate.predict_weather_candidate(package, x_f6, features, target)
    pd.testing.assert_frame_equal(x_f6, before)
    expected = package["preprocessors"][segment].transform(combined)
    np.testing.assert_allclose(package["models"][segment].seen[-1], expected)
    other = "sun" if segment == "sat" else "sat"
    assert package["models"][other].seen == []
    assert result["point"] == expected_point
    assert result["suggested_meals"] == expected_meals
    assert result["package_status"] == candidate.PACKAGE_STATUS


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(model_package_schema_version=2),
    lambda p: p.update(feature_cols=list(reversed(p["feature_cols"]))),
    lambda p: p["models"].pop("sun"),
    lambda p: setattr(p["models"]["sat"], "n_features_in_", 33),
    lambda p: p["preprocessors"]["sat"].statistics_.__setitem__(0, np.inf),
    lambda p: p.update(location_id="boston_02108"),
    lambda p: p.update(history_df=p["history_df"].iloc[:-1]),
    lambda p: p.update(metadata={}),
    lambda p: p["activation"].update(automatic_activation_allowed=True),
    lambda p: p["feature_contract"].update(feature_order_sha256="0" * 64),
])
def test_malformed_candidate_rejected_before_prediction(package, mutation):
    mutation(package)
    with pytest.raises(ValueError):
        candidate.validate_weather_candidate(package)


def test_location_mismatch_rejected(package):
    with pytest.raises(ValueError, match="location"):
        candidate.validate_weather_candidate(package, expected_location_id="boston_02108")


@pytest.mark.parametrize("fault", ["missing", "infinite", "negative", "inverted_temperatures", "boolean"])
def test_weather_is_never_imputed(package, weather, fault):
    features = weather.iloc[0].to_dict()
    if fault == "missing":
        del features["wx_snowdepth_max_m"]
    elif fault == "infinite":
        features["wx_gust_max_kmh"] = np.inf
    elif fault == "negative":
        features["wx_snowfall_cm"] = -1
    elif fault == "boolean":
        features["wx_precip_mm"] = True
    else:
        features["wx_apparent_min_c"] = 99
    x = build_locked_f6_feature_row(package["history_df"], "2025-08-02", "2025-08-01")
    with pytest.raises(ValueError):
        candidate.predict_weather_candidate(package, x, features, "2025-08-02")
    assert package["models"]["sat"].seen == []


def test_bad_transformed_values_and_predictions_rejected(package, weather):
    x = build_locked_f6_feature_row(package["history_df"], "2025-08-02", "2025-08-01")
    with patch.object(package["preprocessors"]["sat"], "transform", return_value=np.full((1, 39), np.nan)):
        with pytest.raises(ValueError, match="transformed"):
            candidate.predict_weather_candidate(package, x, weather.iloc[0].to_dict(), "2025-08-02")
    assert package["models"]["sat"].seen == []
    package["quantile_models"]["sat"].value = np.inf
    with pytest.raises(ValueError, match="predictions"):
        candidate.predict_weather_candidate(package, x, weather.iloc[0].to_dict(), "2025-08-02")


def test_wrong_daytype_and_reordered_f6_fail(package, weather):
    x = build_locked_f6_feature_row(package["history_df"], "2025-08-02", "2025-08-01")
    with pytest.raises(ValueError, match="daytype"):
        candidate.predict_weather_candidate(package, x, weather.iloc[0].to_dict(), "2025-08-03")
    with pytest.raises(ValueError):
        candidate.combine_weather_features(x.iloc[:, ::-1], weather.iloc[0].to_dict())


@pytest.mark.parametrize("fault", ["missing_day", "duplicate_day", "missing_feature", "not_enough_sundays"])
def test_incomplete_training_weather_or_history_fails_without_fit(history, weather, fault):
    if fault == "missing_day":
        weather = weather.iloc[:-1]
    elif fault == "duplicate_day":
        weather = pd.concat([weather, weather.iloc[[0]]])
    elif fault == "missing_feature":
        weather = weather.drop(columns=[candidate.WEATHER_FEATURES[-1]])
    else:
        history = history[history[DATE_COL].dt.weekday == 5]
    with patch.object(candidate, "fit_final_models_by_daytype") as fit:
        with pytest.raises(ValueError):
            candidate.build_weather_candidate_package(
                location_id="ny_12550", attendance=history, weather_features=weather,
                package_id="weather_bad_v1", baseline_source={}, weather_input={}, weather_context={},
            )
        fit.assert_not_called()


def test_training_writes_only_versioned_candidate_and_roundtrips(tmp_path, monkeypatch, history):
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "models" / "visitor_model_ny_12550.joblib"
    baseline.parent.mkdir()
    context = {"zip_code": "12550", "country_code": "US", "timezone": "America/New_York"}
    joblib.dump({"location_id": "ny_12550", "weather_context": context}, baseline)
    original = baseline.read_bytes()
    monkeypatch.setattr(trainer, "VisitorPredictor", lambda path: SimpleNamespace(
        uses_locked_f6=True, package_id="f6_test_v1", history_df=history,
    ))
    monkeypatch.setattr(candidate, "fit_final_models_by_daytype", spy_fit)
    times = [f"{day.date().isoformat()}T{hour}:00" for day in history[DATE_COL] for hour in (11, 12, 13)]
    hourly = {"time": times}
    for name, value in zip(list(HOURLY_UNITS)[1:], (5., 1., 20., 0., 0.)):
        hourly[name] = [value] * len(times)
    weather_path = tmp_path / "historical_weather.json"
    weather_path.write_text(json.dumps({"latitude": 41.51142, "longitude": -74.04898,
                                       "timezone": "America/New_York", "hourly_units": HOURLY_UNITS, "hourly": hourly}))
    kwargs = dict(location_id="ny_12550", baseline_model=baseline, weather_input=weather_path,
                  package_id="weather_io_v1", latitude=41.5034, longitude=-74.0104,
                  output_dir=tmp_path / "models" / "candidates")
    destination = trainer.train_weather_candidate(**kwargs)
    assert baseline.read_bytes() == original
    assert (destination / "weather_candidate.joblib").is_file()
    assert not (destination / "model_package.joblib").exists()
    loaded = candidate.load_weather_candidate(destination, expected_location_id="ny_12550")
    assert loaded["baseline_model_sha256"] == candidate.sha256_file(baseline)
    assert loaded["weather_input"]["sha256"] == candidate.sha256_file(weather_path)
    with pytest.raises(FileExistsError):
        trainer.train_weather_candidate(**kwargs)
    with pytest.raises(ValueError, match="inside"):
        trainer.train_weather_candidate(**{**kwargs, "output_dir": baseline.parent})
    assert baseline.read_bytes() == original
    metadata_path = destination / "metadata.json"
    metadata_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        candidate.load_weather_candidate(destination)


@pytest.mark.parametrize("package_id", ["../escape_v1", "latest", "weather", "/tmp/weather_v1", "weather_v0"])
def test_unversioned_or_unsafe_destination_rejected(tmp_path, monkeypatch, package_id):
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError):
        trainer.candidate_package_dir(tmp_path / "models" / "candidates", package_id)


@pytest.mark.parametrize("changes", [{"latitude": 40.75}, {"longitude": -71.1}, {"timezone": "UTC"}, {"latitude": float("nan")}])
def test_historical_grid_or_timezone_mismatch_rejected(changes):
    payload = {"latitude": 41.51142, "longitude": -74.04898, "timezone": "America/New_York", **changes}
    with pytest.raises(ValueError):
        trainer._validate_weather_location(payload, timezone="America/New_York", latitude=41.5034, longitude=-74.0104)
