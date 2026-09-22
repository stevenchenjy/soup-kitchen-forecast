from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest

from src import weather_shadow as shadow
from src.config import PROJECT_ROOT
from src.predictor import VisitorPredictor
from src.weather_candidate import history_sha256
from src.weather_snapshots import HOURLY_UNITS, extract_weather_features, snapshot_sha256

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")

# Fixed-clock tests must use a fixed package. The active package advances after
# nightly training and correctly fails the historical-origin leakage guard.
BASELINE = PROJECT_ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
ACTIVE_MODEL = PROJECT_ROOT / "models/visitor_model_ny_12550.joblib"
START = datetime(2026, 9, 18, 12, 55, tzinfo=timezone.utc)
CUTOFF = START + timedelta(minutes=5)


@pytest.fixture
def setup_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "model_file_for_location", lambda _: BASELINE)
    clock = Mock(return_value=START)
    store = Mock()
    candidate = tmp_path / "weather_candidate.joblib"
    candidate.write_bytes(b"test stub; loader is patched")
    package = {
        "baseline_model_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "created_at_utc": "2026-09-17T00:00:00+00:00",
        "training_weather": "realized_historical_bootstrap",
        "history": {"sha256": history_sha256(VisitorPredictor(str(BASELINE)).history_df)},
        "training_end_date": "2026-07-12",
        "weather_context": {"latitude": 41.50343, "longitude": -74.01042},
    }
    loader = Mock(return_value=package)
    predictor = Mock(return_value={
        "point": 100.0, "q80": 120.5, "suggested_meals": 121,
        "package_id": "weather-test-v1", "model_segment": "sat",
    })
    raw = {"timezone": "America/New_York", "hourly_units": dict(HOURLY_UNITS), "hourly": {
        "time": [f"2026-09-19T{hour}:00" for hour in [11, 12, 13]],
        "apparent_temperature": [18, 19, 20], "precipitation": [99, 1.2, .5],
        "wind_gusts_10m": [10, 20, 25], "snowfall": [0, 0, 0], "snow_depth": [0, 0, 0],
    }}
    weather = {
        "retrieved_at": START.isoformat(), "request_started_at": START.isoformat(),
        "cutoff_at": CUTOFF.isoformat(), "location_id": "ny_12550", "service_date": "2026-09-19",
        "timezone": "America/New_York", "source": "open_meteo_live_forecast", "feature_contract_id": "weather_11_13_v1",
        "features": extract_weather_features(raw, "2026-09-19"), "raw_response": raw,
        "geolocation": {"latitude": 41.50343, "longitude": -74.01042},
        "response_location": {"latitude": 41.51142, "longitude": -74.04898},
    }
    weather["snapshot_sha256"] = snapshot_sha256(weather)
    fetcher = Mock(return_value=weather)
    monkeypatch.setattr(shadow, "load_weather_candidate", loader)
    monkeypatch.setattr(shadow, "predict_weather_candidate", predictor)
    monkeypatch.setattr(shadow, "fetch_weather_snapshot", fetcher)
    return candidate, clock, store, package, loader, predictor, fetcher


def capture(fixture, **kwargs):
    candidate, clock, store, *_ = fixture
    return shadow.capture_shadow_forecast("ny_12550", "2026-09-19", CUTOFF, candidate, now=clock, store=store, **kwargs)


def saved(record):
    result = deepcopy(record)
    result["persisted_at"] = (START + timedelta(seconds=10)).isoformat()
    return result


def test_pair_matches_production_and_shares_attendance_row(setup_capture, monkeypatch):
    candidate, clock, store, _, _, predict_candidate, _ = setup_capture
    active_before = hashlib.sha256(ACTIVE_MODEL.read_bytes()).hexdigest()
    record = capture(setup_capture)
    monkeypatch.setattr("src.config.forecast_today", lambda *_: date(2026, 9, 18))
    live = VisitorPredictor(str(BASELINE)).predict_next("2026-09-19")
    assert record["status"] == "paired"
    assert record["payload"]["baseline"]["point"] == pytest.approx(live.predicted_visitors)
    assert record["payload"]["baseline"]["suggested_meals"] == live.suggested_meals
    x = predict_candidate.call_args.args[1]
    for key, value in record["payload"]["f6_features"].items():
        assert (pd.isna(x.iloc[0][key]) if value is None else x.iloc[0][key] == value)
    store.append.assert_called_once_with(record)
    assert hashlib.sha256(ACTIVE_MODEL.read_bytes()).hexdigest() == active_before


def test_baseline_trained_after_origin_is_rejected_before_weather(setup_capture, tmp_path):
    package = joblib.load(BASELINE)
    history = package["history_df"].copy()
    history.loc[history.index[-1], "service_date"] = pd.Timestamp("2026-09-20")
    package["history_df"] = history
    future_baseline = tmp_path / "future_baseline.joblib"
    joblib.dump(package, future_baseline)
    with pytest.raises(ValueError, match="attendance after the forecast origin"):
        capture(setup_capture, baseline_path=future_baseline)
    setup_capture[-1].assert_not_called()
    setup_capture[2].append.assert_not_called()


def test_weather_failure_retains_baseline_and_logs_failure(setup_capture):
    setup_capture[-1].side_effect = RuntimeError("weather unavailable")
    record = capture(setup_capture)
    assert record["status"] == "weather_unavailable"
    assert record["payload"]["baseline"]["point"] > 0
    assert record["payload"]["candidate"] is None
    assert not record["payload"]["eligible_for_evaluation"]
    setup_capture[-2].assert_not_called()
    setup_capture[2].append.assert_called_once()


def test_stale_candidate_still_saves_weather_without_paired_claim(setup_capture):
    setup_capture[3]["baseline_model_sha256"] = "0" * 64
    record = capture(setup_capture)
    assert record["status"] == "candidate_unavailable"
    assert record["payload"]["weather"] is not None
    assert "stale" in record["payload"]["failure"]["message"]


@pytest.mark.parametrize("instant,match", [(CUTOFF + timedelta(seconds=1), "passed"), (CUTOFF - timedelta(hours=1), "Too early")])
def test_no_backfill_or_early_capture(setup_capture, instant, match):
    setup_capture[1].return_value = instant
    with pytest.raises(ValueError, match=match):
        capture(setup_capture)
    setup_capture[-1].assert_not_called()
    setup_capture[2].append.assert_not_called()


def test_request_finishing_after_cutoff_is_ineligible(setup_capture):
    def fail_late(*args, **kwargs):
        setup_capture[1].return_value = CUTOFF + timedelta(seconds=1)
        raise ValueError("completed after cutoff")
    setup_capture[-1].side_effect = fail_late
    record = capture(setup_capture)
    assert record["status"] == "cutoff_missed"
    assert not record["payload"]["eligible_for_evaluation"]
    setup_capture[2].append.assert_called_once()


def test_storage_failure_surfaces(setup_capture):
    setup_capture[2].append.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        capture(setup_capture)


def test_eval_deduplicates_cutoff_reruns_and_keeps_other_cutoffs_separate(setup_capture):
    first = saved(capture(setup_capture))
    second = deepcopy(first)
    second["run_id"] = "second"
    second["recorded_at"] = (START + timedelta(seconds=2)).isoformat()
    second["weather_retrieved_at"] = (START + timedelta(seconds=1)).isoformat()
    second["payload"]["weather"]["retrieved_at"] = second["weather_retrieved_at"]
    second["payload"]["weather"]["snapshot_sha256"] = snapshot_sha256(second["payload"]["weather"])
    third = deepcopy(second)
    third["run_id"] = "other-cutoff"
    third["cutoff_at"] = (CUTOFF + timedelta(hours=1)).isoformat()
    third["payload"]["cutoff_local_time"] = "10:00:00"
    third["payload"]["weather"]["cutoff_at"] = third["cutoff_at"]
    third["payload"]["weather"]["snapshot_sha256"] = snapshot_sha256(third["payload"]["weather"])
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [125]})
    result = shadow.evaluate_shadow_runs([first, second, third], actual, now=START + timedelta(days=2))
    assert result["paired_rows"] == 2
    assert result["exclusions"]["superseded_capture"] == 1
    assert {row["run_id"] for row in result["predictions"]} == {"second", "other-cutoff"}
    assert len(result["metrics"]) == 4  # baseline/candidate for two distinct cutoffs
    assert len(result["strategy_metrics"]) == 4
    assert all(row["fitted_versions"] == 1 for row in result["strategy_metrics"])
    candidate_metric = next(row for row in result["metrics"] if row["model"] == "candidate")
    assert candidate_metric["mae"] == 25
    assert candidate_metric["shortfall_meals"] == 4


@pytest.mark.parametrize("status", ["closed", "unknown"])
def test_eval_excludes_closed_and_unknown_services(setup_capture, status):
    record = saved(capture(setup_capture))
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [0], "service_status": [status]})
    result = shadow.evaluate_shadow_runs([record], actual, now=START + timedelta(days=2))
    assert result["paired_rows"] == 0
    assert result["exclusions"]["service_not_confirmed_open"] == 1


def test_missing_or_future_actual_never_becomes_zero(setup_capture):
    record = saved(capture(setup_capture))
    empty = pd.DataFrame(columns=["service_date", "visitors"])
    assert shadow.evaluate_shadow_runs([record], empty, now=START + timedelta(days=2))["paired_rows"] == 0
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [100]})
    assert shadow.evaluate_shadow_runs([record], actual, now=START)["paired_rows"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        shadow.evaluate_shadow_runs([record], pd.concat([actual, actual]), now=START + timedelta(days=2))


def test_eval_rejects_late_record_even_if_marked_paired(setup_capture):
    record = saved(capture(setup_capture))
    record["weather_retrieved_at"] = (CUTOFF + timedelta(seconds=1)).isoformat()
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [100]})
    result = shadow.evaluate_shadow_runs([record], actual, now=START + timedelta(days=2))
    assert result["paired_rows"] == 0
    assert result["exclusions"]["invalid_provenance"] == 1


def test_candidate_with_different_training_location_is_not_paired(setup_capture):
    setup_capture[3]["weather_context"]["latitude"] = 42.0
    record = capture(setup_capture)
    assert record["status"] == "candidate_unavailable"
    assert "coordinates" in record["payload"]["failure"]["message"]


@pytest.mark.parametrize("field", ["persisted_at", "snapshot"])
def test_eval_requires_on_time_storage_and_intact_weather(setup_capture, field):
    record = saved(capture(setup_capture))
    if field == "persisted_at":
        record[field] = (CUTOFF + timedelta(seconds=1)).isoformat()
    else:
        record["payload"]["weather"]["features"]["wx_precip_mm"] = 500
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [100]})
    result = shadow.evaluate_shadow_runs([record], actual, now=START + timedelta(days=2))
    assert result["paired_rows"] == 0
    assert result["exclusions"]["invalid_provenance"] == 1


def test_closed_service_without_count_is_not_pending_attendance(setup_capture):
    record = saved(capture(setup_capture))
    actual = pd.DataFrame({"service_date": ["2026-09-19"], "visitors": [float("nan")], "service_status": ["closed"]})
    result = shadow.evaluate_shadow_runs([record], actual, now=START + timedelta(days=2))
    assert result["exclusions"] == {"service_not_confirmed_open": 1}
