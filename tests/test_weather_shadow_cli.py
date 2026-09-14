"""Exercise the preparation rule through the public shadow-study commands."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("weather_shadow_cli", ROOT / "scripts/run_weather_shadow.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


@pytest.fixture
def command(tmp_path, monkeypatch, capsys):
    study = {
        "location_id": "ny_12550", "study_id": "weather_11_13_v1",
        "timezone": "America/New_York", "anchor_local_time": "10:00",
        "lead_hours": 12, "duration_basis": "elapsed_utc",
        "start_minutes_before_cutoff": 5, "capture_window_minutes": 15,
        "candidate": "models/candidates/test_weather_shadow_v1",
        "local_store": "data/weather_shadow/test.sqlite",
    }
    config = tmp_path / "study.json"
    config.write_text(json.dumps({"schema_version": 1, "studies": [study]}))
    store = Mock()
    store.load_runs.return_value = []
    make_store = Mock(return_value=store)
    read = Mock(return_value=[])
    capture = Mock(return_value={
        "run_id": "captured", "status": "paired", "service_date": "2026-09-19",
        "cutoff_at": "2026-09-19T02:00:00+00:00",
        "payload": {"baseline": {}, "candidate": {}},
    })
    monkeypatch.setattr(cli, "LocalShadowStore", make_store)
    monkeypatch.setattr(cli, "read_shadow_runs", read)
    monkeypatch.setattr(cli, "capture_shadow_forecast", capture)

    def run(arguments, at="2026-09-19T01:55:00+00:00"):
        instant = datetime.fromisoformat(at)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

        monkeypatch.setattr(cli, "datetime", Clock)
        monkeypatch.setattr(sys, "argv", ["run_weather_shadow.py", *arguments,
                                           "--location", "ny_12550", "--config", str(config)])
        result = cli.main()
        return result, json.loads(capsys.readouterr().out)

    return run, study, config, store, make_store, read, capture


@pytest.mark.parametrize("service_date,cutoff", [
    ("2026-09-19", "2026-09-19T02:00:00+00:00"),
    ("2026-09-20", "2026-09-20T02:00:00+00:00"),
    ("2026-03-08", "2026-03-08T02:00:00+00:00"),
    ("2026-11-01", "2026-11-01T03:00:00+00:00"),
])
def test_manual_capture_forwards_confirmed_cutoff(command, service_date, cutoff):
    run, study, _, store, make_store, read, capture = command
    code, result = run(["capture", "--service-date", service_date])
    assert code == 0
    assert result["status"] == "paired"
    capture.assert_called_once_with(
        "ny_12550", service_date, datetime.fromisoformat(cutoff), ROOT / study["candidate"],
        study_id="weather_11_13_v1", capture_window_minutes=15, store=store,
    )
    make_store.assert_called_once_with(ROOT / study["local_store"])
    read.assert_not_called()


@pytest.mark.parametrize("cutoff", [
    "2026-09-18T21:00:00-04:00", "2026-09-19T03:00:00Z", "2026-09-18T22:00:00",
])
def test_wrong_explicit_cutoff_rejects_before_capture_or_store(command, cutoff):
    run, _, _, store, make_store, read, capture = command
    with pytest.raises(ValueError, match="does not match"):
        run(["capture", "--service-date", "2026-09-19", "--cutoff-at", cutoff])
    capture.assert_not_called()
    make_store.assert_not_called()
    store.load_runs.assert_not_called()
    read.assert_not_called()


def test_equivalent_local_explicit_cutoff_is_accepted(command):
    run, *_, capture = command
    code, _ = run(["capture", "--service-date", "2026-09-19", "--cutoff-at", "2026-09-18T22:00:00-04:00"])
    assert code == 0
    assert capture.call_args.args[2] == datetime(2026, 9, 19, 2, tzinfo=timezone.utc)


def test_zero_window_override_reaches_capture_validation(command):
    run, *_, capture = command
    capture.side_effect = ValueError("capture_window_minutes must be between 1 and 60")
    with pytest.raises(ValueError, match="capture_window_minutes"):
        run(["capture", "--service-date", "2026-09-19", "--capture-window-minutes", "0"])
    assert capture.call_args.kwargs["capture_window_minutes"] == 0


@pytest.mark.parametrize("instant", [
    "2026-09-19T01:54:59+00:00",  # One second before collection starts.
    "2026-09-19T02:00:01+00:00",  # No late collection.
    "2026-09-21T01:55:00+00:00",  # Monday is not a service date.
])
def test_not_due_uses_no_forecast_or_store_io(command, instant):
    run, _, _, store, make_store, read, capture = command
    code, result = run(["capture-due"], at=instant)
    assert code == 0
    assert result["status"] == "not_due"
    assert len(result["next"]) == 1
    capture.assert_not_called()
    make_store.assert_not_called()
    store.load_runs.assert_not_called()
    read.assert_not_called()


@pytest.mark.parametrize("instant,service_date,cutoff", [
    ("2026-09-19T01:55:00+00:00", "2026-09-19", "2026-09-19T02:00:00+00:00"),
    ("2026-09-20T01:57:00+00:00", "2026-09-20", "2026-09-20T02:00:00+00:00"),
    ("2026-03-07T20:55:00-05:00", "2026-03-08", "2026-03-08T02:00:00+00:00"),
    ("2026-10-31T22:55:00-04:00", "2026-11-01", "2026-11-01T03:00:00+00:00"),
])
def test_due_capture_uses_correct_service_and_cutoff_across_dst(command, instant, service_date, cutoff):
    run, study, _, store, _, read, capture = command
    code, result = run(["capture-due"], at=instant)
    assert code == 0
    assert result["captures"][0]["service_date"] == service_date
    capture.assert_called_once_with(
        "ny_12550", service_date, cutoff, ROOT / study["candidate"],
        study_id="weather_11_13_v1", capture_window_minutes=15, store=store,
    )
    store.load_runs.assert_called_once_with("ny_12550")
    read.assert_not_called()


def eligible_record():
    return {
        "service_date": "2026-09-19", "status": "paired",
        "cutoff_at": "2026-09-18T22:00:00-04:00",
        "persisted_at": "2026-09-19T01:55:15+00:00",
        "weather_retrieved_at": "2026-09-19T01:55:10+00:00",
        "payload": {"study_id": "weather_11_13_v1", "eligible_for_evaluation": True},
    }


def test_due_skips_existing_eligible_pair_with_equivalent_timezone(command):
    run, _, _, store, _, _, capture = command
    store.load_runs.return_value = [eligible_record()]
    code, result = run(["capture-due"], at="2026-09-19T01:56:00+00:00")
    assert code == 0
    assert result["captures"] == [{"service_date": "2026-09-19", "status": "already_captured"}]
    capture.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"service_date": "2026-09-20"},
    {"status": "candidate_unavailable"},
    {"cutoff_at": "2026-09-19T03:00:00+00:00"},
    {"persisted_at": "2026-09-19T02:00:01+00:00"},
    {"weather_retrieved_at": "2026-09-19T02:00:01+00:00"},
    {"persisted_at": None},
    {"payload": {"study_id": "other-study", "eligible_for_evaluation": True}},
    {"payload": {"study_id": "weather_11_13_v1", "eligible_for_evaluation": False}},
])
def test_ineligible_existing_record_does_not_suppress_due_capture(command, changes):
    run, _, _, store, _, _, capture = command
    record = eligible_record()
    record.update(deepcopy(changes))
    store.load_runs.return_value = [record]
    code, result = run(["capture-due"])
    assert code == 0
    assert result["captures"][0]["status"] == "paired"
    capture.assert_called_once()


def test_due_failure_is_reported_with_nonzero_exit(command):
    run, *_, capture = command
    capture.return_value["status"] = "weather_unavailable"
    capture.return_value["payload"]["failure"] = {"message": "Forecast service unavailable"}
    code, result = run(["capture-due"])
    assert code == 2
    assert result["captures"][0]["status"] == "weather_unavailable"
    assert result["captures"][0]["failure"]["message"] == "Forecast service unavailable"


def test_schedule_renders_dst_local_cutoffs_without_store_or_forecast(command):
    run, _, _, store, make_store, read, capture = command
    code, result = run(["schedule", "--count", "2"], at="2026-10-30T12:00:00+00:00")
    assert code == 0
    assert [entry["cutoff_local"] for entry in result["upcoming"]] == [
        "2026-10-30T22:00:00-04:00", "2026-10-31T23:00:00-04:00",
    ]
    capture.assert_not_called()
    make_store.assert_not_called()
    store.load_runs.assert_not_called()
    read.assert_not_called()


@pytest.mark.parametrize("field,value", [("duration_basis", "wall_clock"), ("timezone", "UTC")])
def test_invalid_study_timing_rejects_before_capture_or_store(command, field, value):
    run, study, config, store, make_store, read, capture = command
    study[field] = value
    config.write_text(json.dumps({"schema_version": 1, "studies": [study]}))
    with pytest.raises(ValueError, match="elapsed hours and the configured location timezone"):
        run(["capture-due"])
    capture.assert_not_called()
    make_store.assert_not_called()
    store.load_runs.assert_not_called()
    read.assert_not_called()
