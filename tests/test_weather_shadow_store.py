from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sqlite3
from unittest.mock import patch
from urllib.error import HTTPError
from uuid import uuid4

import pytest

from src import weather_shadow_store as store


def record(**overrides):
    row = {
        "run_id": str(uuid4()),
        "location_id": "ny_12550",
        "service_date": "2026-09-19",
        "cutoff_at": "2026-09-18T18:00:00+00:00",
        "weather_retrieved_at": "2026-09-18T17:55:00+00:00",
        "recorded_at": "2026-09-18T17:55:05+00:00",
        "status": "paired",
        "payload": {
            "baseline": {"package_id": "F6-package", "predicted_visitors": 120.5},
            "candidate": {"package_id": "weather-v1", "predicted_visitors": 117.4},
            "weather": {"hourly": {"snowfall": [0.0, 0.2]}},
        },
    }
    row.update(overrides)
    return row


def saved_record(row=None):
    saved = deepcopy(record() if row is None else row)
    saved["persisted_at"] = "2026-09-18T17:56:05.123456+00:00"
    return saved


def test_append_preserves_reruns_and_raw_weather(tmp_path):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    first = record()
    rerun = record(recorded_at="2026-09-18T17:56:00Z")
    assert log.append(first) == first["run_id"]
    assert log.append(rerun) == rerun["run_id"]
    first["payload"]["weather"]["hourly"]["snowfall"][1] = 99
    saved = log.load_runs("ny_12550")
    assert [row["run_id"] for row in saved] == [rerun["run_id"], first["run_id"]]
    assert saved[1]["payload"]["weather"]["hourly"]["snowfall"] == [0.0, 0.2]


def test_idempotent_retry_and_conflicting_id(tmp_path):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    row = record()
    log.append(row)
    assert log.append(deepcopy(row)) == row["run_id"]
    conflict = deepcopy(row)
    conflict["payload"]["candidate"]["predicted_visitors"] += 1
    with pytest.raises(store.ShadowRecordConflictError, match="different content"):
        log.append(conflict)
    assert len(log.load_runs("ny_12550")) == 1
    assert log.load_runs("ny_12550")[0]["payload"] == row["payload"]


@pytest.mark.parametrize("sql", [
    "UPDATE weather_shadow_runs SET status='weather_unavailable'",
    "DELETE FROM weather_shadow_runs",
    "INSERT OR REPLACE INTO weather_shadow_runs SELECT * FROM weather_shadow_runs",
])
def test_sqlite_triggers_reject_mutations(tmp_path, sql):
    path = tmp_path / "shadow.sqlite"
    log = store.LocalShadowStore(path)
    log.append(record())
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable|already exists"):
            conn.execute(sql)
    assert len(log.load_runs("ny_12550")) == 1


def test_empty_read_does_not_create_database(tmp_path):
    path = tmp_path / "missing" / "shadow.sqlite"
    assert store.LocalShadowStore(path).load_runs("ny_12550") == []
    assert not path.parent.exists()


def test_reads_filter_location_and_limit(tmp_path):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    row = record()
    log.append(row)
    log.append(record(location_id="another_location"))
    later = record(recorded_at="2026-09-18T17:56:00Z")
    log.append(later)
    assert [r["run_id"] for r in log.load_runs("ny_12550", limit=1)] == [later["run_id"]]


@pytest.mark.parametrize("overrides,match", [
    ({"run_id": "not-a-uuid"}, "UUID"),
    ({"location_id": "../attendance"}, "location_id"),
    ({"location_id": "NY_12550"}, "location_id"),
    ({"service_date": "20260919"}, "YYYY-MM-DD"),
    ({"cutoff_at": "2026-09-18T18:00:00"}, "timezone"),
    ({"recorded_at": "not-a-date"}, "timestamp"),
    ({"weather_retrieved_at": None}, "require weather_retrieved_at"),
    ({"weather_retrieved_at": "2026-09-18T18:00:01Z"}, "by cutoff_at"),
    ({"recorded_at": "2026-09-18T18:00:01Z"}, "by cutoff_at"),
    ({"status": "unknown"}, "status must"),
    ({"payload": []}, "JSON object"),
    ({"payload": {"bad": float("nan")}}, "JSON compliant"),
])
def test_invalid_records_fail_before_creating_database(tmp_path, overrides, match):
    path = tmp_path / "shadow.sqlite"
    with pytest.raises(ValueError, match=match):
        store.LocalShadowStore(path).append(record(**overrides))
    assert not path.exists()


def test_naive_weather_timestamp_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        store.LocalShadowStore(tmp_path / "shadow.sqlite").append(
            record(weather_retrieved_at="2026-09-18T17:55:00")
        )


@pytest.mark.parametrize("status", ["weather_unavailable", "candidate_unavailable", "cutoff_missed"])
def test_failed_capture_can_be_saved_after_cutoff(tmp_path, status):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    row = record(status=status, weather_retrieved_at=None, recorded_at="2026-09-18T18:00:01Z")
    assert log.append(row) == row["run_id"]


def test_offsets_are_normalized_and_idempotent(tmp_path):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    row = record(recorded_at="2026-09-18T13:55:05-04:00")
    log.append(row)
    row["recorded_at"] = "2026-09-18T17:55:05Z"
    assert log.append(row) == row["run_id"]
    assert log.load_runs("ny_12550")[0]["recorded_at"] == "2026-09-18T17:55:05.000000+00:00"


@pytest.mark.parametrize("limit", [0, -1, 10001, True, "10"])
def test_read_limit_is_bounded(tmp_path, limit):
    with pytest.raises(ValueError, match="limit"):
        store.LocalShadowStore(tmp_path / "shadow.sqlite").load_runs("ny_12550", limit=limit)


def test_default_path_never_opens_attendance_database(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(store, "_supabase_config", lambda: None)
    row = record()
    assert store.save_shadow_run(row) == row["run_id"]
    assert (tmp_path / "data/weather_shadow/ny_12550.sqlite").exists()
    assert not (tmp_path / "data/locations").exists()
    assert store.read_shadow_runs("ny_12550")[0]["payload"] == row["payload"]


def test_supabase_append_only_posts_no_upsert():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    row = record()
    with patch.object(log, "_request", return_value=None) as request:
        assert log.append(row) == row["run_id"]
    request.assert_called_once_with("POST", payload=store._normalize_record(row))


def test_supabase_conflict_only_accepts_identical_retry():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    row = record()
    with patch.object(log, "_request", side_effect=[store._SupabaseHTTPError(409, "duplicate"), [saved_record(row)]]) as request:
        assert log.append(row) == row["run_id"]
    assert [call.args[0] for call in request.call_args_list] == ["POST", "GET"]
    altered = deepcopy(row)
    altered["payload"]["baseline"]["predicted_visitors"] += 10
    with patch.object(log, "_request", side_effect=[store._SupabaseHTTPError(409, "duplicate"), [saved_record(altered)]]):
        with pytest.raises(store.ShadowRecordConflictError):
            log.append(row)


def test_cloud_error_surfaces_without_local_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(store, "_supabase_config", lambda: {"url": "https://example.test", "key": "key"})
    with patch.object(store.SupabaseShadowStore, "_request", side_effect=store._SupabaseHTTPError(401, "unauthorized")) as request:
        with pytest.raises(RuntimeError, match="401"):
            store.save_shadow_run(record())
    assert request.call_count == 1
    assert not (tmp_path / "data").exists()


def test_supabase_partial_configuration_is_not_local_fallback(monkeypatch):
    monkeypatch.setattr(store, "_secret_value", lambda *names: "https://example.test" if "SUPABASE_URL" in names else None)
    with pytest.raises(RuntimeError, match="SERVICE_ROLE_KEY"):
        store._supabase_config()


def test_supabase_no_configuration_uses_local(monkeypatch):
    monkeypatch.setattr(store, "_secret_value", lambda *names: None)
    assert store._supabase_config() is None


def test_supabase_paginates_even_when_server_returns_short_pages():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    newer = record(recorded_at="2026-09-18T17:56:00Z")
    older = record(recorded_at="2026-09-18T17:55:00Z")
    with patch.object(log, "_request", side_effect=[[saved_record(newer)], [saved_record(older)], []]) as request:
        rows = log.load_runs("ny_12550")
    assert [r["run_id"] for r in rows] == [newer["run_id"], older["run_id"]]
    assert request.call_count == 3
    assert "or" not in request.call_args_list[0].kwargs["params"]
    assert "run_id.lt." + newer["run_id"] in request.call_args_list[1].kwargs["params"]["or"]
    assert all(call.kwargs["params"]["location_id"] == "eq.ny_12550" for call in request.call_args_list)


def test_supabase_read_limit_stops_pagination():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    with patch.object(log, "_request", return_value=[saved_record()]) as request:
        assert len(log.load_runs("ny_12550", limit=1)) == 1
    assert request.call_count == 1


def test_supabase_pagination_cannot_loop_on_same_record():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    with patch.object(log, "_request", return_value=[saved_record()]):
        with pytest.raises(RuntimeError, match="did not advance"):
            log.load_runs("ny_12550")


def test_supabase_http_error_is_reported():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    failure = HTTPError(log.url, 403, "Forbidden", {}, io.BytesIO(b"permission denied"))
    with patch.object(store, "urlopen", side_effect=failure):
        with pytest.raises(RuntimeError, match=r"403.*permission denied"):
            log.append(record())


def test_shared_migration_contains_only_separate_shadow_log():
    sql = (Path(__file__).resolve().parents[1] / "supabase/migrations/20260913_weather_shadow_snapshots.sql").read_text()
    assert "create table if not exists public.weather_shadow_runs" in sql
    assert "grant select on public.weather_shadow_runs to service_role" in sql
    assert "grant insert (" in sql
    assert "persisted_at" not in sql.split("grant insert (")[1].split(") on public.weather_shadow_runs")[0]
    assert "persisted_at timestamptz not null default clock_timestamp()" in sql
    assert "revoke all on public.weather_shadow_runs from public, anon, authenticated" in sql
    assert "before update or delete or truncate" in sql
    assert "weather_shadow_paired_before_cutoff" in sql
    assert "prediction_logs" not in sql
    assert "attendance" not in sql.lower().replace("attendance outcomes", "")


def test_sqlite_receipt_is_database_time_not_claimed_recorded_at(tmp_path):
    log = store.LocalShadowStore(tmp_path / "shadow.sqlite")
    row = record(
        weather_retrieved_at="2000-01-01T12:00:00Z",
        recorded_at="2000-01-01T12:00:01Z",
        cutoff_at="2000-01-01T13:00:00Z",
    )
    before = datetime.now(timezone.utc)
    log.append(row)
    after = datetime.now(timezone.utc)
    saved = log.load_runs("ny_12550")[0]
    receipt = datetime.fromisoformat(saved["persisted_at"])
    # SQLite's clock has millisecond resolution; Python's clock has microseconds.
    assert before - timedelta(milliseconds=1) <= receipt <= after + timedelta(milliseconds=1)
    assert receipt > datetime.fromisoformat(store._normalize_record(row)["cutoff_at"])
    assert saved["status"] == "paired"  # Kept immutable; evaluation excludes late receipt.
    assert "persisted_at" not in row
    assert log.append(row) == row["run_id"]
    assert log.load_runs("ny_12550")[0]["persisted_at"] == saved["persisted_at"]


@pytest.mark.parametrize("backend", ["local", "supabase"])
def test_caller_cannot_forge_persistence_receipt(tmp_path, backend):
    log = (store.LocalShadowStore(tmp_path / "shadow.sqlite") if backend == "local"
           else store.SupabaseShadowStore("https://example.test", "test-key"))
    row = saved_record()
    with patch.object(store, "urlopen") as request:
        with pytest.raises(ValueError, match="extra=.*persisted_at"):
            log.append(row)
    request.assert_not_called()
    assert not (tmp_path / "shadow.sqlite").exists()


def test_supabase_receipt_roundtrip_is_read_only_and_preserved_on_retry():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    row = record()
    server_row = saved_record(row)
    server_row["persisted_at"] = "2026-09-18T13:56:05.987654-04:00"
    with patch.object(log, "_request", return_value=None) as request:
        assert log.append(row) == row["run_id"]
    assert "persisted_at" not in request.call_args.kwargs["payload"]
    with patch.object(log, "_request", side_effect=[[server_row], []]):
        loaded = log.load_runs("ny_12550")
    assert loaded[0]["persisted_at"] == "2026-09-18T17:56:05.987654+00:00"
    with patch.object(log, "_request", side_effect=[store._SupabaseHTTPError(409, "duplicate"), [server_row]]):
        assert log.append(row) == row["run_id"]


def test_receipt_missing_from_server_is_not_fabricated():
    log = store.SupabaseShadowStore("https://example.test", "test-key")
    with patch.object(log, "_request", return_value=[record()]):
        with pytest.raises(ValueError, match="no database persisted_at receipt"):
            log.load_runs("ny_12550")
