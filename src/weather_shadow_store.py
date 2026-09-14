"""Append-only weather experiment records, separate from staff prediction logs.

A new run_id preserves every rerun. Reusing a run_id is permitted only for an
identical record (e.g. retrying after an uncertain network response). Attendance
and evaluation outcomes are joined later; saved forecasts are never rewritten.
Reads include persisted_at, a database-generated insertion receipt. It is not
accepted as an input field and must be checked against cutoff during evaluation.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from src.config import PROJECT_ROOT
from src.prediction_logs import _secret_value

TABLE = "weather_shadow_runs"
MAX_READ_RUNS = 10000
PAGE_SIZE = 500
_FIELDS = (
    "run_id", "location_id", "service_date", "cutoff_at",
    "weather_retrieved_at", "recorded_at", "status", "payload",
)
_STATUSES = {"paired", "weather_unavailable", "candidate_unavailable", "cutoff_missed"}
_LOCATION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS weather_shadow_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    location_id TEXT NOT NULL,
    service_date TEXT NOT NULL,
    cutoff_at TEXT NOT NULL,
    weather_retrieved_at TEXT,
    recorded_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('paired', 'weather_unavailable', 'candidate_unavailable', 'cutoff_missed')
    ),
    payload TEXT NOT NULL,
    persisted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_weather_shadow_runs_location_date
    ON weather_shadow_runs(location_id, service_date, recorded_at, run_id);
CREATE TRIGGER IF NOT EXISTS weather_shadow_runs_no_update
BEFORE UPDATE ON weather_shadow_runs BEGIN
    SELECT RAISE(ABORT, 'weather shadow records are immutable');
END;
CREATE TRIGGER IF NOT EXISTS weather_shadow_runs_no_delete
BEFORE DELETE ON weather_shadow_runs BEGIN
    SELECT RAISE(ABORT, 'weather shadow records are immutable');
END;
CREATE TRIGGER IF NOT EXISTS weather_shadow_runs_no_replace
BEFORE INSERT ON weather_shadow_runs
WHEN EXISTS (SELECT 1 FROM weather_shadow_runs WHERE run_id = NEW.run_id)
BEGIN
    SELECT RAISE(ABORT, 'weather shadow run_id already exists');
END;
"""


class ShadowRecordConflictError(ValueError):
    """A run_id was reused with different content."""


class _SupabaseHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        super().__init__(f"Weather shadow Supabase request failed ({status}): {body}")


def _location_id(value: Any) -> str:
    if not isinstance(value, str) or not _LOCATION_ID.fullmatch(value):
        raise ValueError("location_id must contain only lowercase letters, digits, underscores or hyphens")
    return value


def _utc_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json(value: Any) -> str:
    # Reject NaN/Infinity and unsupported objects instead of silently damaging
    # the saved raw weather or the feature/prediction provenance.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(_FIELDS) - record.keys()
    extra = record.keys() - set(_FIELDS)
    if missing or extra:
        raise ValueError(f"Invalid shadow record fields: missing={sorted(missing)}, extra={sorted(extra)}")
    normalized = dict(record)
    try:
        if not isinstance(record["run_id"], str):
            raise ValueError("run_id must be a UUID string")
        normalized["run_id"] = str(UUID(record["run_id"]))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("run_id must be a UUID string") from exc
    normalized["location_id"] = _location_id(record["location_id"])
    try:
        service_date = date.fromisoformat(record["service_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("service_date must use YYYY-MM-DD") from exc
    if service_date.isoformat() != record["service_date"]:
        raise ValueError("service_date must use YYYY-MM-DD")
    for field in ("cutoff_at", "recorded_at"):
        normalized[field] = _utc_timestamp(record[field], field)
    if record["weather_retrieved_at"] is not None:
        normalized["weather_retrieved_at"] = _utc_timestamp(
            record["weather_retrieved_at"], "weather_retrieved_at"
        )
    if record["status"] not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    if record["status"] == "paired":
        if normalized["weather_retrieved_at"] is None:
            raise ValueError("paired records require weather_retrieved_at")
        if max(normalized["weather_retrieved_at"], normalized["recorded_at"]) > normalized["cutoff_at"]:
            raise ValueError("paired weather and predictions must be saved by cutoff_at")
    if not isinstance(record["payload"], dict):
        raise ValueError("payload must be a JSON object")
    # Return an independent JSON-native copy, preserving the exact values that
    # either backend will store and ensuring callers cannot mutate the record.
    normalized["payload"] = json.loads(_canonical_json(record["payload"]))
    return normalized


def _normalize_saved_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Separate database receipts from the eight caller-controlled fields."""
    if "persisted_at" not in record:
        raise ValueError("Saved weather shadow record has no database persisted_at receipt")
    saved = dict(record)
    persisted_at = _utc_timestamp(saved.pop("persisted_at"), "persisted_at")
    normalized = _normalize_record(saved)
    normalized["persisted_at"] = persisted_at
    return normalized


def _read_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_READ_RUNS:
        raise ValueError(f"limit must be between 1 and {MAX_READ_RUNS}")
    return limit


def local_shadow_path(location_id: str) -> Path:
    return PROJECT_ROOT / "data" / "weather_shadow" / f"{_location_id(location_id)}.sqlite"


def _decoded_sqlite_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = json.loads(result["payload"])
    return _normalize_saved_record(result)


def _verify_retry(existing: Mapping[str, Any], record: dict[str, Any]) -> str:
    saved = _normalize_saved_record(existing)
    saved.pop("persisted_at")
    if saved != record:
        raise ShadowRecordConflictError(
            f"run_id {record['run_id']} already exists with different content; use a new UUID for a rerun"
        )
    return record["run_id"]


class LocalShadowStore:
    """Isolated log; reads include insertion receipts and never mutate storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, Any]) -> str:
        row = _normalize_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=15)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA_SQL)
            try:
                with conn:
                    conn.execute(
                        f"INSERT INTO {TABLE} ({', '.join(_FIELDS)}) VALUES ({', '.join('?' for _ in _FIELDS)})",
                        [row[key] if key != "payload" else _canonical_json(row[key]) for key in _FIELDS],
                    )
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    f"SELECT * FROM {TABLE} WHERE run_id = ?", (row["run_id"],)
                ).fetchone()
                if existing is None:
                    raise
                return _verify_retry(_decoded_sqlite_row(existing), row)
        return row["run_id"]

    def load_runs(self, location_id: str, *, limit: int = MAX_READ_RUNS) -> list[dict[str, Any]]:
        location_id = _location_id(location_id)
        limit = _read_limit(limit)
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM {TABLE} WHERE location_id = ? "
                "ORDER BY service_date DESC, recorded_at DESC, run_id DESC LIMIT ?",
                (location_id, limit),
            ).fetchall()
        return [_decoded_sqlite_row(row) for row in rows]


def _supabase_config() -> dict[str, str] | None:
    url = _secret_value("SUPABASE_URL", "url")
    key = _secret_value("SUPABASE_SERVICE_ROLE_KEY", "service_role_key")
    if not url and not key:
        return None
    if not url or not key:
        raise RuntimeError(
            "Weather shadow Supabase storage requires both SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY. Anonymous credentials cannot write immutable experiment records."
        )
    return {"url": url.rstrip("/"), "key": key}


class SupabaseShadowStore:
    """Shared append-only deployment log; never falls back to local storage."""

    def __init__(self, url: str, key: str) -> None:
        self.url = url.rstrip("/") + f"/rest/v1/{TABLE}"
        self.key = key

    def _request(self, method: str, *, params: dict[str, str] | None = None,
                 payload: dict[str, Any] | None = None) -> Any:
        url = self.url + ("?" + urlencode(params) if params else "")
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        data = _canonical_json(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise _SupabaseHTTPError(
                exc.code, exc.read().decode("utf-8", errors="replace")[:2000]
            ) from exc
        return json.loads(body) if body else None

    def append(self, record: Mapping[str, Any]) -> str:
        row = _normalize_record(record)
        try:
            self._request("POST", payload=row)
        except _SupabaseHTTPError as exc:
            if exc.status != 409:
                raise
            existing = self._request(
                "GET", params={"select": "*", "run_id": f"eq.{row['run_id']}", "limit": "1"}
            )
            if not isinstance(existing, list) or len(existing) != 1:
                raise
            return _verify_retry(existing[0], row)
        return row["run_id"]

    def load_runs(self, location_id: str, *, limit: int = MAX_READ_RUNS) -> list[dict[str, Any]]:
        location_id = _location_id(location_id)
        limit = _read_limit(limit)
        rows: list[dict[str, Any]] = []
        # Keyset pagination prevents newly appended reruns from shifting
        # offsets and duplicating rows already read. Continue until an empty
        # page even when the server caps responses below PAGE_SIZE.
        last: dict[str, Any] | None = None
        while len(rows) < limit:
            params = {
                "select": "*", "location_id": f"eq.{location_id}",
                "order": "service_date.desc,recorded_at.desc,run_id.desc",
                "limit": str(min(PAGE_SIZE, limit - len(rows))),
            }
            if last is not None:
                day, recorded, run_id = last["service_date"], last["recorded_at"], last["run_id"]
                params["or"] = (
                    f"(service_date.lt.{day},"
                    f"and(service_date.eq.{day},recorded_at.lt.{recorded}),"
                    f"and(service_date.eq.{day},recorded_at.eq.{recorded},run_id.lt.{run_id}))"
                )
            page = self._request("GET", params=params)
            if not isinstance(page, list):
                raise RuntimeError("Weather shadow Supabase read returned a non-list response")
            if not page:
                break
            normalized = [_normalize_saved_record(row) for row in page]
            if any(row["location_id"] != location_id for row in normalized):
                raise RuntimeError("Weather shadow Supabase read returned a different location")
            keys = [(row["service_date"], row["recorded_at"], row["run_id"]) for row in normalized]
            if keys != sorted(keys, reverse=True) or len(set(keys)) != len(keys):
                raise RuntimeError("Weather shadow Supabase pagination returned unordered or duplicate records")
            if last is not None and keys[0] >= (last["service_date"], last["recorded_at"], last["run_id"]):
                raise RuntimeError("Weather shadow Supabase pagination did not advance")
            rows.extend(normalized)
            last = normalized[-1]
        return rows[:limit]


def _configured_store(location_id: str) -> LocalShadowStore | SupabaseShadowStore:
    config = _supabase_config()
    if config is not None:
        return SupabaseShadowStore(**config)
    return LocalShadowStore(local_shadow_path(location_id))


def save_shadow_run(record: Mapping[str, Any]) -> str:
    return _configured_store(_location_id(record["location_id"])).append(record)


def read_shadow_runs(location_id: str, *, limit: int = MAX_READ_RUNS) -> list[dict[str, Any]]:
    return _configured_store(_location_id(location_id)).load_runs(location_id, limit=limit)
