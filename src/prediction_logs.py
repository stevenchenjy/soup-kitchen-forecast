from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from src.config import (
    BASELINE_MEALS_PREPARED,
    ESTIMATED_WASTE_REDUCTION_RATE,
    KG_CO2E_PER_KG_FOOD_WASTE,
    MEAL_WEIGHT_KG,
    location_db_file,
)
from src.location_config import list_locations


PREDICTION_LOGS_TABLE_DEFAULT = "prediction_logs"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id TEXT NOT NULL,
    service_date TEXT NOT NULL,
    prediction_created_at TEXT NOT NULL,
    predicted_visitors REAL NOT NULL,
    predicted_quantile REAL,
    residual_buffer REAL,
    suggested_meals INTEGER NOT NULL,
    meal_buffer_pct REAL,
    model_segment TEXT,
    actual_visitors INTEGER,
    absolute_error REAL,
    baseline_meals_prepared INTEGER,
    waste_avoided_meals REAL,
    estimated_co2e_reduction_kg REAL,
    created_by TEXT,
    source_app TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_location_id ON prediction_logs(location_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_service_date ON prediction_logs(service_date);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_location_service ON prediction_logs(location_id, service_date);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _service_date(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _secret_value(*names: str) -> str | None:
    for name in names:
        env_value = os.getenv(name)
        if env_value:
            return env_value

    if st is None:
        return None

    try:
        secrets_root = st.secrets
    except Exception:
        return None

    for name in names:
        try:
            if name in secrets_root and secrets_root[name]:
                return str(secrets_root[name])
        except Exception:
            pass

    try:
        supabase = secrets_root.get("supabase", {})
    except Exception:
        return None
    for name in names:
        key = name.lower()
        if key.startswith("supabase_"):
            key = key.removeprefix("supabase_")
        try:
            if key in supabase and supabase[key]:
                return str(supabase[key])
        except Exception:
            pass
    return None


def _supabase_config() -> dict[str, str] | None:
    url = _secret_value("SUPABASE_URL", "url")
    key = _secret_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "service_role_key", "anon_key", "key")
    table = (
        _secret_value("SUPABASE_PREDICTION_LOGS_TABLE", "prediction_logs_table")
        or PREDICTION_LOGS_TABLE_DEFAULT
    )
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key, "table": table}


def prediction_log_store_mode() -> str:
    return "supabase" if _supabase_config() else "local_sqlite"


def _supabase_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    config = _supabase_config()
    if config is None:
        raise RuntimeError("Supabase is not configured.")
    headers = {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _supabase_url() -> str:
    config = _supabase_config()
    if config is None:
        raise RuntimeError("Supabase is not configured.")
    return f"{config['url']}/rest/v1/{config['table']}"


def _supabase_request(
    method: str,
    params: dict[str, str] | None = None,
    payload: Any | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    url = _supabase_url()
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=_supabase_headers(extra_headers), method=method)
    with urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def _connect(location_id: str) -> sqlite3.Connection:
    db = location_db_file(location_id)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _waste_fields(suggested_meals: int) -> tuple[float, float]:
    waste_avoided = float(suggested_meals) * ESTIMATED_WASTE_REDUCTION_RATE
    co2e = waste_avoided * MEAL_WEIGHT_KG * KG_CO2E_PER_KG_FOOD_WASTE
    return waste_avoided, co2e


def _recalculate_absolute_error(row: dict[str, Any]) -> None:
    if row.get("actual_visitors") is None:
        row["absolute_error"] = None
        return
    row["absolute_error"] = abs(float(row["actual_visitors"]) - float(row["predicted_visitors"]))


def _first_actual_visitors(rows: Iterable[dict[str, Any]]) -> Any | None:
    for row in rows:
        if row.get("actual_visitors") is not None:
            return row["actual_visitors"]
    return None


def _row_from_prediction(
    location_id: str,
    prediction_output: Any,
    created_by: str | None,
    source_app: str | None,
    baseline_meals_prepared: int | None,
) -> dict[str, Any]:
    baseline = BASELINE_MEALS_PREPARED if baseline_meals_prepared is None else baseline_meals_prepared
    suggested_meals = int(prediction_output.suggested_meals)
    waste_avoided, co2e = _waste_fields(suggested_meals)
    now = _now()
    return {
        "location_id": location_id,
        "service_date": _service_date(prediction_output.service_date),
        "prediction_created_at": now,
        "predicted_visitors": float(prediction_output.predicted_visitors),
        "predicted_quantile": float(prediction_output.predicted_quantile),
        "residual_buffer": float(prediction_output.residual_buffer),
        "suggested_meals": suggested_meals,
        "meal_buffer_pct": float(prediction_output.meal_buffer_pct),
        "model_segment": str(prediction_output.model_segment),
        "actual_visitors": None,
        "absolute_error": None,
        "baseline_meals_prepared": baseline,
        "waste_avoided_meals": waste_avoided,
        "estimated_co2e_reduction_kg": co2e,
        "created_by": created_by,
        "source_app": source_app,
        "created_at": now,
        "updated_at": now,
    }


def save_prediction_log(
    location_id: str,
    prediction_output: Any,
    created_by: str | None = None,
    source_app: str | None = None,
    baseline_meals_prepared: int | None = None,
) -> int | str | None:
    row = _row_from_prediction(location_id, prediction_output, created_by, source_app, baseline_meals_prepared)
    if _supabase_config():
        existing_rows = _supabase_request(
            "GET",
            params={
                "select": "id,actual_visitors,created_at",
                "location_id": f"eq.{row['location_id']}",
                "service_date": f"eq.{row['service_date']}",
                "order": "prediction_created_at.desc,id.desc",
            },
        )
        if isinstance(existing_rows, list) and existing_rows:
            keep = existing_rows[0]
            row["actual_visitors"] = _first_actual_visitors(existing_rows)
            if keep.get("created_at"):
                row["created_at"] = keep["created_at"]
            _recalculate_absolute_error(row)
            _supabase_request(
                "PATCH",
                params={"id": f"eq.{keep['id']}"},
                payload=row,
                extra_headers={"Prefer": "return=minimal"},
            )
            for duplicate in existing_rows[1:]:
                _supabase_request(
                    "DELETE",
                    params={"id": f"eq.{duplicate['id']}"},
                    extra_headers={"Prefer": "return=minimal"},
                )
            return keep.get("id")

        result = _supabase_request(
            "POST",
            params={"on_conflict": "location_id,service_date"},
            payload=row,
            extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        if isinstance(result, list) and result:
            return result[0].get("id")
        return None

    with _connect(location_id) as conn:
        existing_rows = conn.execute(
            "SELECT id, actual_visitors, created_at FROM prediction_logs "
            "WHERE location_id = ? AND service_date = ? "
            "ORDER BY prediction_created_at DESC, id DESC",
            (row["location_id"], row["service_date"]),
        )
        existing = existing_rows.fetchall()
        if existing:
            keep = existing[0]
            row["actual_visitors"] = _first_actual_visitors(dict(existing_row) for existing_row in existing)
            row["created_at"] = keep["created_at"]
            _recalculate_absolute_error(row)
            assignments = ", ".join(f"{column} = ?" for column in row if column != "id")
            values = [row[column] for column in row if column != "id"]
            conn.execute(
                f"UPDATE prediction_logs SET {assignments} WHERE id = ?",
                values + [keep["id"]],
            )
            duplicate_ids = [duplicate["id"] for duplicate in existing[1:]]
            if duplicate_ids:
                placeholders = ", ".join("?" for _ in duplicate_ids)
                conn.execute(f"DELETE FROM prediction_logs WHERE id IN ({placeholders})", duplicate_ids)
            conn.commit()
            return keep["id"]

        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        cur = conn.execute(
            f"INSERT INTO prediction_logs ({', '.join(columns)}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )
        conn.commit()
        return cur.lastrowid


def _actual_update_values(row: dict[str, Any], actual_visitors: int) -> dict[str, Any]:
    absolute_error = abs(float(actual_visitors) - float(row["predicted_visitors"]))
    suggested_meals = int(row["suggested_meals"])
    waste_avoided, co2e = _waste_fields(suggested_meals)
    return {
        "actual_visitors": int(actual_visitors),
        "absolute_error": absolute_error,
        "waste_avoided_meals": waste_avoided,
        "estimated_co2e_reduction_kg": co2e,
        "updated_at": _now(),
    }


def update_prediction_logs_with_actual(location_id: str, service_date: str, actual_visitors: int) -> int:
    dt = _service_date(service_date)
    if _supabase_config():
        rows = _supabase_request(
            "GET",
            params={
                "select": "id,predicted_visitors,suggested_meals",
                "location_id": f"eq.{location_id}",
                "service_date": f"eq.{dt}",
            },
        )
        if not isinstance(rows, list):
            return 0
        for row in rows:
            values = _actual_update_values(row, actual_visitors)
            _supabase_request(
                "PATCH",
                params={"id": f"eq.{row['id']}"},
                payload=values,
                extra_headers={"Prefer": "return=minimal"},
            )
        return len(rows)

    with _connect(location_id) as conn:
        rows = conn.execute(
            "SELECT id, predicted_visitors, suggested_meals "
            "FROM prediction_logs WHERE location_id = ? AND service_date = ?",
            (location_id, dt),
        ).fetchall()
        for row in rows:
            values = _actual_update_values(dict(row), actual_visitors)
            conn.execute(
                "UPDATE prediction_logs SET actual_visitors = ?, absolute_error = ?, "
                "waste_avoided_meals = ?, estimated_co2e_reduction_kg = ?, updated_at = ? WHERE id = ?",
                (
                    values["actual_visitors"],
                    values["absolute_error"],
                    values["waste_avoided_meals"],
                    values["estimated_co2e_reduction_kg"],
                    values["updated_at"],
                    row["id"],
                ),
            )
        conn.commit()
        return len(rows)


def _load_local_logs_for_location(location_id: str, limit: int) -> list[dict[str, Any]]:
    db = location_db_file(location_id)
    if not db.exists():
        return []
    with _connect(location_id) as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_logs WHERE location_id = ? "
            "ORDER BY prediction_created_at DESC LIMIT ?",
            (location_id, int(limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def _latest_logs(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("location_id", "")),
            str(row.get("service_date", "")),
            str(row.get("prediction_created_at", "")),
            int(row.get("id") or 0),
        ),
        reverse=True,
    )
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted_rows:
        key = (str(row.get("location_id", "")), str(row.get("service_date", "")))
        grouped_rows.setdefault(key, []).append(row)

    latest_rows: list[dict[str, Any]] = []
    for grouped in grouped_rows.values():
        latest = dict(grouped[0])
        if latest.get("actual_visitors") is None:
            latest["actual_visitors"] = _first_actual_visitors(grouped)
            _recalculate_absolute_error(latest)
        latest_rows.append(latest)

    latest_rows.sort(
        key=lambda row: (str(row.get("service_date", "")), str(row.get("prediction_created_at", ""))),
        reverse=True,
    )
    return latest_rows[: int(limit)]


def load_prediction_logs(location_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if _supabase_config():
        params = {
            "select": "*",
            "order": "service_date.desc,prediction_created_at.desc",
        }
        if location_id is not None:
            params["location_id"] = f"eq.{location_id}"
        rows = _supabase_request("GET", params=params)
        return _latest_logs(rows, limit) if isinstance(rows, list) else []

    if location_id is not None:
        return _latest_logs(_load_local_logs_for_location(location_id, max(int(limit) * 5, 1000)), limit)

    rows: list[dict[str, Any]] = []
    for location in list_locations():
        rows.extend(_load_local_logs_for_location(location.id, max(int(limit) * 5, 1000)))
    return _latest_logs(rows, limit)


def cleanup_logs_without_attendance(location_id: str, attendance_service_dates: Iterable[Any]) -> int:
    valid_dates = {_service_date(value) for value in attendance_service_dates}
    if _supabase_config():
        rows = _supabase_request(
            "GET",
            params={
                "select": "id,service_date",
                "location_id": f"eq.{location_id}",
            },
        )
        if not isinstance(rows, list):
            return 0
        stale_ids = [
            row["id"]
            for row in rows
            if isinstance(row, dict) and _service_date(row.get("service_date")) not in valid_dates
        ]
        for stale_id in stale_ids:
            _supabase_request(
                "DELETE",
                params={"id": f"eq.{stale_id}"},
                extra_headers={"Prefer": "return=minimal"},
            )
        return len(stale_ids)

    with _connect(location_id) as conn:
        rows = conn.execute(
            "SELECT id, service_date FROM prediction_logs WHERE location_id = ?",
            (location_id,),
        ).fetchall()
        stale_ids = [row["id"] for row in rows if _service_date(row["service_date"]) not in valid_dates]
        if stale_ids:
            placeholders = ", ".join("?" for _ in stale_ids)
            conn.execute(f"DELETE FROM prediction_logs WHERE id IN ({placeholders})", stale_ids)
            conn.commit()
        return len(stale_ids)


def summarize_monitoring(location_id: str | None = None) -> dict[str, float | int | None]:
    rows = load_prediction_logs(location_id=location_id, limit=1000)
    rows_with_actuals = [row for row in rows if row.get("actual_visitors") is not None]
    errors = [
        float(row["absolute_error"])
        for row in rows_with_actuals
        if row.get("absolute_error") is not None
    ]
    waste_values = [
        float(row["waste_avoided_meals"])
        for row in rows
        if row.get("waste_avoided_meals") is not None
    ]
    co2e_values = [
        float(row["estimated_co2e_reduction_kg"])
        for row in rows
        if row.get("estimated_co2e_reduction_kg") is not None
    ]
    return {
        "live_mae": (sum(errors) / len(errors)) if errors else None,
        "logs_with_actuals": len(rows_with_actuals),
        "total_estimated_waste_avoided_meals": sum(waste_values),
        "total_estimated_co2e_reduction_kg": sum(co2e_values),
    }
