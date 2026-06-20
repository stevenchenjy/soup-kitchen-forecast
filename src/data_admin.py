from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.config import CLEAN_DATA_FILE, DATE_COL, TARGET_COL, location_db_file
from src.model_training_runs import mark_location_dirty

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


ATTENDANCE_TABLE_DEFAULT = "attendance"
ATTENDANCE_CHANGE_LOG_TABLE_DEFAULT = "attendance_change_log"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    service_date TEXT PRIMARY KEY,
    visitors INTEGER NOT NULL CHECK (visitors >= 0)
);
CREATE TABLE IF NOT EXISTS attendance_change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id TEXT NOT NULL,
    service_date TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('ADD', 'UPDATE', 'DELETE')),
    previous_visitors INTEGER,
    new_visitors INTEGER,
    changed_by TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attendance_change_log_undo
ON attendance_change_log(location_id, changed_by, created_at DESC, id DESC);
"""



def _connect(location_id: str) -> sqlite3.Connection:
    db = location_db_file(location_id)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _streamlit_secret_value(*names: str) -> str | None:
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


def _env_secret_value(*names: str) -> str | None:
    for name in names:
        env_value = os.getenv(name)
        if env_value:
            return env_value
    return None


def _secret_value(*names: str) -> str | None:
    return _streamlit_secret_value(*names) or _env_secret_value(*names)


def _supabase_config_for_source(source: str) -> dict[str, str] | None:
    getter = _streamlit_secret_value if source == "streamlit secrets" else _env_secret_value
    url = getter("SUPABASE_URL", "url")
    key = getter("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "service_role_key", "anon_key", "key")
    table = getter("SUPABASE_ATTENDANCE_TABLE", "attendance_table") or ATTENDANCE_TABLE_DEFAULT
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key, "table": table}


def _supabase_config() -> dict[str, str] | None:
    streamlit_config = _supabase_config_for_source("streamlit secrets")
    if streamlit_config:
        return streamlit_config
    return _supabase_config_for_source("environment variables")


def attendance_config_source() -> str:
    if _supabase_config_for_source("streamlit secrets"):
        return "streamlit secrets"
    if _supabase_config_for_source("environment variables"):
        return "environment variables"
    return "fallback"


def attendance_store_mode() -> str:
    return "supabase" if _supabase_config() else "sqlite"


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


def _supabase_url(table: str | None = None) -> str:
    config = _supabase_config()
    if config is None:
        raise RuntimeError("Supabase is not configured.")
    return f"{config['url']}/rest/v1/{table or config['table']}"


def _supabase_request(
    method: str,
    params: dict[str, str] | None = None,
    payload: Any | None = None,
    extra_headers: dict[str, str] | None = None,
    table: str | None = None,
) -> Any:
    url = _supabase_url(table)
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=_supabase_headers(extra_headers), method=method)
    with urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def _change_log_table() -> str:
    return _secret_value("SUPABASE_ATTENDANCE_CHANGE_LOG_TABLE", "attendance_change_log_table") or ATTENDANCE_CHANGE_LOG_TABLE_DEFAULT


def _change_row(
    location_id: str,
    service_date: str,
    operation: str,
    previous_visitors: int | None,
    new_visitors: int | None,
    changed_by: str,
) -> dict[str, Any]:
    return {
        "location_id": location_id,
        "service_date": service_date,
        "operation": operation,
        "previous_visitors": previous_visitors,
        "new_visitors": new_visitors,
        "changed_by": changed_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _insert_change_supabase(change: dict[str, Any]) -> int | str | None:
    rows = _supabase_request(
        "POST",
        payload=change,
        extra_headers={"Prefer": "return=representation"},
        table=_change_log_table(),
    )
    return rows[0].get("id") if isinstance(rows, list) and rows else None


def _delete_change_supabase(change_id: int | str) -> None:
    _supabase_request(
        "DELETE",
        params={"id": f"eq.{change_id}"},
        extra_headers={"Prefer": "return=minimal"},
        table=_change_log_table(),
    )


def _mark_location_dirty(location_id: str, attendance_updated_at: str | None = None) -> None:
    try:
        mark_location_dirty(location_id, attendance_updated_at)
    except Exception:
        pass


def _normalize_attendance_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.strftime("%Y-%m-%d")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL, TARGET_COL])
    out[TARGET_COL] = out[TARGET_COL].round().astype(int)
    out = out[out[TARGET_COL] >= 0]
    return out.drop_duplicates(subset=[DATE_COL], keep="last").sort_values(DATE_COL)


def _format_attendance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[DATE_COL, TARGET_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df[[DATE_COL, TARGET_COL]].reset_index(drop=True)


def _load_clean_data_sqlite(location_id: str) -> pd.DataFrame:
    with _connect(location_id) as conn:
        df = pd.read_sql_query("SELECT service_date, visitors FROM attendance ORDER BY service_date", conn)
    return _format_attendance_df(df)


def _save_clean_data_sqlite(df: pd.DataFrame, location_id: str) -> None:
    out = _normalize_attendance_df(df)
    with _connect(location_id) as conn:
        conn.execute("DELETE FROM attendance")
        conn.executemany(
            "INSERT INTO attendance(service_date, visitors) VALUES(?, ?)",
            list(out[[DATE_COL, TARGET_COL]].itertuples(index=False, name=None)),
        )
        conn.commit()


def _upsert_record_sqlite(
    service_date: str,
    visitors: int,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    v = int(visitors)
    with _connect(location_id) as conn:
        previous = conn.execute("SELECT visitors FROM attendance WHERE service_date = ?", (dt,)).fetchone()
        if changed_by:
            operation = "UPDATE" if previous else "ADD"
            change = _change_row(location_id, dt, operation, int(previous["visitors"]) if previous else None, v, changed_by)
            conn.execute(
                "INSERT INTO attendance_change_log "
                "(location_id, service_date, operation, previous_visitors, new_visitors, changed_by, created_at) "
                "VALUES (:location_id, :service_date, :operation, :previous_visitors, :new_visitors, :changed_by, :created_at)",
                change,
            )
        conn.execute(
            "INSERT INTO attendance(service_date, visitors) VALUES(?, ?) "
            "ON CONFLICT(service_date) DO UPDATE SET visitors=excluded.visitors",
            (dt, v),
        )
        conn.commit()
    return load_clean_data(location_id)


def _delete_record_sqlite(
    service_date: str,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    with _connect(location_id) as conn:
        previous = conn.execute("SELECT visitors FROM attendance WHERE service_date = ?", (dt,)).fetchone()
        if changed_by and previous:
            change = _change_row(location_id, dt, "DELETE", int(previous["visitors"]), None, changed_by)
            conn.execute(
                "INSERT INTO attendance_change_log "
                "(location_id, service_date, operation, previous_visitors, new_visitors, changed_by, created_at) "
                "VALUES (:location_id, :service_date, :operation, :previous_visitors, :new_visitors, :changed_by, :created_at)",
                change,
            )
        conn.execute("DELETE FROM attendance WHERE service_date = ?", (dt,))
        conn.commit()
    return load_clean_data(location_id)


def _load_clean_data_supabase(location_id: str) -> pd.DataFrame:
    rows = _supabase_request(
        "GET",
        params={
            "select": "service_date,visitors",
            "location_id": f"eq.{location_id}",
            "order": "service_date.asc",
        },
    )
    if not isinstance(rows, list):
        return pd.DataFrame(columns=[DATE_COL, TARGET_COL])
    return _format_attendance_df(pd.DataFrame(rows))


def _delete_supabase_row(location_id: str, service_date: str) -> None:
    _supabase_request(
        "DELETE",
        params={
            "location_id": f"eq.{location_id}",
            "service_date": f"eq.{service_date}",
        },
        extra_headers={"Prefer": "return=minimal"},
    )


def _save_clean_data_supabase(df: pd.DataFrame, location_id: str) -> None:
    out = _normalize_attendance_df(df)
    existing_rows = _supabase_request(
        "GET",
        params={
            "select": "service_date",
            "location_id": f"eq.{location_id}",
        },
    )
    existing_dates = {
        str(row.get("service_date"))
        for row in existing_rows
        if isinstance(row, dict) and row.get("service_date")
    } if isinstance(existing_rows, list) else set()
    incoming_dates = set(out[DATE_COL].tolist())

    now = datetime.now(timezone.utc).isoformat()
    for stale_date in existing_dates - incoming_dates:
        _delete_supabase_row(location_id, stale_date)

    if out.empty:
        _mark_location_dirty(location_id, now)
        return

    rows = [
        {
            "location_id": location_id,
            "service_date": row[DATE_COL],
            "visitors": int(row[TARGET_COL]),
            "updated_at": now,
        }
        for row in out[[DATE_COL, TARGET_COL]].to_dict(orient="records")
    ]
    _supabase_request(
        "POST",
        params={"on_conflict": "location_id,service_date"},
        payload=rows,
        extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
    )
    _mark_location_dirty(location_id, now)


def _upsert_record_supabase(
    service_date: str,
    visitors: int,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    change_id = None
    if changed_by:
        existing = _supabase_request(
            "GET",
            params={"select": "visitors", "location_id": f"eq.{location_id}", "service_date": f"eq.{dt}", "limit": "1"},
        )
        previous = int(existing[0]["visitors"]) if isinstance(existing, list) and existing else None
        change_id = _insert_change_supabase(
            _change_row(
                location_id,
                dt,
                "UPDATE" if previous is not None else "ADD",
                previous,
                int(visitors),
                changed_by,
            )
        )
    try:
        _supabase_request(
            "POST",
            params={"on_conflict": "location_id,service_date"},
            payload={
                "location_id": location_id,
                "service_date": dt,
                "visitors": int(visitors),
                "updated_at": now,
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
    except Exception:
        # A failed write must not leave a rollback entry for a change that never happened.
        if change_id is not None:
            _delete_change_supabase(change_id)
        raise
    _mark_location_dirty(location_id, now)
    return load_clean_data(location_id)


def _delete_record_supabase(
    service_date: str,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    change_id = None
    if changed_by:
        existing = _supabase_request(
            "GET",
            params={"select": "visitors", "location_id": f"eq.{location_id}", "service_date": f"eq.{dt}", "limit": "1"},
        )
        if isinstance(existing, list) and existing:
            change_id = _insert_change_supabase(
                _change_row(location_id, dt, "DELETE", int(existing[0]["visitors"]), None, changed_by)
            )
    try:
        _delete_supabase_row(location_id, dt)
    except Exception:
        if change_id is not None:
            _delete_change_supabase(change_id)
        raise
    _mark_location_dirty(location_id, datetime.now(timezone.utc).isoformat())
    return load_clean_data(location_id)



def bootstrap_location_from_csv(location_id: str, csv_path: str | Path = CLEAN_DATA_FILE) -> None:
    if _supabase_config():
        if not load_clean_data(location_id).empty:
            return
    else:
        db = location_db_file(location_id)
        if db.exists():
            with _connect(location_id) as conn:
                n = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
                if n > 0:
                    return

    p = Path(csv_path)
    if not p.exists():
        return
    df = pd.read_csv(p, parse_dates=[DATE_COL])
    if df.empty:
        return
    save_clean_data(df[[DATE_COL, TARGET_COL]], location_id=location_id)



def load_clean_data(location_id: str) -> pd.DataFrame:
    if _supabase_config():
        return _load_clean_data_supabase(location_id)
    return _load_clean_data_sqlite(location_id)



def save_clean_data(df: pd.DataFrame, location_id: str) -> None:
    if _supabase_config():
        _save_clean_data_supabase(df, location_id)
        return
    _save_clean_data_sqlite(df, location_id)



def upsert_record(
    service_date: str,
    visitors: int,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    if _supabase_config():
        return _upsert_record_supabase(service_date, visitors, location_id, changed_by)
    return _upsert_record_sqlite(service_date, visitors, location_id, changed_by)



def delete_record(
    service_date: str,
    location_id: str,
    changed_by: str | None = None,
) -> pd.DataFrame:
    if _supabase_config():
        return _delete_record_supabase(service_date, location_id, changed_by)
    return _delete_record_sqlite(service_date, location_id, changed_by)


def latest_attendance_change(location_id: str, changed_by: str) -> dict[str, Any] | None:
    if _supabase_config():
        rows = _supabase_request(
            "GET",
            params={
                "select": "*",
                "location_id": f"eq.{location_id}",
                "changed_by": f"eq.{changed_by}",
                "order": "created_at.desc,id.desc",
                "limit": "1",
            },
            table=_change_log_table(),
        )
        return dict(rows[0]) if isinstance(rows, list) and rows else None

    with _connect(location_id) as conn:
        row = conn.execute(
            "SELECT * FROM attendance_change_log WHERE location_id = ? AND changed_by = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (location_id, changed_by),
        ).fetchone()
    return dict(row) if row else None


def _apply_undo(change: dict[str, Any]) -> None:
    location_id = str(change["location_id"])
    service_date = str(change["service_date"])
    operation = str(change["operation"])
    if operation == "ADD":
        if _supabase_config():
            _delete_supabase_row(location_id, service_date)
        else:
            with _connect(location_id) as conn:
                conn.execute("DELETE FROM attendance WHERE service_date = ?", (service_date,))
                conn.commit()
        return

    previous_visitors = int(change["previous_visitors"])
    if _supabase_config():
        _supabase_request(
            "POST",
            params={"on_conflict": "location_id,service_date"},
            payload={
                "location_id": location_id,
                "service_date": service_date,
                "visitors": previous_visitors,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
    else:
        with _connect(location_id) as conn:
            conn.execute(
                "INSERT INTO attendance(service_date, visitors) VALUES(?, ?) "
                "ON CONFLICT(service_date) DO UPDATE SET visitors=excluded.visitors",
                (service_date, previous_visitors),
            )
            conn.commit()


def _consume_undo_history(location_id: str, changed_by: str) -> None:
    if _supabase_config():
        _supabase_request(
            "DELETE",
            params={"location_id": f"eq.{location_id}", "changed_by": f"eq.{changed_by}"},
            extra_headers={"Prefer": "return=minimal"},
            table=_change_log_table(),
        )
        return
    with _connect(location_id) as conn:
        conn.execute(
            "DELETE FROM attendance_change_log WHERE location_id = ? AND changed_by = ?",
            (location_id, changed_by),
        )
        conn.commit()


def undo_last_attendance_input(location_id: str, changed_by: str) -> dict[str, Any] | None:
    change = latest_attendance_change(location_id, changed_by)
    if change is None:
        return None

    _apply_undo(change)

    # Keep prediction monitoring aligned before consuming the only available undo.
    from src.prediction_logs import set_prediction_logs_actual

    actual = None if change["operation"] == "ADD" else int(change["previous_visitors"])
    set_prediction_logs_actual(location_id, str(change["service_date"]), actual)
    _consume_undo_history(location_id, changed_by)
    _mark_location_dirty(location_id, datetime.now(timezone.utc).isoformat())
    return change
