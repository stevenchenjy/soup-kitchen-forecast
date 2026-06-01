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

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


ATTENDANCE_TABLE_DEFAULT = "attendance"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    service_date TEXT PRIMARY KEY,
    visitors INTEGER NOT NULL CHECK (visitors >= 0)
);
"""



def _connect(location_id: str) -> sqlite3.Connection:
    db = location_db_file(location_id)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(SCHEMA_SQL)
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


def _upsert_record_sqlite(service_date: str, visitors: int, location_id: str) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    v = int(visitors)
    with _connect(location_id) as conn:
        conn.execute(
            "INSERT INTO attendance(service_date, visitors) VALUES(?, ?) "
            "ON CONFLICT(service_date) DO UPDATE SET visitors=excluded.visitors",
            (dt, v),
        )
        conn.commit()
    return load_clean_data(location_id)


def _delete_record_sqlite(service_date: str, location_id: str) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    with _connect(location_id) as conn:
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

    for stale_date in existing_dates - incoming_dates:
        _delete_supabase_row(location_id, stale_date)

    if out.empty:
        return

    now = datetime.now(timezone.utc).isoformat()
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


def _upsert_record_supabase(service_date: str, visitors: int, location_id: str) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
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
    return load_clean_data(location_id)


def _delete_record_supabase(service_date: str, location_id: str) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    _delete_supabase_row(location_id, dt)
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



def upsert_record(service_date: str, visitors: int, location_id: str) -> pd.DataFrame:
    if _supabase_config():
        return _upsert_record_supabase(service_date, visitors, location_id)
    return _upsert_record_sqlite(service_date, visitors, location_id)



def delete_record(service_date: str, location_id: str) -> pd.DataFrame:
    if _supabase_config():
        return _delete_record_supabase(service_date, location_id)
    return _delete_record_sqlite(service_date, location_id)
