from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


ATTENDANCE_TABLE_DEFAULT = "attendance"
MODEL_TRAINING_RUNS_TABLE_DEFAULT = "model_training_runs"


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


def _supabase_config() -> dict[str, str] | None:
    url = _secret_value("SUPABASE_URL", "url")
    key = _secret_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "service_role_key", "anon_key", "key")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def model_training_run_store_mode() -> str:
    return "supabase" if _supabase_config() else "unconfigured"


def supabase_configured() -> bool:
    return _supabase_config() is not None


def _table_name(env_name: str, secret_name: str, default: str) -> str:
    return _secret_value(env_name, secret_name) or default


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


def _supabase_url(table_name: str) -> str:
    config = _supabase_config()
    if config is None:
        raise RuntimeError("Supabase is not configured.")
    return f"{config['url']}/rest/v1/{table_name}"


def _supabase_request(
    table_name: str,
    method: str,
    params: dict[str, str] | None = None,
    payload: Any | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    url = _supabase_url(table_name)
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=_supabase_headers(extra_headers), method=method)
    with urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def _attendance_table() -> str:
    return _table_name("SUPABASE_ATTENDANCE_TABLE", "attendance_table", ATTENDANCE_TABLE_DEFAULT)


def _training_runs_table() -> str:
    return _table_name(
        "SUPABASE_MODEL_TRAINING_RUNS_TABLE",
        "model_training_runs_table",
        MODEL_TRAINING_RUNS_TABLE_DEFAULT,
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_attendance_updated_at(location_id: str) -> str | None:
    rows = _supabase_request(
        _attendance_table(),
        "GET",
        params={
            "select": "updated_at",
            "location_id": f"eq.{location_id}",
            "order": "updated_at.desc.nullslast",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0].get("updated_at")
    return None


def latest_training_run(location_id: str) -> dict[str, Any] | None:
    rows = _supabase_request(
        _training_runs_table(),
        "GET",
        params={
            "select": "*",
            "location_id": f"eq.{location_id}",
            "order": "started_at.desc.nullslast,id.desc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def latest_successful_training_run(location_id: str) -> dict[str, Any] | None:
    rows = _supabase_request(
        _training_runs_table(),
        "GET",
        params={
            "select": "*",
            "location_id": f"eq.{location_id}",
            "status": "eq.success",
            "order": "finished_at.desc.nullslast,started_at.desc.nullslast,id.desc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def create_training_run(
    *,
    location_id: str,
    status: str,
    attendance_rows: int | None = None,
    latest_attendance_updated_at_value: str | None = None,
    model_path: str | None = None,
    artifact_dir: str | None = None,
    commit_sha: str | None = None,
    metrics: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> int | None:
    payload = {
        "location_id": location_id,
        "status": status,
        "attendance_rows": attendance_rows,
        "latest_attendance_updated_at": latest_attendance_updated_at_value,
        "model_path": model_path,
        "artifact_dir": artifact_dir,
        "commit_sha": commit_sha,
        "metrics": metrics,
        "error_message": error_message,
    }
    rows = _supabase_request(
        _training_runs_table(),
        "POST",
        payload=payload,
        extra_headers={"Prefer": "return=representation"},
    )
    if isinstance(rows, list) and rows:
        return rows[0].get("id")
    return None


def update_training_run(run_id: int, **fields: Any) -> None:
    _supabase_request(
        _training_runs_table(),
        "PATCH",
        params={"id": f"eq.{run_id}"},
        payload=fields,
        extra_headers={"Prefer": "return=minimal"},
    )
