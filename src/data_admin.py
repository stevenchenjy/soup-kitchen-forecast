from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA_FILE, DATE_COL, TARGET_COL, location_db_file

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



def bootstrap_location_from_csv(location_id: str, csv_path: str | Path = CLEAN_DATA_FILE) -> None:
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
    with _connect(location_id) as conn:
        df = pd.read_sql_query("SELECT service_date, visitors FROM attendance ORDER BY service_date", conn)
    if df.empty:
        return pd.DataFrame(columns=[DATE_COL, TARGET_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)



def save_clean_data(df: pd.DataFrame, location_id: str) -> None:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.strftime("%Y-%m-%d")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL, TARGET_COL])
    out[TARGET_COL] = out[TARGET_COL].round().astype(int)
    out = out[out[TARGET_COL] >= 0]
    out = out.drop_duplicates(subset=[DATE_COL], keep="last").sort_values(DATE_COL)

    with _connect(location_id) as conn:
        conn.execute("DELETE FROM attendance")
        conn.executemany(
            "INSERT INTO attendance(service_date, visitors) VALUES(?, ?)",
            list(out[[DATE_COL, TARGET_COL]].itertuples(index=False, name=None)),
        )
        conn.commit()



def upsert_record(service_date: str, visitors: int, location_id: str) -> pd.DataFrame:
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



def delete_record(service_date: str, location_id: str) -> pd.DataFrame:
    dt = pd.to_datetime(service_date).strftime("%Y-%m-%d")
    with _connect(location_id) as conn:
        conn.execute("DELETE FROM attendance WHERE service_date = ?", (dt,))
        conn.commit()
    return load_clean_data(location_id)
