import calendar
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class ServiceRecord:
    service_date: date
    visitors: int
    year: int
    month: int
    slot: str
    team: str



def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date | None:
    c = calendar.Calendar()
    hits = [d for d in c.itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    if n <= 0 or n > len(hits):
        return None
    return hits[n - 1]



def _parse_slot(slot: str) -> tuple[int, int] | None:
    if not isinstance(slot, str):
        return None
    m = re.match(r"(\d)(?:st|nd|rd|th)\s+(Sat|Sun)", slot.strip(), re.IGNORECASE)
    if not m:
        return None
    nth = int(m.group(1))
    weekday = 5 if m.group(2).lower() == "sat" else 6
    return nth, weekday



def _safe_int(v) -> int | None:
    if pd.isna(v):
        return None
    try:
        return int(float(v))
    except Exception:
        return None



def parse_year_sheet(df: pd.DataFrame, year: int) -> list[ServiceRecord]:
    records: list[ServiceRecord] = []
    header = df.iloc[2].tolist()
    month_cols = {m: header.index(m) for m in MONTHS if m in header}

    for ridx in range(3, len(df)):
        slot = df.iloc[ridx, 0]
        if isinstance(slot, str) and slot.strip().lower() == "totals":
            break
        parsed = _parse_slot(str(slot))
        if not parsed:
            continue
        nth, weekday = parsed
        team = str(df.iloc[ridx, 1]) if not pd.isna(df.iloc[ridx, 1]) else ""

        for month_name, cidx in month_cols.items():
            month = MONTHS.index(month_name) + 1
            visitors = _safe_int(df.iloc[ridx, cidx])
            if visitors is None:
                continue
            dt = nth_weekday_of_month(year, month, weekday, nth)
            if dt is None:
                continue
            records.append(
                ServiceRecord(
                    service_date=dt,
                    visitors=visitors,
                    year=year,
                    month=month,
                    slot=str(slot),
                    team=team,
                )
            )
    return records



def load_and_clean_raw_data(raw_file: str | Path) -> pd.DataFrame:
    xls = pd.ExcelFile(raw_file)
    all_records: list[ServiceRecord] = []

    for s in xls.sheet_names:
        if not s.isdigit():
            continue
        year = int(s)
        sheet = pd.read_excel(raw_file, sheet_name=s, header=None)
        all_records.extend(parse_year_sheet(sheet, year))

    if not all_records:
        raise ValueError("No service records parsed from Excel file")

    df = pd.DataFrame([r.__dict__ for r in all_records])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    df = df.drop_duplicates(subset=[DATE_COL, "slot"], keep="last")
    df = df[df[TARGET_COL] > 0]

    return df



def infer_next_service_date(history_dates: Iterable[pd.Timestamp]) -> pd.Timestamp:
    last = pd.to_datetime(max(history_dates))
    wd = last.weekday()

    if wd == 5:
        return last + pd.Timedelta(days=1)
    if wd == 6:
        return last + pd.Timedelta(days=6)

    days_to_sat = (5 - wd) % 7
    if days_to_sat == 0:
        days_to_sat = 7
    return last + pd.Timedelta(days=days_to_sat)



def slot_label_for_date(dt: pd.Timestamp) -> str:
    dt = pd.to_datetime(dt)
    weekday = "Sat" if dt.weekday() == 5 else "Sun"
    nth = (dt.day - 1) // 7 + 1
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(nth, "th")
    return f"{nth}{suffix} {weekday}"



def add_basic_calendar_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["year_num"] = d.dt.year
    out["month_num"] = d.dt.month
    out["day_num"] = d.dt.day
    out["weekday_num"] = d.dt.weekday
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["weekday_num"] >= 5).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    out["slot"] = d.apply(slot_label_for_date)
    out["slot_num"] = ((d.dt.day - 1) // 7 + 1).clip(1, 5)
    out["is_sun"] = (d.dt.weekday == 6).astype(int)
    return out
