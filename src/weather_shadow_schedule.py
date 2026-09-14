"""Preparation cutoffs and capture times for the weekend weather shadow study.

The confirmed rule is twelve *elapsed* hours before 10 a.m. on the service date
in Newburgh. Subtracting in UTC preserves that duration across daylight-saving
changes; it is deliberately not a fixed 10 p.m. wall-clock schedule.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone as dt_timezone
import math
from numbers import Real
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _service_date(value: str | date) -> date:
    if isinstance(value, datetime):
        raise ValueError("service_date must be an explicit calendar date, not a datetime.")
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("service_date must be a YYYY-MM-DD calendar date.")
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("service_date must be a valid YYYY-MM-DD calendar date.") from exc
    if not isinstance(value, date):
        raise ValueError("service_date must be a YYYY-MM-DD string or date.")
    if value.weekday() not in (5, 6):
        raise ValueError("service_date must be a Saturday or Sunday.")
    return value


def _rule(timezone: str, anchor_time: str, lead_hours: Real) -> tuple[ZoneInfo, time, timedelta]:
    try:
        if not isinstance(timezone, str) or not timezone:
            raise ValueError("missing timezone")
        zone = ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("timezone must be a valid IANA timezone name.") from exc
    if not isinstance(anchor_time, str) or not re.fullmatch(r"\d{2}:\d{2}", anchor_time):
        raise ValueError("anchor_time must be local time in HH:MM format.")
    try:
        anchor = time.fromisoformat(anchor_time)
    except ValueError as exc:
        raise ValueError("anchor_time must be a valid local time in HH:MM format.") from exc
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, Real) or not math.isfinite(lead_hours) or lead_hours <= 0:
        raise ValueError("lead_hours must be a positive finite number of elapsed hours.")
    try:
        lead = timedelta(hours=float(lead_hours))
    except OverflowError as exc:
        raise ValueError("lead_hours exceeds the supported date range.") from exc
    return zone, anchor, lead


def _cutoff(target: date, zone: ZoneInfo, anchor: time, lead: timedelta) -> datetime:
    naive = datetime.combine(target, anchor)
    local = naive.replace(tzinfo=zone)
    # Custom anchors must identify a unique real instant, including on DST days.
    if local.astimezone(dt_timezone.utc).astimezone(zone).replace(tzinfo=None) != naive:
        raise ValueError("anchor_time does not exist on this service date in the configured timezone.")
    alternate = naive.replace(tzinfo=zone, fold=1)
    if local.utcoffset() != alternate.utcoffset():
        raise ValueError("anchor_time is ambiguous on this service date in the configured timezone.")
    try:
        return local.astimezone(dt_timezone.utc) - lead
    except OverflowError as exc:
        raise ValueError("Preparation cutoff exceeds the supported date range.") from exc


def preparation_cutoff(
    service_date: str | date,
    timezone: str = "America/New_York",
    anchor_time: str = "10:00",
    lead_hours: Real = 12,
) -> datetime:
    """Return the aware UTC cutoff exactly ``lead_hours`` before the local anchor.

    Ordinary Saturday/Sunday cutoffs are 10 p.m. the previous day. For the
    Sunday spring/fall transitions they are 9 p.m./11 p.m., respectively.
    """
    return _cutoff(_service_date(service_date), *_rule(timezone, anchor_time, lead_hours))


def _aware_now(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be an aware datetime with a UTC offset.")
    return now.astimezone(dt_timezone.utc)


def _capture_lead(capture_window_minutes: int, start_minutes_before_cutoff: int) -> timedelta:
    if isinstance(capture_window_minutes, bool) or not isinstance(capture_window_minutes, int) or not 1 <= capture_window_minutes <= 60:
        raise ValueError("capture_window_minutes must be an integer from 1 to 60.")
    if isinstance(start_minutes_before_cutoff, bool) or not isinstance(start_minutes_before_cutoff, int) or not 0 <= start_minutes_before_cutoff <= capture_window_minutes:
        raise ValueError("start_minutes_before_cutoff must be an integer from 0 through capture_window_minutes.")
    return timedelta(minutes=start_minutes_before_cutoff)


def _entry(target: date, cutoff: datetime, zone: ZoneInfo, capture_lead: timedelta) -> dict[str, str]:
    capture = cutoff - capture_lead
    return {
        "service_date": target.isoformat(),
        "cutoff_at": cutoff.isoformat(),
        "capture_at": capture.isoformat(),
        "cutoff_local": cutoff.astimezone(zone).isoformat(),
        "capture_local": capture.astimezone(zone).isoformat(),
    }


def capture_due_services(
    now: datetime,
    *,
    timezone: str = "America/New_York",
    anchor_time: str = "10:00",
    lead_hours: Real = 12,
    capture_window_minutes: int = 15,
    start_minutes_before_cutoff: int = 5,
) -> list[dict[str, str]]:
    """List services whose capture start has arrived and whose cutoff has not passed.

    The default due interval starts five minutes before cutoff and includes the
    cutoff instant. It never allows late backfill. ``capture_window_minutes``
    checks that the scheduled start fits the capture workflow's allowed window;
    it does not move the scheduled start fifteen minutes earlier.
    """
    instant = _aware_now(now)
    zone, anchor, lead = _rule(timezone, anchor_time, lead_hours)
    capture_lead = _capture_lead(capture_window_minutes, start_minutes_before_cutoff)
    # Undo the elapsed-hour lead to locate the corresponding service calendar.
    first = (instant + lead).astimezone(zone).date() - timedelta(days=1)
    result = []
    for offset in range(3):
        target = first + timedelta(days=offset)
        if target.weekday() not in (5, 6):
            continue
        cutoff = _cutoff(target, zone, anchor, lead)
        if cutoff - capture_lead <= instant <= cutoff:
            result.append(_entry(target, cutoff, zone, capture_lead))
    return result


def next_preparation_cutoffs(
    now: datetime,
    count: int = 4,
    *,
    timezone: str = "America/New_York",
    anchor_time: str = "10:00",
    lead_hours: Real = 12,
    capture_window_minutes: int = 15,
    start_minutes_before_cutoff: int = 5,
) -> list[dict[str, str]]:
    """List the next ``count`` service cutoffs strictly after an aware ``now``.

    A listed capture time may already have arrived if its cutoff is still in the
    future; use ``capture_due_services`` to determine which capture can run now.
    """
    instant = _aware_now(now)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    zone, anchor, lead = _rule(timezone, anchor_time, lead_hours)
    capture_lead = _capture_lead(capture_window_minutes, start_minutes_before_cutoff)
    target = (instant + lead).astimezone(zone).date() - timedelta(days=1)
    result = []
    while len(result) < count:
        if target.weekday() in (5, 6):
            cutoff = _cutoff(target, zone, anchor, lead)
            if cutoff > instant:
                result.append(_entry(target, cutoff, zone, capture_lead))
        target += timedelta(days=1)
    return result
