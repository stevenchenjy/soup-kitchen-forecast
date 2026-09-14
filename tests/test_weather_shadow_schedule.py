from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.weather_shadow_schedule import (
    capture_due_services,
    next_preparation_cutoffs,
    preparation_cutoff,
)


@pytest.mark.parametrize("service_date,cutoff_utc,cutoff_local", [
    ("2026-09-19", "2026-09-19T02:00:00+00:00", "2026-09-18T22:00:00-04:00"),
    ("2026-09-20", "2026-09-20T02:00:00+00:00", "2026-09-19T22:00:00-04:00"),
    ("2026-01-10", "2026-01-10T03:00:00+00:00", "2026-01-09T22:00:00-05:00"),
    ("2026-01-11", "2026-01-11T03:00:00+00:00", "2026-01-10T22:00:00-05:00"),
    ("2026-03-08", "2026-03-08T02:00:00+00:00", "2026-03-07T21:00:00-05:00"),
    ("2026-11-01", "2026-11-01T03:00:00+00:00", "2026-10-31T23:00:00-04:00"),
])
def test_exact_elapsed_hours_including_dst(service_date, cutoff_utc, cutoff_local):
    cutoff = preparation_cutoff(service_date)
    assert cutoff.isoformat() == cutoff_utc
    zone = ZoneInfo("America/New_York")
    assert cutoff.astimezone(zone).isoformat() == cutoff_local
    anchor = datetime.combine(date.fromisoformat(service_date), datetime.min.time()).replace(hour=10, tzinfo=zone)
    assert anchor.astimezone(timezone.utc) - cutoff == timedelta(hours=12)
    assert preparation_cutoff(date.fromisoformat(service_date)) == cutoff


@pytest.mark.parametrize("service_date", ["2026-03-08", "2026-11-01", "2026-09-19", "2026-09-20"])
def test_due_interval_and_no_late_backfill(service_date):
    cutoff = preparation_cutoff(service_date)
    start = cutoff - timedelta(minutes=5)
    assert capture_due_services(start - timedelta(microseconds=1)) == []
    for instant in (start, start + timedelta(minutes=2), cutoff):
        rows = capture_due_services(instant)
        assert len(rows) == 1
        assert rows[0]["service_date"] == service_date
        assert rows[0]["cutoff_at"] == cutoff.isoformat()
        assert rows[0]["capture_at"] == start.isoformat()
    assert capture_due_services(cutoff + timedelta(microseconds=1)) == []
    assert capture_due_services(cutoff + timedelta(hours=12)) == []


def test_now_offset_does_not_change_due_services():
    instant = datetime.fromisoformat("2026-10-31T22:57:00-04:00")
    local = capture_due_services(instant)
    assert local == capture_due_services(instant.astimezone(timezone.utc))
    assert local[0]["service_date"] == "2026-11-01"
    assert local[0]["capture_local"] == "2026-10-31T22:55:00-04:00"


def test_configured_capture_start_and_allowed_window():
    cutoff = preparation_cutoff("2026-09-19")
    now = cutoff - timedelta(minutes=10)
    assert capture_due_services(now) == []
    assert capture_due_services(now, start_minutes_before_cutoff=10)[0]["service_date"] == "2026-09-19"
    assert capture_due_services(cutoff, start_minutes_before_cutoff=0)
    assert not capture_due_services(cutoff - timedelta(seconds=1), start_minutes_before_cutoff=0)


def test_next_cutoffs_include_dst_and_skip_elapsed_services():
    now = datetime.fromisoformat("2026-10-30T00:00:00+00:00")
    rows = next_preparation_cutoffs(now)
    assert [row["service_date"] for row in rows] == ["2026-10-31", "2026-11-01", "2026-11-07", "2026-11-08"]
    assert [row["cutoff_local"] for row in rows[:3]] == [
        "2026-10-30T22:00:00-04:00", "2026-10-31T23:00:00-04:00", "2026-11-06T22:00:00-05:00",
    ]
    cutoff = preparation_cutoff("2026-11-01")
    assert next_preparation_cutoffs(cutoff, count=1)[0]["service_date"] == "2026-11-07"


def test_next_cutoff_can_have_capture_already_due():
    cutoff = preparation_cutoff("2026-09-19")
    now = cutoff - timedelta(minutes=2)
    assert next_preparation_cutoffs(now, count=1) == capture_due_services(now)


@pytest.mark.parametrize("bad", ["2026-09-21", "2026-02-30", "2026-9-19", "20260919", "2026-09-19T10:00:00", None, 19, datetime(2026, 9, 19)])
def test_invalid_service_dates_rejected(bad):
    with pytest.raises(ValueError, match="service_date"):
        preparation_cutoff(bad)


@pytest.mark.parametrize("kwargs,match", [
    ({"timezone": "Not/A_Zone"}, "timezone"),
    ({"timezone": ""}, "timezone"),
    ({"timezone": None}, "timezone"),
    ({"anchor_time": "25:00"}, "anchor_time"),
    ({"anchor_time": "10:00:00"}, "anchor_time"),
    ({"anchor_time": "9:00"}, "anchor_time"),
    ({"anchor_time": None}, "anchor_time"),
    ({"lead_hours": 0}, "lead_hours"),
    ({"lead_hours": -1}, "lead_hours"),
    ({"lead_hours": float("nan")}, "lead_hours"),
    ({"lead_hours": float("inf")}, "lead_hours"),
    ({"lead_hours": "12"}, "lead_hours"),
    ({"lead_hours": True}, "lead_hours"),
])
def test_invalid_rule_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        preparation_cutoff("2026-09-19", **kwargs)


@pytest.mark.parametrize("service_date,anchor_time,match", [
    ("2026-03-08", "02:30", "does not exist"),
    ("2026-11-01", "01:30", "ambiguous"),
])
def test_custom_dst_anchor_must_be_unique(service_date, anchor_time, match):
    with pytest.raises(ValueError, match=match):
        preparation_cutoff(service_date, anchor_time=anchor_time)


@pytest.mark.parametrize("function", [capture_due_services, next_preparation_cutoffs])
@pytest.mark.parametrize("now", [datetime(2026, 9, 18, 22), "2026-09-18T22:00:00-04:00", None])
def test_now_requires_aware_datetime(function, now):
    with pytest.raises(ValueError, match="aware datetime"):
        function(now)


@pytest.mark.parametrize("kwargs", [
    {"capture_window_minutes": 0}, {"capture_window_minutes": 61},
    {"capture_window_minutes": 15.5}, {"capture_window_minutes": True},
    {"start_minutes_before_cutoff": -1}, {"start_minutes_before_cutoff": 16},
    {"start_minutes_before_cutoff": 2.5}, {"start_minutes_before_cutoff": True},
])
@pytest.mark.parametrize("function", [capture_due_services, next_preparation_cutoffs])
def test_invalid_capture_settings_rejected(kwargs, function):
    with pytest.raises(ValueError):
        function(datetime(2026, 9, 18, tzinfo=timezone.utc), **kwargs)


@pytest.mark.parametrize("count", [0, -1, True, 2.5])
def test_count_requires_positive_integer(count):
    with pytest.raises(ValueError, match="count"):
        next_preparation_cutoffs(datetime(2026, 9, 18, tzinfo=timezone.utc), count=count)
