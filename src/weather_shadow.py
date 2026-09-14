"""Prospective, append-only comparison of a weather candidate with active F6.

This module is an opt-in study runner. It never changes the staff recommendation,
active package, attendance database, or existing prediction log.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import DATE_COL, FORECAST_MAX_DAYS_AHEAD, model_file_for_location, parse_service_date
from src.location_config import get_location
from src.predictor import VisitorPredictor
from src.production_features import build_locked_f6_feature_row, service_horizon_between
from src.weather_candidate import history_sha256, load_weather_candidate, predict_weather_candidate
from src.weather_shadow_store import save_shadow_run
from src.weather_snapshots import FEATURE_CONTRACT_ID, extract_weather_features, fetch_weather_snapshot, snapshot_sha256


def utc_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("A timezone-aware preparation cutoff/clock is required.")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_snapshot(weather: dict, *, location_id: str, target: date, cutoff: datetime, zone: str) -> None:
    if weather.get("source") != "open_meteo_live_forecast" or weather.get("feature_contract_id") != FEATURE_CONTRACT_ID:
        raise ValueError("Expected the live weather snapshot contract.")
    if weather.get("snapshot_sha256") != snapshot_sha256(weather):
        raise ValueError("Weather snapshot checksum mismatch.")
    if weather.get("location_id") != location_id or weather.get("service_date") != target.isoformat() or weather.get("timezone") != zone:
        raise ValueError("Weather snapshot location/date/timezone mismatch.")
    if utc_datetime(weather["cutoff_at"]) != cutoff:
        raise ValueError("Weather snapshot cutoff mismatch.")
    if not utc_datetime(weather["request_started_at"]) <= utc_datetime(weather["retrieved_at"]) <= cutoff:
        raise ValueError("Invalid weather availability timestamps.")
    if weather["features"] != extract_weather_features(weather["raw_response"], target, zone):
        raise ValueError("Weather features do not match the saved hourly response.")


def _prediction(model: VisitorPredictor, row: pd.DataFrame, target: date) -> dict[str, Any]:
    segment = "sat" if target.weekday() == 5 else "sun"
    x = model.preprocessors[segment].transform(row)
    point = float(model.models[segment].predict(x)[0])
    quantile = float(model.quantile_models[segment].predict(x)[0])
    if not np.isfinite([point, quantile]).all() or min(point, quantile) < 0:
        raise ValueError("F6 produced invalid predictions.")
    return {
        "point": point, "q80": quantile, "suggested_meals": int(np.ceil(quantile)),
        "package_id": model.package_id, "model_segment": segment,
        "feature_order_sha256": model.feature_contract["feature_order_sha256"],
        "recommendation_policy_id": model.recommendation_policy_id,
    }


def capture_shadow_forecast(
    location_id: str,
    service_date: str | date,
    cutoff_at: str | datetime,
    candidate_path: str | Path,
    *,
    study_id: str = "weather_11_13_v1",
    baseline_path: str | Path | None = None,
    store: Any = None,
    now: Callable[[], datetime] | None = None,
    capture_window_minutes: int = 15,
) -> dict[str, Any]:
    """Capture within the last 15 minutes before the actual preparation cutoff.

    A missed cutoff is never backfilled using today's forecast. Calls starting
    after it fail without requesting weather. Calls that finish late are saved
    as cutoff_missed and excluded from paired evaluation. Persistence failures
    surface to the caller; failed weather/candidate runs retain the F6 result.
    """
    clock = now or (lambda: datetime.now(timezone.utc))
    started = utc_datetime(clock())
    cutoff = utc_datetime(cutoff_at)
    location = get_location(location_id)
    zone = ZoneInfo(location.timezone)
    target = parse_service_date(service_date, timezone=location.timezone)
    if not study_id.strip():
        raise ValueError("study_id cannot be empty.")
    if target.weekday() not in {5, 6}:
        raise ValueError("Service date must be Saturday or Sunday.")
    lead = (target - started.astimezone(zone).date()).days
    if not 0 <= lead < FORECAST_MAX_DAYS_AHEAD:
        raise ValueError("Service date must be within the live 16-day forecast range.")
    if cutoff >= datetime.combine(target, time(11), zone):
        raise ValueError("Preparation cutoff must be before 11 a.m. local service time.")
    if started > cutoff:
        raise ValueError("Preparation cutoff already passed; retrospective capture is forbidden.")
    if not 1 <= capture_window_minutes <= 60:
        raise ValueError("Capture window must be between 1 and 60 minutes.")
    if started < cutoff - timedelta(minutes=capture_window_minutes):
        raise ValueError("Too early: capture within the configured window before the preparation cutoff.")

    path = Path(baseline_path) if baseline_path is not None else model_file_for_location(location_id)
    before_hash = _sha256(path)
    baseline = VisitorPredictor(str(path))
    if before_hash != _sha256(path):
        raise ValueError("Active F6 package changed while loading; retry before cutoff.")
    if not baseline.uses_locked_f6:
        raise ValueError("Weather comparison requires the locked F6 baseline.")
    if (baseline.weather_zip_code, baseline.weather_country_code, baseline.weather_timezone) != (
        location.zip_code, location.country_code, location.timezone
    ):
        raise ValueError("Baseline location context differs from requested location.")
    origin = min(started.astimezone(zone).date(), target - timedelta(days=1))
    history_dates = pd.to_datetime(baseline.history_df[DATE_COL])
    if history_dates.empty or history_dates.max().date() > origin:
        raise ValueError("Baseline training history includes attendance after the forecast origin.")
    row = build_locked_f6_feature_row(baseline.history_df, target, origin)
    baseline_result = _prediction(baseline, row, target)
    history_bytes = baseline.history_df.sort_values(DATE_COL).to_csv(index=False).encode()
    payload: dict[str, Any] = {
        "schema_version": 1, "study_id": study_id, "timezone": location.timezone,
        "capture_started_at": started.isoformat(), "capture_window_minutes": capture_window_minutes,
        "forecast_origin": origin.isoformat(),
        "calendar_days_ahead": (target - origin).days,
        "service_horizon": service_horizon_between(origin, target),
        "cutoff_days_before_service": (target - cutoff.astimezone(zone).date()).days,
        "cutoff_local_time": cutoff.astimezone(zone).strftime("%H:%M:%S"),
        "baseline_model_sha256": before_hash,
        "attendance_history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "attendance_latest_service_date": history_dates.max().date().isoformat(),
        "f6_features": {name: float(value) for name, value in row.iloc[0].items()},
        "baseline": baseline_result, "candidate": None, "weather": None,
        "baseline_generated_at": utc_datetime(clock()).isoformat(),
        "eligible_for_evaluation": False,
    }
    # Normalize missing F6 feature values to JSON null; the fitted F6 imputer
    # still receives the original NaN row above and in the candidate below.
    payload["f6_features"] = {
        key: value if np.isfinite(value) else None for key, value in payload["f6_features"].items()
    }
    status = "weather_unavailable"
    weather_retrieved_at = None
    try:
        weather = fetch_weather_snapshot(target, cutoff, location, now=clock)
        _validate_snapshot(weather, location_id=location_id, target=target, cutoff=cutoff, zone=location.timezone)
        received = utc_datetime(weather["retrieved_at"])
        if received > cutoff or received < started:
            raise ValueError("Weather was not retrieved within this pre-cutoff capture.")
        payload["weather"] = weather
        weather_retrieved_at = received.isoformat()
    except Exception as exc:
        payload["failure"] = {"stage": "weather", "type": type(exc).__name__, "message": str(exc)[:500]}
    else:
        status = "candidate_unavailable"
        try:
            candidate_file = Path(candidate_path)
            if candidate_file.is_dir():
                candidate_file = candidate_file / "weather_candidate.joblib"
            candidate_hash = _sha256(candidate_file)
            package = load_weather_candidate(candidate_file, expected_location_id=location_id)
            if _sha256(candidate_file) != candidate_hash:
                raise ValueError("Candidate changed while loading.")
            if package["baseline_model_sha256"] != before_hash:
                raise ValueError("Candidate is stale for active F6; retrain it from this baseline package.")
            if package["history"]["sha256"] != history_sha256(baseline.history_df) or date.fromisoformat(package["training_end_date"]) > origin:
                raise ValueError("Candidate attendance history differs from the paired F6 training snapshot.")
            for coordinate in ("latitude", "longitude"):
                expected = package["weather_context"][coordinate]
                if abs(weather["geolocation"][coordinate] - expected) > 1e-4:
                    raise ValueError("Live weather requested coordinates differ from candidate training location.")
                if abs(weather["response_location"][coordinate] - expected) > .25:
                    raise ValueError("Live weather response grid differs from candidate training location.")
            if utc_datetime(package["created_at_utc"]) > started:
                raise ValueError("Candidate was not available when this capture started.")
            payload["candidate"] = predict_weather_candidate(package, row, weather["features"], target)
            payload["candidate_model_sha256"] = candidate_hash
            payload["candidate_created_at"] = package["created_at_utc"]
            payload["candidate_training_weather"] = package.get("training_weather", "realized_historical_bootstrap")
            payload["candidate_generated_at"] = utc_datetime(clock()).isoformat()
            status = "paired"
        except Exception as exc:
            payload["failure"] = {"stage": "candidate", "type": type(exc).__name__, "message": str(exc)[:500]}
    recorded_at = utc_datetime(clock())
    if recorded_at < started:
        raise ValueError("Clock moved backwards during capture.")
    if recorded_at > cutoff:
        status = "cutoff_missed"
    payload["eligible_for_evaluation"] = status == "paired"
    record = {
        "run_id": str(uuid4()), "location_id": location_id, "service_date": target.isoformat(),
        "cutoff_at": cutoff.isoformat(), "weather_retrieved_at": weather_retrieved_at,
        "recorded_at": recorded_at.isoformat(), "status": status, "payload": payload,
    }
    if store is None:
        save_shadow_run(record)
    else:
        store.append(record)
    return record


def evaluate_shadow_runs(
    runs: list[dict[str, Any]], attendance: pd.DataFrame, *, now: datetime | None = None
) -> dict[str, Any]:
    """Score latest eligible capture per study/date/cutoff, without changing logs.

    Different cutoff schedules and model versions receive separate summaries.
    Attendance CSV may include location_id and service_status; explicitly closed
    or unknown services are excluded. No missing service is filled with zero.
    """
    as_of = utc_datetime(now or datetime.now(timezone.utc))
    actuals = attendance.copy()
    if not {DATE_COL, "visitors"}.issubset(actuals.columns):
        raise ValueError("Attendance requires service_date and visitors columns.")
    if "location_id" not in actuals:
        locations = {record["location_id"] for record in runs}
        if len(locations) > 1:
            raise ValueError("Multi-location evaluation requires attendance location_id.")
        actuals["location_id"] = next(iter(locations), "")
    actuals[DATE_COL] = actuals[DATE_COL].map(lambda value: parse_service_date(value).isoformat())
    if actuals.duplicated(["location_id", DATE_COL]).any():
        raise ValueError("Attendance contains duplicate location/service dates; reconcile revisions first.")
    actuals["visitors"] = pd.to_numeric(actuals["visitors"], errors="raise")
    present = actuals.visitors.dropna().to_numpy(dtype=float)
    if not np.isfinite(present).all() or (present < 0).any() or (present % 1 != 0).any():
        raise ValueError("Attendance must be finite, nonnegative integer counts or missing.")
    lookup = actuals.set_index(["location_id", DATE_COL]).to_dict("index")
    latest: dict[tuple, dict] = {}
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for record in runs:
        payload = record["payload"]
        if record["status"] != "paired" or not payload.get("eligible_for_evaluation"):
            exclude(record["status"])
            continue
        try:
            cutoff = utc_datetime(record["cutoff_at"])
            generated = utc_datetime(record["recorded_at"])
            retrieved = utc_datetime(record["weather_retrieved_at"])
            started = utc_datetime(payload["capture_started_at"])
            persisted = utc_datetime(record["persisted_at"])
            if not started <= retrieved <= generated <= cutoff or not started <= persisted <= cutoff:
                raise ValueError("Invalid capture timing")
            if cutoff >= datetime.combine(date.fromisoformat(record["service_date"]), time(11), ZoneInfo(payload["timezone"])):
                raise ValueError("Cutoff is not before service")
            if not payload.get("baseline") or not payload.get("candidate") or not payload.get("weather"):
                raise ValueError("Missing paired input/output")
            _validate_snapshot(payload["weather"], location_id=record["location_id"], target=date.fromisoformat(record["service_date"]), cutoff=cutoff, zone=payload["timezone"])
            if utc_datetime(payload["weather"]["retrieved_at"]) != retrieved:
                raise ValueError("Outer and weather receipt timestamps differ")
            for field in ("baseline_generated_at", "candidate_generated_at"):
                if not started <= utc_datetime(payload[field]) <= generated:
                    raise ValueError("Prediction generation timestamp mismatch")
            if utc_datetime(payload["candidate_created_at"]) > started:
                raise ValueError("Candidate created after capture")
        except (KeyError, TypeError, ValueError):
            exclude("invalid_provenance")
            continue
        key = (payload["study_id"], record["location_id"], record["service_date"], cutoff.isoformat())
        if key in latest:
            exclude("superseded_capture")
            previous = latest[key]
            if (retrieved, generated, record["run_id"]) <= (
                utc_datetime(previous["weather_retrieved_at"]), utc_datetime(previous["recorded_at"]), previous["run_id"]
            ):
                continue
        latest[key] = record

    rows = []
    for record in latest.values():
        payload = record["payload"]
        actual_row = lookup.get((record["location_id"], record["service_date"]))
        service_end = datetime.combine(date.fromisoformat(record["service_date"]), time(13), ZoneInfo(payload["timezone"]))
        if as_of < service_end or actual_row is None:
            exclude("awaiting_actual")
            continue
        status = actual_row.get("service_status", "open")
        if pd.isna(status) or status != "open":
            exclude("service_not_confirmed_open")
            continue
        if pd.isna(actual_row["visitors"]):
            exclude("awaiting_actual")
            continue
        row = {
            "run_id": record["run_id"], "study_id": payload["study_id"], "location_id": record["location_id"],
            "service_date": record["service_date"], "cutoff_at": record["cutoff_at"],
            "cutoff_days_before_service": payload["cutoff_days_before_service"],
            "cutoff_local_time": payload["cutoff_local_time"], "daytype": payload["baseline"]["model_segment"],
            "baseline_package_id": payload["baseline"]["package_id"],
            "candidate_package_id": payload["candidate"]["package_id"], "actual": float(actual_row["visitors"]),
            "baseline_model_sha256": payload["baseline_model_sha256"],
            "candidate_model_sha256": payload["candidate_model_sha256"],
        }
        for label in ("baseline", "candidate"):
            prediction = payload[label]
            values = [float(prediction["point"]), float(prediction["q80"]), float(prediction["suggested_meals"])]
            if not np.isfinite(values).all() or min(values) < 0 or values[2] != np.ceil(values[1]):
                raise ValueError("Stored predictions violate the finite ceil(Q80) contract.")
            row.update({f"{label}_{key}": value for key, value in zip(["point", "q80", "meals"], values)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    strategy_summaries = []
    schedule_cols = ["study_id", "location_id", "cutoff_days_before_service", "cutoff_local_time", "daytype"]
    group_cols = schedule_cols + ["baseline_package_id", "candidate_package_id", "baseline_model_sha256", "candidate_model_sha256"]
    if not frame.empty:
        # Track the paired strategy across explicitly versioned refits as well as
        # each fitted version. Never call pooled strategy results one fixed model.
        for columns, destination in ((group_cols, summaries), (schedule_cols, strategy_summaries)):
            for group, data in frame.groupby(columns, sort=True):
                for label in ("baseline", "candidate"):
                    error = data[f"{label}_point"] - data.actual
                    residual = data.actual - data[f"{label}_q80"]
                    shortfall = np.maximum(data.actual - data[f"{label}_meals"], 0)
                    destination.append({
                        **dict(zip(columns, group)), "model": label, "n": len(data),
                        "fitted_versions": data[f"{label}_model_sha256"].nunique(),
                        "unique_service_dates": data.service_date.nunique(), "mae": float(abs(error).mean()),
                        "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
                        "q80_pinball": float(np.maximum(.8 * residual, -.2 * residual).mean()),
                        "q80_coverage": float((residual <= 0).mean()),
                        "shortfall_days": int((shortfall > 0).sum()), "shortfall_meals": float(shortfall.sum()),
                        "surplus_meals_per_service": float(np.maximum(data[f"{label}_meals"] - data.actual, 0).mean()),
                    })
    return {
        "as_of": as_of.isoformat(), "input_runs": len(runs), "paired_rows": len(rows),
        "exclusions": exclusions, "predictions": rows, "metrics": summaries,
        "strategy_metrics": strategy_summaries,
        "interpretation": "Prospective forecast evaluation only; no automatic promotion. Missing attendance is not zero. Different cutoffs and package versions are reported separately.",
    }
