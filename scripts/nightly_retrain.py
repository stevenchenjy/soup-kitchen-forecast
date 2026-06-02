from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_backtest import train_location
from src.config import artifact_dir_for_location, model_file_for_location
from src.data_admin import load_clean_data
from src.location_config import Location, list_locations
from src.model_training_runs import (
    create_training_run,
    latest_attendance_updated_at,
    latest_successful_training_run,
    supabase_configured,
    update_training_run,
)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_metrics(location_id: str) -> dict[str, Any] | None:
    metrics_path = artifact_dir_for_location(location_id) / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _selected_locations(args: argparse.Namespace) -> list[Location]:
    locations = list_locations()
    if args.all:
        return locations
    matches = [location for location in locations if location.id == args.location]
    if not matches:
        raise ValueError(f"Unknown location: {args.location}")
    return matches


def _should_skip_for_unchanged_attendance(
    location_id: str,
    latest_attendance_update: str | None,
    force: bool,
) -> str | None:
    if force:
        return None
    latest_success = latest_successful_training_run(location_id)
    if latest_success is None:
        return None

    previous_attendance_update = latest_success.get("latest_attendance_updated_at")
    current_dt = _parse_timestamp(latest_attendance_update)
    previous_dt = _parse_timestamp(previous_attendance_update)
    if current_dt is not None and previous_dt is not None and current_dt <= previous_dt:
        return "Attendance has not changed since the last successful training run."
    if latest_attendance_update and latest_attendance_update == previous_attendance_update:
        return "Attendance has not changed since the last successful training run."
    return None


def _record_skipped(
    location_id: str,
    attendance_rows: int,
    latest_attendance_update: str | None,
    reason: str,
) -> None:
    create_training_run(
        location_id=location_id,
        status="skipped",
        attendance_rows=attendance_rows,
        latest_attendance_updated_at_value=latest_attendance_update,
        model_path=_relative_path(model_file_for_location(location_id)),
        artifact_dir=_relative_path(artifact_dir_for_location(location_id)),
        commit_sha=os.getenv("GITHUB_SHA"),
        error_message=reason,
    )


def retrain_one_location(location: Location, args: argparse.Namespace) -> str:
    print(f"=== {location.id}: checking attendance ===", flush=True)
    attendance_df = load_clean_data(location.id)
    attendance_rows = int(len(attendance_df))
    latest_attendance_update = latest_attendance_updated_at(location.id)
    model_path = model_file_for_location(location.id)
    artifact_dir = artifact_dir_for_location(location.id)

    if attendance_rows < args.min_train_size:
        reason = f"Only {attendance_rows} attendance rows; need at least {args.min_train_size}."
        print(f"{location.id}: skipped - {reason}", flush=True)
        _record_skipped(location.id, attendance_rows, latest_attendance_update, reason)
        return "skipped"

    unchanged_reason = _should_skip_for_unchanged_attendance(location.id, latest_attendance_update, args.force)
    if unchanged_reason:
        print(f"{location.id}: skipped - {unchanged_reason}", flush=True)
        _record_skipped(location.id, attendance_rows, latest_attendance_update, unchanged_reason)
        return "skipped"

    run_id = create_training_run(
        location_id=location.id,
        status="running",
        attendance_rows=attendance_rows,
        latest_attendance_updated_at_value=latest_attendance_update,
        model_path=_relative_path(model_path),
        artifact_dir=_relative_path(artifact_dir),
        commit_sha=os.getenv("GITHUB_SHA"),
    )
    print(f"{location.id}: training started", flush=True)

    try:
        trained_model_path = train_location(
            location_id=location.id,
            min_train_size=args.min_train_size,
            quantile=args.quantile,
        )
        metrics = _load_metrics(location.id)
        if run_id is not None:
            update_training_run(
                run_id,
                status="success",
                finished_at=datetime.now(timezone.utc).isoformat(),
                attendance_rows=attendance_rows,
                latest_attendance_updated_at=latest_attendance_update,
                model_path=_relative_path(trained_model_path),
                artifact_dir=_relative_path(artifact_dir),
                commit_sha=os.getenv("GITHUB_SHA"),
                metrics=metrics,
                error_message=None,
            )
        print(f"{location.id}: success - {trained_model_path}", flush=True)
        return "success"
    except Exception as exc:
        error_message = f"{exc}\n{traceback.format_exc(limit=5)}"
        if run_id is not None:
            update_training_run(
                run_id,
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                attendance_rows=attendance_rows,
                latest_attendance_updated_at=latest_attendance_update,
                model_path=_relative_path(model_path),
                artifact_dir=_relative_path(artifact_dir),
                commit_sha=os.getenv("GITHUB_SHA"),
                error_message=error_message,
            )
        print(f"{location.id}: failed - {exc}", flush=True)
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly retraining for all configured locations.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--location", help="Train one location ID from data/locations.json")
    target.add_argument("--all", action="store_true", help="Check all configured locations")
    parser.add_argument("--force", action="store_true", help="Retrain even if attendance has not changed")
    parser.add_argument("--min-train-size", type=int, default=18)
    parser.add_argument("--quantile", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not supabase_configured():
        print("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.", flush=True)
        return 2

    locations = _selected_locations(args)
    print(f"Nightly retrain starting for {len(locations)} location(s). Force={args.force}", flush=True)
    results = {"success": 0, "skipped": 0, "failed": 0}
    for location in locations:
        try:
            status = retrain_one_location(location, args)
        except Exception as exc:
            status = "failed"
            print(f"{location.id}: failed before run record could be completed - {exc}", flush=True)
        results[status] += 1

    print(
        "Nightly retrain complete: "
        f"{results['success']} success, {results['skipped']} skipped, {results['failed']} failed.",
        flush=True,
    )
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
