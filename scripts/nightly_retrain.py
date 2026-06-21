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
    clear_location_dirty,
    create_training_run,
    get_retrain_state,
    latest_attendance_updated_at,
    list_dirty_location_ids,
    mark_location_dirty,
    supabase_configured,
)


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


def _locations_by_id() -> dict[str, Location]:
    return {location.id: location for location in list_locations()}


def _selected_locations(args: argparse.Namespace) -> list[Location]:
    locations = list_locations()
    if args.all:
        return locations
    matches = [location for location in locations if location.id == args.location]
    if not matches:
        raise ValueError(f"Unknown location: {args.location}")
    return matches


def _dirty_locations(args: argparse.Namespace) -> list[Location]:
    locations_by_id = _locations_by_id()
    if args.location and args.location not in locations_by_id:
        raise ValueError(f"Unknown location: {args.location}")
    if args.force:
        return _selected_locations(args)

    dirty_ids = set(list_dirty_location_ids())
    if args.location:
        dirty_ids &= {args.location}
    locations = [locations_by_id[location_id] for location_id in dirty_ids if location_id in locations_by_id]
    locations.sort(key=lambda location: location.id)
    return locations


def _latest_attendance_update(location_id: str) -> str | None:
    state = get_retrain_state(location_id)
    if state and state.get("last_attendance_updated_at"):
        return state["last_attendance_updated_at"]
    return latest_attendance_updated_at(location_id)


def _print_summary(
    *,
    locations_checked: int,
    trained_locations: list[str],
    failed_locations: list[str],
    reason: str | None = None,
) -> None:
    lines = [
        "Nightly retrain summary:",
        f"- Locations checked: {locations_checked}",
        f"- Locations trained: {len(trained_locations)}",
    ]
    if trained_locations:
        lines.append(f"- Trained locations: {', '.join(trained_locations)}")
    lines.append(f"- Locations failed: {len(failed_locations)}")
    if failed_locations:
        lines.append(f"- Failed locations: {', '.join(failed_locations)}")
    if reason:
        lines.append(f"- Reason: {reason}")

    summary = "\n".join(lines)
    print(summary, flush=True)

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        try:
            with Path(step_summary_path).open("a", encoding="utf-8") as step_summary:
                step_summary.write(summary + "\n")
        except OSError as exc:
            print(f"Warning: could not write GitHub Actions step summary: {exc}", flush=True)


def retrain_one_location(location: Location, args: argparse.Namespace) -> str:
    print(f"=== {location.id}: checking attendance ===", flush=True)
    attendance_df = load_clean_data(location.id)
    attendance_rows = int(len(attendance_df))
    latest_attendance_update = _latest_attendance_update(location.id)
    model_path = model_file_for_location(location.id)
    artifact_dir = artifact_dir_for_location(location.id)

    print(f"{location.id}: training started", flush=True)

    try:
        if attendance_rows < args.min_train_size:
            raise ValueError(f"Only {attendance_rows} attendance rows; need at least {args.min_train_size}.")
        trained_model_path = train_location(
            location_id=location.id,
            min_train_size=args.min_train_size,
            quantile=args.quantile,
        )
        metrics = _load_metrics(location.id)
        finished_at = datetime.now(timezone.utc).isoformat()
        create_training_run(
            location_id=location.id,
            status="success",
            finished_at=finished_at,
            attendance_rows=attendance_rows,
            latest_attendance_updated_at_value=latest_attendance_update,
            model_path=_relative_path(trained_model_path),
            artifact_dir=_relative_path(artifact_dir),
            commit_sha=os.getenv("GITHUB_SHA"),
            metrics=metrics,
            error_message=None,
        )
        clear_location_dirty(location.id, finished_at)
        print(f"{location.id}: success - {trained_model_path}", flush=True)
        return "success"
    except Exception as exc:
        error_message = f"{exc}\n{traceback.format_exc(limit=5)}"
        mark_location_dirty(location.id, latest_attendance_update)
        create_training_run(
            location_id=location.id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            attendance_rows=attendance_rows,
            latest_attendance_updated_at_value=latest_attendance_update,
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

    locations_checked = len(_selected_locations(args))
    locations = _dirty_locations(args)
    if not locations:
        _print_summary(
            locations_checked=locations_checked,
            trained_locations=[],
            failed_locations=[],
            reason="no dirty locations to retrain",
        )
        return 0

    print(f"Nightly retrain starting for {len(locations)} location(s). Force={args.force}", flush=True)
    trained_locations: list[str] = []
    failed_locations: list[str] = []
    for location in locations:
        try:
            status = retrain_one_location(location, args)
        except Exception as exc:
            status = "failed"
            print(f"{location.id}: failed before run record could be completed - {exc}", flush=True)
        if status == "success":
            trained_locations.append(location.id)
        else:
            failed_locations.append(location.id)

    _print_summary(
        locations_checked=locations_checked,
        trained_locations=trained_locations,
        failed_locations=failed_locations,
    )
    return 1 if failed_locations else 0


if __name__ == "__main__":
    raise SystemExit(main())
