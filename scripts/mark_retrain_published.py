from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.location_config import list_locations
from src.model_training_runs import (
    clear_location_dirty_if_unchanged,
    latest_successful_training_run,
    supabase_configured,
    update_training_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark successfully published retrained locations clean.")
    parser.add_argument("location_ids", nargs="*", help="Location IDs whose artifacts were published")
    parser.add_argument("--locations-file", help="Optional newline-delimited file containing location IDs")
    parser.add_argument(
        "--commit-sha",
        help="Published Git commit containing the trained model and artifacts",
    )
    return parser.parse_args()


def _requested_location_ids(args: argparse.Namespace) -> list[str]:
    location_ids = list(args.location_ids)
    if args.locations_file:
        path = Path(args.locations_file)
        if path.exists():
            location_ids.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(location_id for location_id in location_ids if location_id))


def main() -> int:
    args = parse_args()
    location_ids = _requested_location_ids(args)
    if not location_ids:
        print("No retrained locations to mark published.", flush=True)
        return 0

    if not supabase_configured():
        print("Supabase is not configured. Published locations were not marked clean.", flush=True)
        return 2

    configured_ids = {location.id for location in list_locations()}
    unknown_ids = [location_id for location_id in location_ids if location_id not in configured_ids]
    if unknown_ids:
        print(f"Unknown location IDs: {', '.join(unknown_ids)}", flush=True)
        return 2

    published_at = datetime.now(timezone.utc).isoformat()
    for location_id in location_ids:
        training_run = latest_successful_training_run(location_id)
        if training_run is None:
            print(f"{location_id}: no successful training run found; dirty state retained", flush=True)
            return 1
        if args.commit_sha and training_run.get("id") is not None:
            update_training_run(
                int(training_run["id"]),
                commit_sha=args.commit_sha,
            )
        cleared = clear_location_dirty_if_unchanged(
            location_id,
            training_run.get("latest_attendance_updated_at"),
            published_at,
        )
        if cleared:
            print(f"{location_id}: published - dirty state cleared", flush=True)
        else:
            print(f"{location_id}: published, but newer attendance exists - dirty state retained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
