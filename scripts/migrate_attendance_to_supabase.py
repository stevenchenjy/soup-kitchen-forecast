from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import location_db_file
from src.data_admin import _supabase_config, _supabase_request, attendance_config_source
from src.location_config import list_locations


BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local SQLite attendance rows to Supabase.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--location", help="Single location id to migrate, e.g. ny_12550")
    scope.add_argument("--all", action="store_true", help="Migrate every location in data/locations.json")
    return parser.parse_args()


def read_local_attendance(location_id: str) -> list[dict]:
    db = location_db_file(location_id)
    if not db.exists():
        return []

    uri = f"file:{db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        try:
            rows = conn.execute(
                "SELECT service_date, visitors FROM attendance ORDER BY service_date"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            raise

    return [
        {
            "service_date": str(service_date)[:10],
            "visitors": int(visitors),
        }
        for service_date, visitors in rows
    ]


def chunked(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def upsert_attendance(location_id: str, rows: list[dict]) -> int:
    if not rows:
        return 0

    uploaded = 0
    now = datetime.now(timezone.utc).isoformat()
    for batch in chunked(rows, BATCH_SIZE):
        payload = [
            {
                "location_id": location_id,
                "service_date": row["service_date"],
                "visitors": row["visitors"],
                "updated_at": now,
            }
            for row in batch
        ]
        _supabase_request(
            "POST",
            params={"on_conflict": "location_id,service_date"},
            payload=payload,
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )
        uploaded += len(batch)
    return uploaded


def migrate_location(location_id: str) -> tuple[int, int]:
    rows = read_local_attendance(location_id)
    uploaded = upsert_attendance(location_id, rows)
    print(f"{location_id}: found {len(rows)} local rows, uploaded/upserted {uploaded} rows")
    return len(rows), uploaded


def main() -> int:
    args = parse_args()
    config = _supabase_config()
    source = attendance_config_source()
    print(f"Supabase configured: {'yes' if config else 'no'}", flush=True)
    print(f"Configuration source: {source}", flush=True)
    if config is None:
        print("Supabase is not configured. Add Supabase secrets before running this migration.", file=sys.stderr)
        return 1

    location_ids = [args.location] if args.location else [location.id for location in list_locations()]
    total_found = 0
    total_uploaded = 0
    for location_id in location_ids:
        found, uploaded = migrate_location(location_id)
        total_found += found
        total_uploaded += uploaded

    print(f"Total: found {total_found} local rows, uploaded/upserted {total_uploaded} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
