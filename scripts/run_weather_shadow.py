"""Capture prospective F6/weather pairs and score them when attendance arrives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.weather_shadow import capture_shadow_forecast, evaluate_shadow_runs
from src.location_config import get_location
from src.weather_shadow_schedule import capture_due_services, next_preparation_cutoffs, preparation_cutoff
from src.weather_shadow_store import LocalShadowStore, read_shadow_runs

STUDY_CONFIG = ROOT / "config/weather_shadow_studies.json"


def load_study(location_id: str, path: Path = STUDY_CONFIG) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported weather study configuration schema.")
    matches = [study for study in config.get("studies", []) if study.get("location_id") == location_id]
    if len(matches) != 1:
        raise ValueError(f"Exactly one preparation schedule must be configured for {location_id}.")
    study = matches[0]
    if study.get("duration_basis") != "elapsed_utc" or study.get("timezone") != get_location(location_id).timezone:
        raise ValueError("Study must use elapsed hours and the configured location timezone.")
    return study


def schedule_options(study: dict) -> dict:
    return {key: study[key] for key in (
        "timezone", "anchor_time", "lead_hours", "capture_window_minutes", "start_minutes_before_cutoff"
    ) if key in study} | {"anchor_time": study["anchor_local_time"]}


def _local_store(args, study: dict):
    configured = args.local_store or study.get("local_store")
    if configured is None:
        return None
    path = Path(configured)
    return LocalShadowStore(path if path.is_absolute() else ROOT / path)


def _already_saved(runs: list[dict], study: dict, due: dict) -> bool:
    cutoff = datetime.fromisoformat(due["cutoff_at"])
    for record in runs:
        if (record["service_date"] != due["service_date"] or record["status"] != "paired"
                or record["payload"].get("study_id") != study["study_id"]):
            continue
        try:
            if (datetime.fromisoformat(record["cutoff_at"]) == cutoff
                    and datetime.fromisoformat(record["persisted_at"]) <= cutoff
                    and datetime.fromisoformat(record["weather_retrieved_at"]) <= cutoff
                    and record["payload"].get("eligible_for_evaluation")):
                return True
        except (KeyError, ValueError, TypeError):
            continue
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Run within 15 minutes before the actual preparation cutoff.")
    capture.add_argument("--location", required=True)
    capture.add_argument("--service-date", required=True)
    capture.add_argument("--cutoff-at", help="Optional explicit cutoff; must equal the confirmed preparation rule.")
    capture.add_argument("--candidate", type=Path, help="Override the study's versioned weather candidate.")
    capture.add_argument("--study-id")
    capture.add_argument("--capture-window-minutes", type=int)
    capture.add_argument("--local-store", type=Path, help="Override the local SQLite path configured for this study.")
    capture.add_argument("--config", type=Path, default=STUDY_CONFIG)
    due_parser = commands.add_parser("capture-due", help="Capture only in the configured pre-cutoff window; otherwise do nothing.")
    due_parser.add_argument("--location", required=True)
    due_parser.add_argument("--config", type=Path, default=STUDY_CONFIG)
    due_parser.add_argument("--candidate", type=Path)
    due_parser.add_argument("--local-store", type=Path)
    plan = commands.add_parser("schedule", help="Show upcoming computed cutoffs without making predictions or requesting weather.")
    plan.add_argument("--location", required=True)
    plan.add_argument("--count", type=int, default=4)
    plan.add_argument("--config", type=Path, default=STUDY_CONFIG)
    evaluate = commands.add_parser("evaluate", help="Join saved pairs to actual attendance; never backfill forecasts.")
    evaluate.add_argument("--location", required=True)
    evaluate.add_argument("--attendance-csv", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path, help="New directory for this evaluation, never overwritten.")
    evaluate.add_argument("--local-store", type=Path)
    evaluate.add_argument("--config", type=Path, default=STUDY_CONFIG)
    args = parser.parse_args()
    study = load_study(args.location, args.config)
    now = datetime.now(timezone.utc)
    if args.command == "schedule":
        print(json.dumps({"location_id": args.location, "rule": f"{study['lead_hours']} elapsed hours before {study['anchor_local_time']} local service time", "upcoming": next_preparation_cutoffs(now, count=args.count, **schedule_options(study))}, indent=2))
        return 0
    if args.command == "capture-due":
        due = capture_due_services(now, **schedule_options(study))
        if not due:
            print(json.dumps({"status": "not_due", "next": next_preparation_cutoffs(now, count=1, **schedule_options(study))}, indent=2))
            return 0
        store = _local_store(args, study)
        runs = store.load_runs(args.location) if store else read_shadow_runs(args.location)
        failed = False
        outputs = []
        for item in due:
            if _already_saved(runs, study, item):
                outputs.append({"service_date": item["service_date"], "status": "already_captured"})
                continue
            record = capture_shadow_forecast(
                args.location, item["service_date"], item["cutoff_at"], args.candidate or ROOT / study["candidate"],
                study_id=study["study_id"], capture_window_minutes=study["capture_window_minutes"], store=store,
            )
            outputs.append({"run_id": record["run_id"], "service_date": item["service_date"], "status": record["status"], "failure": record["payload"].get("failure")})
            failed |= record["status"] != "paired"
        print(json.dumps({"captures": outputs}, indent=2))
        return 2 if failed else 0
    if args.command == "capture":
        cutoff = preparation_cutoff(args.service_date, timezone=study["timezone"], anchor_time=study["anchor_local_time"], lead_hours=study["lead_hours"])
        if args.cutoff_at and datetime.fromisoformat(args.cutoff_at.replace("Z", "+00:00")) != cutoff:
            raise ValueError("Explicit cutoff does not match the confirmed 12-hour preparation rule.")
        record = capture_shadow_forecast(
            args.location, args.service_date, cutoff, args.candidate or ROOT / study["candidate"],
            study_id=args.study_id or study["study_id"], capture_window_minutes=(args.capture_window_minutes if args.capture_window_minutes is not None else study["capture_window_minutes"]),
            store=_local_store(args, study),
        )
        print(json.dumps({
            "run_id": record["run_id"], "status": record["status"],
            "service_date": record["service_date"], "cutoff_at": record["cutoff_at"],
            "baseline": record["payload"]["baseline"], "candidate": record["payload"]["candidate"],
            "failure": record["payload"].get("failure"),
            "note": "Comparison only. Existing staff recommendation is unchanged.",
        }, indent=2))
        return 0 if record["status"] == "paired" else 2
    if args.output_dir.exists():
        raise ValueError("Evaluation output directory already exists; choose a new run directory.")
    store = _local_store(args, study)
    if store:
        runs = store.load_runs(args.location)
    else:
        runs = read_shadow_runs(args.location)
    attendance = pd.read_csv(args.attendance_csv)
    if "location_id" in attendance:
        attendance = attendance[attendance.location_id.eq(args.location)].copy()
    result = evaluate_shadow_runs(runs, attendance)
    result["attendance_csv_sha256"] = hashlib.sha256(args.attendance_csv.read_bytes()).hexdigest()
    result["location_id"] = args.location
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(result["metrics"]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(result["strategy_metrics"]).to_csv(args.output_dir / "strategy_metrics.csv", index=False)
    pd.DataFrame(result["predictions"]).to_csv(args.output_dir / "paired_predictions.csv", index=False)
    print(json.dumps({
        "paired_rows": result["paired_rows"], "input_runs": result["input_runs"],
        "exclusions": result["exclusions"], "output_dir": str(args.output_dir),
        "note": "No accuracy estimate until completed services have attendance and eligible pre-cutoff pairs.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Weather comparison: {exc}", file=sys.stderr)
        raise SystemExit(1)
