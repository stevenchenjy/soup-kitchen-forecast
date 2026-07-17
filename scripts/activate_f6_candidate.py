"""Explicit, hash-guarded activation and rollback for the ny_12550 F6 package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_publication import (
    copy_file_exclusive,
    sha256_file,
    validate_f6_package_file,
    validate_legacy_package_file,
    validate_package_in_fresh_process,
)


REFERENCE_SMOKES = [
    ("2026-07-18", "2026-07-17"),
    ("2026-07-19", "2026-07-17"),
    ("2026-08-01", "2026-07-17"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _require_distinct_paths(candidate: Path, active: Path, backup: Path) -> None:
    paths = {candidate.resolve(), active.resolve(), backup.resolve()}
    if len(paths) != 3:
        raise ValueError("Candidate, active, and backup paths must be distinct")


def _validated_temporary_copy(
    source: Path,
    active: Path,
    *,
    expected_sha256: str,
    expected_schema: int,
    expected_package_id: str | None = None,
    smokes: list[tuple[str, str]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    active.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{active.name}.activation-", suffix=".joblib", dir=active.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise ValueError("Temporary activation file hash differs from source")
        if expected_schema == 2:
            validate_f6_package_file(
                temporary,
                expected_sha256=expected_sha256,
                expected_package_id=expected_package_id,
            )
        else:
            validate_legacy_package_file(
                temporary, expected_sha256=expected_sha256
            )
        fresh = validate_package_in_fresh_process(
            temporary,
            expected_schema=expected_schema,
            expected_sha256=expected_sha256,
            expected_package_id=expected_package_id,
            smokes=smokes,
        )
        return temporary, fresh
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _pre_activation_validation(
    *,
    candidate: Path,
    active: Path,
    backup: Path,
    expected_candidate_sha256: str,
    expected_legacy_sha256: str,
    package_id: str,
) -> dict[str, Any]:
    _require_distinct_paths(candidate, active, backup)
    if backup.exists():
        raise FileExistsError(f"Backup destination already exists: {backup}")
    candidate_validation = validate_f6_package_file(
        candidate,
        expected_sha256=expected_candidate_sha256,
        expected_package_id=package_id,
    )
    active_validation = validate_legacy_package_file(
        active, expected_sha256=expected_legacy_sha256
    )
    candidate_fresh = validate_package_in_fresh_process(
        candidate,
        expected_schema=2,
        expected_sha256=expected_candidate_sha256,
        expected_package_id=package_id,
        smokes=REFERENCE_SMOKES,
    )
    legacy_fresh = validate_package_in_fresh_process(
        active,
        expected_schema=1,
        expected_sha256=expected_legacy_sha256,
        smokes=[REFERENCE_SMOKES[0]],
    )
    return {
        "candidate": candidate_validation,
        "active_before": active_validation,
        "candidate_fresh_process": candidate_fresh,
        "legacy_fresh_process": legacy_fresh,
    }


def dry_run(
    *,
    candidate: Path,
    active: Path,
    backup: Path,
    expected_candidate_sha256: str,
    expected_legacy_sha256: str,
    package_id: str,
    receipt: Path,
) -> dict[str, Any]:
    validation = _pre_activation_validation(
        candidate=candidate,
        active=active,
        backup=backup,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_legacy_sha256=expected_legacy_sha256,
        package_id=package_id,
    )
    payload = {
        "mode": "dry-run",
        "status": "passed",
        "timestamp_utc": _now(),
        "git_commit": _git_commit(),
        "model_files_changed": False,
        "backup_created": False,
        **validation,
    }
    _write_json_atomic(receipt, payload)
    return payload


def activate(
    *,
    candidate: Path,
    active: Path,
    backup: Path,
    backup_metadata: Path,
    expected_candidate_sha256: str,
    expected_legacy_sha256: str,
    package_id: str,
    receipt: Path,
) -> dict[str, Any]:
    validation = _pre_activation_validation(
        candidate=candidate,
        active=active,
        backup=backup,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_legacy_sha256=expected_legacy_sha256,
        package_id=package_id,
    )
    backup_sha256 = copy_file_exclusive(active, backup)
    if backup_sha256 != expected_legacy_sha256:
        raise ValueError("Backup hash differs from the verified pre-activation hash")
    backup_validation = validate_legacy_package_file(
        backup, expected_sha256=expected_legacy_sha256
    )
    backup_fresh = validate_package_in_fresh_process(
        backup,
        expected_schema=1,
        expected_sha256=expected_legacy_sha256,
        smokes=[REFERENCE_SMOKES[0]],
    )
    backup_payload = {
        "source_path": str(active.resolve()),
        "source_sha256": expected_legacy_sha256,
        "backup_path": str(backup.resolve()),
        "backup_sha256": backup_sha256,
        "schema_version": backup_validation["schema_version"],
        "feature_count": backup_validation["feature_count"],
        "creation_timestamp_utc": _now(),
        "git_commit": _git_commit(),
        "rollback_target": str(active.resolve()),
        "fresh_process_validation": backup_fresh,
    }
    _write_json_atomic(backup_metadata, backup_payload)

    temporary: Path | None = None
    replaced = False
    try:
        temporary, temporary_fresh = _validated_temporary_copy(
            candidate,
            active,
            expected_sha256=expected_candidate_sha256,
            expected_schema=2,
            expected_package_id=package_id,
            smokes=REFERENCE_SMOKES,
        )
        os.replace(temporary, active)
        temporary = None
        replaced = True
        active_validation = validate_f6_package_file(
            active,
            expected_sha256=expected_candidate_sha256,
            expected_package_id=package_id,
        )
        active_fresh = validate_package_in_fresh_process(
            active,
            expected_schema=2,
            expected_sha256=expected_candidate_sha256,
            expected_package_id=package_id,
            smokes=REFERENCE_SMOKES,
        )
    except Exception:
        if replaced:
            rollback_temporary, _ = _validated_temporary_copy(
                backup,
                active,
                expected_sha256=expected_legacy_sha256,
                expected_schema=1,
                smokes=[REFERENCE_SMOKES[0]],
            )
            os.replace(rollback_temporary, active)
            validate_legacy_package_file(
                active, expected_sha256=expected_legacy_sha256
            )
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    payload = {
        "mode": "activate",
        "status": "passed",
        "timestamp_utc": _now(),
        "git_commit": _git_commit(),
        "atomic_replace": True,
        "backup": backup_payload,
        "temporary_file_fresh_process": temporary_fresh,
        "active_after": active_validation,
        "active_fresh_process": active_fresh,
        **validation,
    }
    _write_json_atomic(receipt, payload)
    return payload


def rollback(
    *,
    candidate: Path,
    active: Path,
    backup: Path,
    expected_candidate_sha256: str,
    expected_legacy_sha256: str,
    package_id: str,
    receipt: Path,
) -> dict[str, Any]:
    _require_distinct_paths(candidate, active, backup)
    validate_f6_package_file(
        active,
        expected_sha256=expected_candidate_sha256,
        expected_package_id=package_id,
    )
    backup_validation = validate_legacy_package_file(
        backup, expected_sha256=expected_legacy_sha256
    )
    temporary, temporary_fresh = _validated_temporary_copy(
        backup,
        active,
        expected_sha256=expected_legacy_sha256,
        expected_schema=1,
        smokes=[REFERENCE_SMOKES[0]],
    )
    os.replace(temporary, active)
    active_validation = validate_legacy_package_file(
        active, expected_sha256=expected_legacy_sha256
    )
    active_fresh = validate_package_in_fresh_process(
        active,
        expected_schema=1,
        expected_sha256=expected_legacy_sha256,
        smokes=[REFERENCE_SMOKES[0]],
    )
    payload = {
        "mode": "rollback",
        "status": "passed",
        "timestamp_utc": _now(),
        "git_commit": _git_commit(),
        "atomic_replace": True,
        "backup": backup_validation,
        "temporary_file_fresh_process": temporary_fresh,
        "active_after": active_validation,
        "active_fresh_process": active_fresh,
    }
    _write_json_atomic(receipt, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate or roll back an F6 model package.")
    parser.add_argument("--mode", choices=("dry-run", "activate", "rollback"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--active", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--backup-metadata")
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-legacy-sha256", required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--receipt", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = Path(args.candidate)
    active = Path(args.active)
    backup = Path(args.backup)
    receipt = Path(args.receipt)
    common = {
        "candidate": candidate,
        "active": active,
        "backup": backup,
        "expected_candidate_sha256": args.expected_candidate_sha256,
        "expected_legacy_sha256": args.expected_legacy_sha256,
        "package_id": args.package_id,
        "receipt": receipt,
    }
    if args.mode == "dry-run":
        payload = dry_run(**common)
    elif args.mode == "activate":
        metadata_path = Path(
            args.backup_metadata or f"{backup}.metadata.json"
        )
        if metadata_path.exists():
            raise FileExistsError(f"Backup metadata already exists: {metadata_path}")
        payload = activate(backup_metadata=metadata_path, **common)
    else:
        payload = rollback(**common)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
