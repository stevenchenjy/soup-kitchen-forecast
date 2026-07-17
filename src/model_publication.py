"""Validated, atomic publication primitives for production model packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd

from src.predictor import VisitorPredictor
from src.production_features import (
    LOCKED_F6_FEATURE_ORDER_SHA256,
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
    feature_order_sha256,
    locked_feature_contract_metadata,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_f6_package_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_package_id: str | None = None,
) -> dict[str, Any]:
    package_path = Path(path).resolve()
    if not package_path.is_file():
        raise FileNotFoundError(f"Model package does not exist: {package_path}")
    actual_sha256 = sha256_file(package_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Model package SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    predictor = VisitorPredictor(str(package_path))
    if predictor.model_package_schema_version != MODEL_PACKAGE_SCHEMA_VERSION:
        raise ValueError("F6 publication requires model-package schema version 2")
    if expected_package_id is not None and predictor.package_id != expected_package_id:
        raise ValueError(
            f"Model package ID mismatch: {predictor.package_id} != {expected_package_id}"
        )
    if len(predictor.feature_cols) != 33:
        raise ValueError("F6 publication requires exactly 33 features")
    feature_hash = feature_order_sha256(predictor.feature_cols)
    if feature_hash != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise ValueError("F6 publication feature-order hash does not match the lock")
    if predictor.feature_contract != locked_feature_contract_metadata():
        raise ValueError("F6 publication feature contract differs from the tracked lock")
    if predictor.recommendation_policy_id != RECOMMENDATION_POLICY_ID:
        raise ValueError("F6 publication recommendation policy is not locked C0")
    if set(predictor.models) != {"sat", "sun"}:
        raise ValueError("F6 publication requires separate Saturday and Sunday models")

    history_dates = pd.to_datetime(predictor.history_df["service_date"], errors="raise")
    return {
        "path": str(package_path),
        "sha256": actual_sha256,
        "package_id": predictor.package_id,
        "schema_version": predictor.model_package_schema_version,
        "feature_count": len(predictor.feature_cols),
        "feature_order_sha256": feature_hash,
        "feature_set_id": predictor.feature_contract["feature_set_id"],
        "recommendation_policy_id": predictor.recommendation_policy_id,
        "model_segments": sorted(predictor.models),
        "history_row_count": len(predictor.history_df),
        "history_maximum_date": history_dates.max().date().isoformat(),
    }


def validate_legacy_package_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_feature_count: int = 26,
) -> dict[str, Any]:
    package_path = Path(path).resolve()
    if not package_path.is_file():
        raise FileNotFoundError(f"Legacy model package does not exist: {package_path}")
    actual_sha256 = sha256_file(package_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Legacy package SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    predictor = VisitorPredictor(str(package_path))
    if predictor.model_package_schema_version != 1:
        raise ValueError("Rollback package must use model-package schema version 1")
    if len(predictor.feature_cols) != expected_feature_count:
        raise ValueError(
            f"Rollback package feature count is {len(predictor.feature_cols)}, "
            f"expected {expected_feature_count}"
        )
    return {
        "path": str(package_path),
        "sha256": actual_sha256,
        "package_id": predictor.package_id,
        "schema_version": predictor.model_package_schema_version,
        "feature_count": len(predictor.feature_cols),
        "recommendation_policy_id": predictor.recommendation_policy_id,
    }


def validate_package_in_fresh_process(
    path: str | Path,
    *,
    expected_schema: int,
    expected_sha256: str,
    expected_package_id: str | None = None,
    smokes: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "scripts.validate_model_package",
        "--package",
        str(Path(path).resolve()),
        "--expected-schema",
        str(expected_schema),
        "--expected-sha256",
        expected_sha256,
    ]
    if expected_package_id is not None:
        command.extend(["--package-id", expected_package_id])
    for target, origin in smokes or []:
        command.extend(["--smoke", f"{target}:{origin}"])
    environment = os.environ.copy()
    environment["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "Fresh-process model validation failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError("Fresh-process validation returned invalid JSON") from exc


def copy_file_exclusive(source: str | Path, destination: str | Path) -> str:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("Backup source and destination must be different files")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb") as source_handle, destination_path.open("xb") as output:
            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    return sha256_file(destination_path)


def _validated_sibling_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_package_id: str | None,
    expected_schema: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise ValueError("Temporary model copy SHA-256 differs from its source")
        if expected_schema == MODEL_PACKAGE_SCHEMA_VERSION:
            validate_f6_package_file(
                temporary,
                expected_sha256=expected_sha256,
                expected_package_id=expected_package_id,
            )
        else:
            validate_legacy_package_file(temporary, expected_sha256=expected_sha256)
        validate_package_in_fresh_process(
            temporary,
            expected_schema=expected_schema,
            expected_sha256=expected_sha256,
            expected_package_id=expected_package_id,
        )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_publish_f6_package(
    source: str | Path,
    active: str | Path,
    *,
    expected_source_sha256: str,
    expected_package_id: str,
    previous_active_backup: str | Path,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    active_path = Path(active).resolve()
    backup_path = Path(previous_active_backup).resolve()
    if source_path == active_path:
        raise ValueError("Publication source cannot equal the active model path")
    source_validation = validate_f6_package_file(
        source_path,
        expected_sha256=expected_source_sha256,
        expected_package_id=expected_package_id,
    )
    current_validation = validate_f6_package_file(
        active_path,
        expected_sha256=expected_current_sha256,
    )
    temporary = _validated_sibling_copy(
        source_path,
        active_path,
        expected_sha256=expected_source_sha256,
        expected_package_id=expected_package_id,
        expected_schema=MODEL_PACKAGE_SCHEMA_VERSION,
    )
    backup_temporary: Path | None = None
    replaced = False
    try:
        backup_temporary = _validated_sibling_copy(
            active_path,
            backup_path,
            expected_sha256=current_validation["sha256"],
            expected_package_id=current_validation["package_id"],
            expected_schema=MODEL_PACKAGE_SCHEMA_VERSION,
        )
        os.replace(backup_temporary, backup_path)
        backup_temporary = None
        os.replace(temporary, active_path)
        replaced = True
        active_validation = validate_f6_package_file(
            active_path,
            expected_sha256=expected_source_sha256,
            expected_package_id=expected_package_id,
        )
        validate_package_in_fresh_process(
            active_path,
            expected_schema=MODEL_PACKAGE_SCHEMA_VERSION,
            expected_sha256=expected_source_sha256,
            expected_package_id=expected_package_id,
        )
    except Exception:
        if replaced and backup_path.is_file():
            rollback_temp = _validated_sibling_copy(
                backup_path,
                active_path,
                expected_sha256=current_validation["sha256"],
                expected_package_id=current_validation["package_id"],
                expected_schema=MODEL_PACKAGE_SCHEMA_VERSION,
            )
            os.replace(rollback_temp, active_path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        if backup_temporary is not None:
            backup_temporary.unlink(missing_ok=True)

    return {
        "source": source_validation,
        "previous_active": current_validation,
        "previous_active_backup": {
            "path": str(backup_path),
            "sha256": sha256_file(backup_path),
        },
        "active": active_validation,
        "atomic_replace": True,
    }
