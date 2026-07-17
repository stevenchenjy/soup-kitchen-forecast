from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.activate_f6_candidate import activate, dry_run, rollback
from src.model_publication import sha256_file, validate_f6_package_file, validate_legacy_package_file


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
)
LEGACY_ACTIVE = (
    ROOT / "models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib"
)
CANDIDATE_SHA256 = "9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397"
LEGACY_SHA256 = "ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0"
PACKAGE_ID = "ny_12550_f6_2026-07-12_v1"
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def _copy(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination


def _fresh_result(*args, **kwargs) -> dict:
    return {"validation": {"schema_version": kwargs["expected_schema"]}, "smoke_predictions": []}


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    candidate = _copy(CANDIDATE, tmp_path / "candidate.joblib")
    active = _copy(LEGACY_ACTIVE, tmp_path / "active.joblib")
    backup = tmp_path / "backup.joblib"
    metadata = tmp_path / "backup.metadata.json"
    receipt = tmp_path / "receipt.json"
    return candidate, active, backup, metadata, receipt


def _activation_kwargs(tmp_path: Path) -> dict:
    candidate, active, backup, metadata, receipt = _paths(tmp_path)
    return {
        "candidate": candidate,
        "active": active,
        "backup": backup,
        "backup_metadata": metadata,
        "expected_candidate_sha256": CANDIDATE_SHA256,
        "expected_legacy_sha256": LEGACY_SHA256,
        "package_id": PACKAGE_ID,
        "receipt": receipt,
    }


def test_activation_dry_run_is_non_mutating_and_machine_readable(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    active_before = sha256_file(kwargs["active"])
    with patch(
        "scripts.activate_f6_candidate.validate_package_in_fresh_process",
        side_effect=_fresh_result,
    ):
        result = dry_run(**{key: value for key, value in kwargs.items() if key != "backup_metadata"})

    assert result["status"] == "passed"
    assert result["model_files_changed"] is False
    assert sha256_file(kwargs["active"]) == active_before == LEGACY_SHA256
    assert not kwargs["backup"].exists()
    assert json.loads(kwargs["receipt"].read_text())["mode"] == "dry-run"


def test_candidate_hash_mismatch_is_refused(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    arguments = {key: value for key, value in kwargs.items() if key != "backup_metadata"}
    arguments["expected_candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dry_run(**arguments)


def test_current_active_hash_mismatch_is_refused(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    arguments = {key: value for key, value in kwargs.items() if key != "backup_metadata"}
    arguments["expected_legacy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Legacy package SHA-256 mismatch"):
        dry_run(**arguments)


def test_wrong_package_id_is_refused(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    arguments = {key: value for key, value in kwargs.items() if key != "backup_metadata"}
    arguments["package_id"] = "wrong-package-v1"
    with pytest.raises(ValueError, match="package ID mismatch"):
        dry_run(**arguments)


def test_existing_backup_is_refused(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    kwargs["backup"].write_bytes(b"must-not-overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        dry_run(**{key: value for key, value in kwargs.items() if key != "backup_metadata"})
    assert kwargs["backup"].read_bytes() == b"must-not-overwrite"


def test_activation_atomically_replaces_active_and_preserves_legacy_backup(
    tmp_path: Path,
) -> None:
    kwargs = _activation_kwargs(tmp_path)
    with patch(
        "scripts.activate_f6_candidate.validate_package_in_fresh_process",
        side_effect=_fresh_result,
    ), patch(
        "scripts.activate_f6_candidate.os.replace",
        wraps=__import__("os").replace,
    ) as replace:
        result = activate(**kwargs)

    assert replace.call_count >= 3
    assert result["atomic_replace"] is True
    assert sha256_file(kwargs["active"]) == CANDIDATE_SHA256
    assert sha256_file(kwargs["backup"]) == LEGACY_SHA256
    assert validate_f6_package_file(kwargs["active"])["schema_version"] == 2
    assert validate_legacy_package_file(kwargs["backup"])["feature_count"] == 26
    metadata = json.loads(kwargs["backup_metadata"].read_text())
    assert metadata["backup_sha256"] == LEGACY_SHA256
    assert metadata["rollback_target"] == str(kwargs["active"].resolve())
    assert json.loads(kwargs["receipt"].read_text())["status"] == "passed"


def test_activation_post_validation_failure_restores_legacy_active(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    calls = 0

    def fail_active_validation(*args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise ValueError("simulated active validation failure")
        return _fresh_result(*args, **call_kwargs)

    with patch(
        "scripts.activate_f6_candidate.validate_package_in_fresh_process",
        side_effect=fail_active_validation,
    ):
        with pytest.raises(ValueError, match="active validation"):
            activate(**kwargs)

    assert calls >= 6
    assert sha256_file(kwargs["active"]) == LEGACY_SHA256
    assert sha256_file(kwargs["backup"]) == LEGACY_SHA256


def test_rollback_rehearsal_restores_legacy_then_reapplies_f6(tmp_path: Path) -> None:
    kwargs = _activation_kwargs(tmp_path)
    with patch(
        "scripts.activate_f6_candidate.validate_package_in_fresh_process",
        side_effect=_fresh_result,
    ):
        activate(**kwargs)
        rollback_receipt = tmp_path / "rollback.json"
        rollback(
            candidate=kwargs["candidate"],
            active=kwargs["active"],
            backup=kwargs["backup"],
            expected_candidate_sha256=CANDIDATE_SHA256,
            expected_legacy_sha256=LEGACY_SHA256,
            package_id=PACKAGE_ID,
            receipt=rollback_receipt,
        )
        assert sha256_file(kwargs["active"]) == LEGACY_SHA256
        assert validate_legacy_package_file(kwargs["active"])["feature_count"] == 26

        second_backup = tmp_path / "second-backup.joblib"
        second_metadata = tmp_path / "second-backup.metadata.json"
        reapply_receipt = tmp_path / "reapply.json"
        activate(
            candidate=kwargs["candidate"],
            active=kwargs["active"],
            backup=second_backup,
            backup_metadata=second_metadata,
            expected_candidate_sha256=CANDIDATE_SHA256,
            expected_legacy_sha256=LEGACY_SHA256,
            package_id=PACKAGE_ID,
            receipt=reapply_receipt,
        )

    assert sha256_file(kwargs["active"]) == CANDIDATE_SHA256
    assert validate_f6_package_file(kwargs["active"])["package_id"] == PACKAGE_ID
    assert json.loads(rollback_receipt.read_text())["mode"] == "rollback"
