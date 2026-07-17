from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import pandas as pd
import pytest

from scripts import nightly_retrain
from src.location_config import Location
from src.model_publication import (
    atomic_publish_f6_package,
    sha256_file,
    validate_f6_package_file,
    validate_package_in_fresh_process,
)
from src.predictor import VisitorPredictor


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
)
CANDIDATE_SHA256 = "9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397"
PACKAGE_ID = "ny_12550_f6_2026-07-12_v1"
LEGACY_ACTIVE = (
    ROOT / "models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib"
)
LEGACY_SHA256 = "ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0"
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def _copy(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination


def test_fresh_process_validates_locked_candidate_and_smoke_predictions() -> None:
    result = validate_package_in_fresh_process(
        CANDIDATE,
        expected_schema=2,
        expected_sha256=CANDIDATE_SHA256,
        expected_package_id=PACKAGE_ID,
        smokes=[
            ("2026-07-18", "2026-07-17"),
            ("2026-07-19", "2026-07-17"),
            ("2026-08-01", "2026-07-17"),
        ],
    )
    assert result["validation"]["schema_version"] == 2
    assert result["validation"]["feature_count"] == 33
    assert result["validation"]["recommendation_policy_id"] == (
        "C0_EXISTING_RAW_QUANTILE"
    )
    assert [row["suggested_meals"] for row in result["smoke_predictions"]] == [
        116,
        169,
        114,
    ]


def test_schema_v1_can_never_be_published_through_f6_atomic_path(
    tmp_path: Path,
) -> None:
    source = _copy(CANDIDATE, tmp_path / "candidate.joblib")
    active = _copy(LEGACY_ACTIVE, tmp_path / "active.joblib")
    before = sha256_file(active)

    with pytest.raises(ValueError, match="schema version 2"):
        atomic_publish_f6_package(
            source,
            active,
            expected_source_sha256=CANDIDATE_SHA256,
            expected_package_id=PACKAGE_ID,
            previous_active_backup=tmp_path / "previous.joblib",
            expected_current_sha256=LEGACY_SHA256,
        )

    assert sha256_file(active) == before == LEGACY_SHA256
    assert not (tmp_path / "previous.joblib").exists()


def test_failed_f6_source_validation_preserves_current_active(tmp_path: Path) -> None:
    active = _copy(CANDIDATE, tmp_path / "active.joblib")
    source = _copy(CANDIDATE, tmp_path / "candidate.joblib")
    source.write_bytes(source.read_bytes() + b"corrupt")
    before = sha256_file(active)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        atomic_publish_f6_package(
            source,
            active,
            expected_source_sha256=CANDIDATE_SHA256,
            expected_package_id=PACKAGE_ID,
            previous_active_backup=tmp_path / "previous.joblib",
        )

    assert sha256_file(active) == before == CANDIDATE_SHA256
    assert not (tmp_path / "previous.joblib").exists()


def test_failed_post_replace_validation_restores_previous_active(tmp_path: Path) -> None:
    active = _copy(CANDIDATE, tmp_path / "active.joblib")
    source = _copy(CANDIDATE, tmp_path / "candidate.joblib")
    backup = tmp_path / "previous.joblib"
    before = sha256_file(active)
    calls = 0

    def fresh_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("simulated post-replace validation failure")
        return {"validation": {"schema_version": 2}}

    with patch(
        "src.model_publication.validate_package_in_fresh_process",
        side_effect=fresh_validation,
    ):
        with pytest.raises(ValueError, match="post-replace"):
            atomic_publish_f6_package(
                source,
                active,
                expected_source_sha256=CANDIDATE_SHA256,
                expected_package_id=PACKAGE_ID,
                previous_active_backup=backup,
            )

    assert calls >= 4
    assert sha256_file(active) == before == CANDIDATE_SHA256
    assert sha256_file(backup) == CANDIDATE_SHA256


def test_successful_f6_publication_is_atomic_and_retains_previous_active(
    tmp_path: Path,
) -> None:
    active = _copy(CANDIDATE, tmp_path / "active.joblib")
    source = _copy(CANDIDATE, tmp_path / "candidate.joblib")
    backup = tmp_path / "previous.joblib"

    with patch(
        "src.model_publication.validate_package_in_fresh_process",
        return_value={"validation": {"schema_version": 2}},
    ), patch("src.model_publication.os.replace", wraps=__import__("os").replace) as replace:
        receipt = atomic_publish_f6_package(
            source,
            active,
            expected_source_sha256=CANDIDATE_SHA256,
            expected_package_id=PACKAGE_ID,
            previous_active_backup=backup,
        )

    assert replace.call_count >= 2
    assert sha256_file(active) == CANDIDATE_SHA256
    assert sha256_file(backup) == CANDIDATE_SHA256
    assert receipt["atomic_replace"] is True
    assert VisitorPredictor(str(active)).model_package_schema_version == 2


def test_nightly_f6_training_uses_supabase_frame_and_publishes_schema_v2(
    tmp_path: Path,
) -> None:
    active = _copy(CANDIDATE, tmp_path / "active.joblib")
    previous = tmp_path / "previous.joblib"
    package = joblib.load(CANDIDATE)
    package_id = "ny_12550_f6_nightly_2026-07-12_v1"
    package["package_id"] = package_id
    package["package_status"] = "nightly_pending_publication"
    attendance = package["history_df"][["service_date", "visitors"]].copy()
    location = Location(id="ny_12550", name="Newburgh", zip_code="12550")

    with patch.object(
        nightly_retrain,
        "build_f6_model_package",
        return_value=(package, {"overall": {"BacktestRows": 1}}),
    ) as build, patch.object(
        nightly_retrain, "model_file_for_location", return_value=active
    ), patch.object(
        nightly_retrain, "F6_PREVIOUS_ACTIVE_BACKUP", previous
    ), patch.object(nightly_retrain, "_write_f6_metrics"):
        published = nightly_retrain.train_f6_location(location, attendance)

    build.assert_called_once()
    assert build.call_args.kwargs["attendance"] is attendance
    assert published == active
    result = validate_f6_package_file(active, expected_package_id=package_id)
    assert result["schema_version"] == 2
    assert result["feature_order_sha256"] == (
        "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
    )
    assert result["recommendation_policy_id"] == "C0_EXISTING_RAW_QUANTILE"
    assert previous.is_file()


def test_other_locations_keep_legacy_nightly_training(tmp_path: Path) -> None:
    location = Location(id="other", name="Other", zip_code="00000")
    attendance = pd.DataFrame(
        {"service_date": pd.date_range("2025-01-04", periods=20, freq="7D"), "visitors": 100}
    )
    trained_path = tmp_path / "other.joblib"
    args = SimpleNamespace(min_train_size=18, quantile=0.8)

    with patch.object(nightly_retrain, "load_clean_data", return_value=attendance), patch.object(
        nightly_retrain, "train_legacy_location", return_value=trained_path
    ) as legacy, patch.object(nightly_retrain, "train_f6_location") as f6, patch.object(
        nightly_retrain, "_latest_attendance_update", return_value=None
    ), patch.object(nightly_retrain, "_load_metrics", return_value={}), patch.object(
        nightly_retrain, "create_training_run"
    ), patch.object(nightly_retrain, "artifact_dir_for_location", return_value=tmp_path):
        status = nightly_retrain.retrain_one_location(location, args)

    assert status == "success"
    legacy.assert_called_once_with(location_id="other", min_train_size=18, quantile=0.8)
    f6.assert_not_called()


def test_workflow_stages_rolling_backup_and_validates_ny_schema_v2() -> None:
    workflow = (ROOT / ".github/workflows/nightly-retrain.yml").read_text(
        encoding="utf-8"
    )
    assert "models/backups/ny_12550_previous_active.joblib" in workflow
    assert "validate_f6_package_file(model_path)" in workflow
    assert "ny_12550 nightly publication is not schema v2" in workflow


def test_nightly_code_has_no_csv_or_research_artifact_dependency() -> None:
    source = (ROOT / "scripts/nightly_retrain.py").read_text(encoding="utf-8")
    assert "attendance-csv" not in source
    assert "model_optimization" not in source
    assert nightly_retrain.F6_NIGHTLY_LOCATION_IDS == {"ny_12550"}
