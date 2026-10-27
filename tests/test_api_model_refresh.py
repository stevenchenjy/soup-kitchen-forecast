"""A long-running API process must pick up an atomically published model."""
from pathlib import Path
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from src import api, meal_policy


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    path = tmp_path / "active.joblib"
    path.write_bytes(b"old package")
    policy_path = tmp_path / "meal_policies.json"
    policy_path.write_text(json.dumps({
        "schema_version": 1,
        "policies": {"ny_12550": {
            "policy_id": "P1_EQUAL_WEIGHT_Q80_MINUS_7",
            "base_policy_id": "C0_EXISTING_RAW_QUANTILE",
            "reduction_meals": 7,
            "effective_from_service_date": "2026-10-31",
        }},
    }))
    monkeypatch.setattr(meal_policy, "MEAL_POLICY_CONFIG_PATH", policy_path)
    monkeypatch.setattr(api, "_predictor_cache", {})
    monkeypatch.setattr(api, "model_file_for_location", lambda _: path)
    monkeypatch.setattr(api, "VisitorPredictor", Mock(side_effect=lambda name: Path(name).read_bytes()))
    return path


def test_unchanged_package_reuses_cache(model_file):
    assert api._get_predictor("ny_12550") == b"old package"
    assert api._get_predictor("ny_12550") == b"old package"
    api.VisitorPredictor.assert_called_once()


def test_atomic_publication_refreshes_loaded_model(model_file):
    assert api._get_predictor("ny_12550") == b"old package"
    replacement = model_file.with_name("candidate.joblib")
    replacement.write_bytes(b"new package")
    replacement.replace(model_file)
    assert api._get_predictor("ny_12550") == b"new package"
    assert api.VisitorPredictor.call_count == 2


def test_deleted_package_does_not_serve_cached_predictions(model_file):
    api._get_predictor("ny_12550")
    model_file.unlink()
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 404
    assert "ny_12550" not in api._predictor_cache


def change_policy(model_file):
    path = model_file.with_name("meal_policies.json")
    before = path.stat()
    path.write_text(path.read_text().replace("MINUS_7", "MINUS_6").replace(
        '"reduction_meals": 7', '"reduction_meals": 6'
    ))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_policy_content_change_refreshes_cache_even_with_same_size_and_mtime(model_file):
    api._get_predictor("ny_12550")
    change_policy(model_file)
    api._get_predictor("ny_12550")
    assert api.VisitorPredictor.call_count == 2


def test_policy_change_during_load_does_not_cache_a_mixed_identity(model_file):
    def change_during_load(name):
        change_policy(model_file)
        return Path(name).read_bytes()

    api.VisitorPredictor.side_effect = change_during_load
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 503
    assert "ny_12550" not in api._predictor_cache


def test_loaded_policy_snapshot_must_match_the_observed_config(model_file):
    api.VisitorPredictor.side_effect = None
    api.VisitorPredictor.return_value = SimpleNamespace(
        meal_policy_config_fingerprint="a different loaded snapshot"
    )
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 503
    assert "ny_12550" not in api._predictor_cache


def test_missing_policy_config_does_not_serve_a_cached_plan(model_file):
    api._get_predictor("ny_12550")
    model_file.with_name("meal_policies.json").unlink()
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 503
    assert "ny_12550" not in api._predictor_cache


def test_invalid_loaded_policy_clears_a_previously_cached_predictor(model_file):
    api._get_predictor("ny_12550")
    change_policy(model_file)
    api.VisitorPredictor.side_effect = ValueError("invalid preparation policy")
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 503
    assert "ny_12550" not in api._predictor_cache


def test_replacement_during_load_does_not_cache_wrong_identity(model_file):
    def replace_during_load(name):
        loaded = Path(name).read_bytes()
        replacement = model_file.with_name("candidate.joblib")
        replacement.write_bytes(b"new package")
        replacement.replace(model_file)
        return loaded

    api.VisitorPredictor.side_effect = replace_during_load
    with pytest.raises(HTTPException) as raised:
        api._get_predictor("ny_12550")
    assert raised.value.status_code == 503
    assert "ny_12550" not in api._predictor_cache
