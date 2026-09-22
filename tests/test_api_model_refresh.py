"""A long-running API process must pick up an atomically published model."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from src import api


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    path = tmp_path / "active.joblib"
    path.write_bytes(b"old package")
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
