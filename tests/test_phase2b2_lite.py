from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase2b2_lite import (
    ALIGNMENT_KEY,
    EXPECTED_F6_HASH,
    EXPECTED_SNAPSHOT_SHA256,
    EXPECTED_SOURCE_SHA256,
    OUTPUT_DIR,
    PHASE2B1_PREDICTIONS,
    PREDICTION_KEY,
    SNAPSHOT_PATH,
    SOURCE_PATH,
    build_evaluator,
    prior_phase_fingerprints,
    protected_fingerprints,
)
from scripts.run_phase2a5_supabase_reconciliation import history_from_normalized, locked_feature_contract, sha256_file
from src.modeling import make_point_model
from src.sample_weights import SAMPLE_WEIGHT_CANDIDATES, generate_sample_weights


pytestmark = pytest.mark.skipif(
    not SOURCE_PATH.is_file()
    or not SNAPSHOT_PATH.is_file()
    or not (OUTPUT_DIR / "phase2b2_lite_manifest.json").is_file(),
    reason="Phase 2B2 reproduction requires ignored frozen inputs and artifacts",
)


def test_model_class_parameters_and_seed_remain_locked() -> None:
    model = make_point_model()
    params = model.get_params()
    assert type(model).__name__ == "RandomForestRegressor"
    assert params["n_estimators"] == 400
    assert params["max_depth"] == 8
    assert params["min_samples_leaf"] == 2
    assert params["random_state"] == 42


def test_cached_folds_include_all_expanding_rows_and_unweighted_preprocessing() -> None:
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    lock = locked_feature_contract()
    history = history_from_normalized(snapshot)
    evaluator = build_evaluator(history, lock)
    targets = pd.to_datetime(snapshot["service_date"]).tail(4)
    cache = evaluator.prepare_point_fold_cache(target_dates=targets)
    assert cache.training_frame_build_count == 1
    assert cache.prediction_feature_build_count == len(cache.prediction_folds)
    for fold in cache.training_folds.values():
        assert len(fold.training_row_indices) == len(fold.training_dates) == len(fold.y_train)
        assert not fold.x_train.flags.writeable
        assert "unweighted_imputer" in fold.preprocessing_id
        before = hashlib.sha256(fold.x_train.tobytes()).hexdigest()
        for policy in SAMPLE_WEIGHT_CANDIDATES:
            assert len(generate_sample_weights(policy, len(fold.y_train))) == len(fold.y_train)
        assert hashlib.sha256(fold.x_train.tobytes()).hexdigest() == before


def test_locked_snapshot_and_feature_hashes_are_unchanged() -> None:
    assert sha256_file(SNAPSHOT_PATH) == EXPECTED_SNAPSHOT_SHA256
    assert sha256_file(SOURCE_PATH) == EXPECTED_SOURCE_SHA256
    assert locked_feature_contract()["feature_list_sha256"] == EXPECTED_F6_HASH


def test_phase2b2_lite_artifacts_and_uniform_reproduction() -> None:
    manifest = json.loads((OUTPUT_DIR / "phase2b2_lite_manifest.json").read_text())
    predictions = pd.read_csv(
        OUTPUT_DIR / "05_point_predictions.csv",
        parse_dates=["forecast_origin", "target_date"],
        low_memory=False,
    )
    assert not predictions.duplicated(PREDICTION_KEY).any()
    assert predictions.groupby("sample_weight_id").size().nunique() == 1
    uniform = predictions[predictions["sample_weight_id"] == "SW_UNIFORM"].sort_values(ALIGNMENT_KEY).reset_index(drop=True)
    reference = pd.read_csv(
        PHASE2B1_PREDICTIONS,
        parse_dates=["forecast_origin", "target_date"],
        low_memory=False,
    )
    reference = reference[reference["training_window_id"] == "TW_EXPANDING"].sort_values(ALIGNMENT_KEY).reset_index(drop=True)
    assert uniform[ALIGNMENT_KEY].equals(reference[ALIGNMENT_KEY])
    np.testing.assert_allclose(uniform["point_prediction"], reference["point_prediction"], rtol=0, atol=1e-10)
    assert manifest["training_frame_build_count"] == 1
    assert manifest["prediction_feature_build_count"] == manifest["prediction_key_count_per_candidate"]
    assert manifest["point_model_fit_count"] == 3 * manifest["unique_training_context_count"]
    assert manifest["cached_fold_reuse"]["preprocessing_weighted"] is False


def test_prediction_keys_and_cached_features_align_across_candidates() -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "05_point_predictions.csv", low_memory=False)
    assert predictions.groupby(ALIGNMENT_KEY)["fold_cache_id"].nunique().max() == 1
    assert predictions.groupby(ALIGNMENT_KEY)["x_train_sha256"].nunique().max() == 1
    assert predictions.groupby(ALIGNMENT_KEY)["x_test_sha256"].nunique().max() == 1
    assert not predictions["preprocessing_weighted"].astype(bool).any()


def test_production_saved_models_and_earlier_phases_are_unchanged() -> None:
    contract = json.loads((OUTPUT_DIR / "02_locked_contract.json").read_text())
    assert protected_fingerprints() == contract["protected_fingerprints_before"]
    assert prior_phase_fingerprints() == contract["prior_phase_fingerprints_before"]
