from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase2c_lite_calibration import (
    ALIGNMENT_KEY,
    EXPECTED_B1_PREDICTIONS_SHA256,
    EXPECTED_B2_PREDICTIONS_SHA256,
    EXPECTED_RAW_COVERAGE,
    EXPECTED_SNAPSHOT_SHA256,
    EXPECTED_SOURCE_SHA256,
    OUTPUT_DIR,
    PHASE2B1_PREDICTIONS,
    PHASE2B2_PREDICTIONS,
    PREDICTION_KEY,
    REQUIRED_ARTIFACTS,
    SNAPSHOT_PATH,
    SOURCE_PATH,
    prior_phase_fingerprints,
    protected_fingerprints,
)
from scripts.run_phase2a5_supabase_reconciliation import sha256_file
from src.calibration import C0, CALIBRATION_POLICIES, PRIMARY_POLICIES


pytestmark = pytest.mark.skipif(
    not SOURCE_PATH.is_file() or not (OUTPUT_DIR / "phase2c_manifest.json").is_file(),
    reason="Phase 2C reproduction requires ignored frozen inputs and artifacts",
)


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "05_calibrated_predictions.csv",
        parse_dates=["forecast_origin", "target_date", "calibration_cutoff_date"],
        low_memory=False,
    )


def test_frozen_input_hashes_and_raw_coverage_reproduce() -> None:
    assert sha256_file(SNAPSHOT_PATH) == EXPECTED_SNAPSHOT_SHA256
    assert sha256_file(SOURCE_PATH) == EXPECTED_SOURCE_SHA256
    assert sha256_file(PHASE2B1_PREDICTIONS) == EXPECTED_B1_PREDICTIONS_SHA256
    assert sha256_file(PHASE2B2_PREDICTIONS) == EXPECTED_B2_PREDICTIONS_SHA256
    predictions = load_predictions()
    raw = predictions[predictions["calibration_policy_id"] == C0.calibration_policy_id]
    assert np.isclose(raw["covered"].astype(bool).mean(), EXPECTED_RAW_COVERAGE, rtol=0, atol=1e-15)


def test_prediction_keys_policy_grid_and_point_predictions_are_frozen() -> None:
    predictions = load_predictions()
    assert not predictions.duplicated(PREDICTION_KEY).any()
    combinations = predictions[["calibration_policy_id", "coverage_target"]].drop_duplicates()
    assert len(combinations) == 17
    assert set(combinations["calibration_policy_id"]) == {
        policy.calibration_policy_id for policy in CALIBRATION_POLICIES
    }
    assert set(combinations.loc[combinations["calibration_policy_id"] == C0.calibration_policy_id, "coverage_target"]) == {0.8}
    for policy in PRIMARY_POLICIES:
        values = combinations.loc[combinations["calibration_policy_id"] == policy.calibration_policy_id, "coverage_target"]
        assert set(values) == {0.75, 0.8, 0.85, 0.9}
    reference = pd.read_csv(PHASE2B2_PREDICTIONS, parse_dates=["forecast_origin", "target_date"])
    reference = reference[reference["sample_weight_id"] == "SW_UNIFORM"].sort_values(ALIGNMENT_KEY).reset_index(drop=True)
    for _, part in predictions.groupby(["calibration_policy_id", "coverage_target"]):
        part = part.sort_values(ALIGNMENT_KEY).reset_index(drop=True)
        assert part[ALIGNMENT_KEY].equals(reference[ALIGNMENT_KEY])
        np.testing.assert_allclose(part["point_prediction"], reference["point_prediction"], rtol=0, atol=0)


def test_calibration_cutoffs_are_origin_valid_and_no_buffers_stack() -> None:
    predictions = load_predictions()
    eligible_cutoff = predictions["calibration_cutoff_date"].notna()
    assert (predictions.loc[eligible_cutoff, "calibration_cutoff_date"] < predictions.loc[eligible_cutoff, "target_date"]).all()
    assert (predictions.loc[eligible_cutoff, "calibration_cutoff_date"] <= predictions.loc[eligible_cutoff, "forecast_origin"]).all()
    assert predictions["provenance_valid"].astype(bool).all()
    assert not predictions["legacy_percentage_buffer_added"].astype(bool).any()
    assert not predictions["saved_residual_buffer_added"].astype(bool).any()
    eligible = predictions["recommended_meals"].notna()
    np.testing.assert_array_equal(
        predictions.loc[eligible, "recommended_meals"],
        np.ceil(predictions.loc[eligible, "unrounded_upper_recommendation"]),
    )


def test_lock_precedes_confirmation_and_exact_guardrail_fallback_is_recorded() -> None:
    lock = json.loads((OUTPUT_DIR / "07_locked_calibration_policy.json").read_text())
    assert lock["development_locked_policy_id"] == "C3_LAST26_DAYTYPE"
    assert lock["final_recommended_policy_id"] == C0.calibration_policy_id
    assert lock["temporary_fallback_to_c0"] is True
    assert lock["alternate_calibrated_policy_searched_on_confirmation"] is False
    assert lock["selection_locked_at_utc_before_confirmation_review"] < lock["confirmation_reviewed_at_utc"]
    confirmation = pd.read_csv(OUTPUT_DIR / "08_confirmation_guardrails.csv")
    c3 = confirmation[confirmation["calibration_policy_id"] == "C3_LAST26_DAYTYPE"].iloc[0]
    assert c3["mean_over_vs_raw"] > 5.0
    assert not bool(c3["passes_mean_overpreparation"])


def test_manifest_has_exact_artifacts_zero_fits_and_integrity() -> None:
    assert sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()) == sorted(REQUIRED_ARTIFACTS)
    manifest = json.loads((OUTPUT_DIR / "phase2c_manifest.json").read_text())
    contract = json.loads((OUTPUT_DIR / "02_locked_contract.json").read_text())
    assert manifest["required_artifact_count"] == 17
    assert manifest["calibration_history_cache_build_count"] == 1
    assert manifest["point_model_fit_count"] == 0
    assert manifest["quantile_model_fit_count"] == 0
    assert manifest["total_model_fit_count"] == 0
    assert manifest["point_predictions_changed"] is False
    assert protected_fingerprints() == contract["protected_fingerprints_before"]
    assert prior_phase_fingerprints() == contract["prior_phase_fingerprints_before"]
