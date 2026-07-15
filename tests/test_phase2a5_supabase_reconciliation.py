from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_phase2a5_supabase_reconciliation import (
    EXPECTED_SOURCE_SHA256,
    EXTENSION_END,
    LOCATION_ID,
    MATCHED_END,
    OUTPUT_DIR,
    PHASE1_DIR,
    PHASE2A_DIR,
    PREDICTION_COLUMNS,
    SOURCE_PATH,
    SOURCE_RELATIVE,
    directory_fingerprint,
    enforce_duplicate_selection_rule,
    locked_feature_contract,
    protected_fingerprints,
    sha256_file,
    validate_and_normalize_export,
    window_filter,
)
from src.feature_sets import F6
from src.modeling import make_point_model, make_quantile_model
from src.origin_features import MODEL_FEATURES, T1_VALID_WEEKENDS, W0_NO_WEATHER


ROOT = Path(__file__).resolve().parents[1]


def validation_payload() -> dict:
    return json.loads((OUTPUT_DIR / "02_supabase_csv_validation.json").read_text())


def test_exact_configured_csv_path_is_used() -> None:
    assert SOURCE_RELATIVE == Path(
        "data/locations/ny_12550/Updated/2026-07-15T05-23_export.csv"
    )
    assert SOURCE_PATH == ROOT / SOURCE_RELATIVE


def test_csv_path_exists() -> None:
    assert SOURCE_PATH.is_file()


def test_original_csv_hash_remains_unchanged() -> None:
    assert sha256_file(SOURCE_PATH) == EXPECTED_SOURCE_SHA256
    payload = validation_payload()
    assert payload["sha256_before_validation"] == EXPECTED_SOURCE_SHA256
    assert payload["sha256_after_validation"] == EXPECTED_SOURCE_SHA256
    assert payload["original_csv_unchanged"] is True


def test_location_filtering_is_explicitly_path_scoped() -> None:
    snapshot = pd.read_csv(OUTPUT_DIR / "03_normalized_supabase_snapshot.csv")
    assert set(snapshot["location_id"]) == {LOCATION_ID}
    assert validation_payload()["location_membership_basis"].startswith(
        "configured path inference"
    )


def test_date_parsing_preserves_service_dates_without_timezone_shift() -> None:
    normalized, payload = validate_and_normalize_export()
    assert normalized["service_date"].iloc[0] == "2023-01-01"
    assert normalized["service_date"].iloc[-1] == "2026-07-12"
    assert payload["timestamp_conversion_changes_service_dates"] is False
    assert "date-only" in payload["timezone_interpretation"]


def test_duplicate_selection_rule_stops_when_schema_cannot_resolve() -> None:
    with pytest.raises(ValueError, match="cannot be resolved safely"):
        enforce_duplicate_selection_rule(
            pd.Series(["2026-07-12", "2026-07-12"], dtype="string")
        )
    enforce_duplicate_selection_rule(pd.Series(["2026-07-11", "2026-07-12"]))


def test_status_deleted_cancelled_inactive_and_test_rules_are_documented() -> None:
    payload = validation_payload()
    for key in ["deleted_records", "cancelled_records", "inactive_records", "test_records"]:
        assert payload[key]["status"] == "schema_unavailable"
    assert payload["excluded_row_count"] == 0


def test_latest_expected_date_is_detected_and_nothing_later_exists() -> None:
    payload = validation_payload()
    assert payload["contains_2026_07_12"] is True
    assert payload["maximum_service_date"] == "2026-07-12"
    assert payload["records_after_2026_07_12"] == []


def test_normalized_rows_have_unique_location_date_keys() -> None:
    snapshot = pd.read_csv(OUTPUT_DIR / "03_normalized_supabase_snapshot.csv")
    assert len(snapshot) == 360
    assert not snapshot.duplicated(["location_id", "service_date"]).any()


def test_matched_and_extension_windows_use_exact_boundaries() -> None:
    sample = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2026-06-21", "2026-06-27", "2026-07-12"]),
            "actual": [160, 102, 170],
        }
    )
    matched = window_filter(sample, "matched_history_through_2026_06_21")
    extension = window_filter(sample, "new_extension_after_2026_06_21")
    assert matched["target_date"].max() == MATCHED_END
    assert (extension["target_date"] > MATCHED_END).all()
    assert (extension["target_date"] <= EXTENSION_END).all()


def test_locked_phase2a_feature_list_matches_all_artifacts() -> None:
    lock = locked_feature_contract()
    serialized = json.loads((OUTPUT_DIR / "07_locked_feature_set.json").read_text())
    assert lock["selected_feature_set_id"] == serialized["selected_feature_set_id"] == F6
    assert lock["ordered_feature_list"] == serialized["ordered_feature_list"]
    assert len(lock["ordered_feature_list"]) == 33


def test_f0_and_model_configurations_match_phase1_phase2a() -> None:
    lock = locked_feature_contract()
    assert lock["f0_ordered_feature_list"] == MODEL_FEATURES
    assert lock["training_rules"]["weekday_policy"] == T1_VALID_WEEKENDS
    assert lock["training_rules"]["weather_policy"] == W0_NO_WEATHER
    assert lock["training_rules"]["min_segment_training_rows"] == 18
    point = make_point_model().get_params()
    quantile = make_quantile_model(0.8).get_params()
    assert point["n_estimators"] == 400
    assert point["max_depth"] == 8
    assert point["min_samples_leaf"] == 2
    assert point["random_state"] == 42
    assert quantile["loss"] == "quantile"
    assert quantile["quantile"] == 0.8
    assert quantile["learning_rate"] == 0.05
    assert quantile["max_depth"] == 4
    assert quantile["max_iter"] == 500
    assert quantile["random_state"] == 42


def test_prediction_schema_keys_and_windows_are_valid() -> None:
    predictions = pd.read_csv(
        OUTPUT_DIR / "08_latest_snapshot_predictions.csv",
        parse_dates=["forecast_origin", "target_date", "training_end_date"],
    )
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    key = [
        "data_snapshot_id",
        "candidate_id",
        "forecast_origin",
        "target_date",
        "scenario",
        "weather_policy",
        "weekday_policy",
    ]
    assert not predictions.duplicated(key).any()
    assert (predictions["forecast_origin"] < predictions["target_date"]).all()


def test_attendance_provenance_dates_are_on_or_before_origins() -> None:
    provenance = pd.read_csv(
        OUTPUT_DIR / "feature_value_provenance.csv",
        parse_dates=["forecast_origin", "target_date"],
        low_memory=False,
    )
    attendance = provenance[provenance["source_type"] == "attendance"]
    assert not attendance.empty
    for row in attendance.itertuples(index=False):
        assert all(
            pd.Timestamp(value) <= row.forecast_origin
            for value in json.loads(row.available_source_dates)
        )


def test_fold_local_preprocessing_cutoffs_remain_valid() -> None:
    diagnostics = pd.read_csv(
        OUTPUT_DIR / "fold_preprocessing_diagnostics.csv",
        parse_dates=["training_start_date", "training_end_date"],
    )
    assert not diagnostics.empty
    assert not diagnostics["fit_includes_test_or_future"].astype(bool).any()
    predictions = pd.read_csv(
        OUTPUT_DIR / "08_latest_snapshot_predictions.csv",
        parse_dates=["forecast_origin", "training_end_date"],
    )
    model = predictions[predictions["candidate_id"].isin(["F0_CURRENT_ORIGIN", F6])]
    assert (model["training_end_date"] <= model["forecast_origin"]).all()


def test_prior_phase_artifacts_remain_unchanged() -> None:
    recorded = validation_payload()["prior_artifact_directory_fingerprints_at_validation"]
    assert recorded[str(PHASE1_DIR.relative_to(ROOT))] == directory_fingerprint(PHASE1_DIR)
    assert recorded[str(PHASE2A_DIR.relative_to(ROOT))] == directory_fingerprint(PHASE2A_DIR)


def test_saved_packages_databases_and_production_files_remain_unchanged() -> None:
    recorded = validation_payload()["protected_file_fingerprints_at_validation"]
    assert recorded == protected_fingerprints()
    assert recorded["models/visitor_model_ny_12550.joblib"] == (
        "061be9292fb85cecbd4b5ab2a213bba9d4a692305d23234ff3f81c49affc94f2"
    )
    assert recorded["app.py"] == (
        "39da681d2bf7efd020e58296bbf4293ac3a3f71d2242e4f329ed409663b86f5f"
    )
    assert recorded["src/predictor.py"] == (
        "f4d21d414817094a796c659c1093043431b57c80fb633b2d7745853a663393db"
    )


def test_required_old_phase_directories_exist_and_are_read_only_inputs() -> None:
    assert (PHASE1_DIR / "phase1_manifest.json").is_file()
    assert (PHASE2A_DIR / "phase2a_manifest.json").is_file()
    assert not (OUTPUT_DIR / "phase1_manifest.json").exists()
    assert not (OUTPUT_DIR / "phase2a_manifest.json").exists()


def test_old_snapshot_prediction_differences_are_isolated_to_source_insertion() -> None:
    comparison = pd.read_csv(
        OUTPUT_DIR / "old_snapshot_prediction_reconciliation.csv",
        parse_dates=["target_date", "forecast_origin_old", "forecast_origin_new"],
    )
    changed = comparison[
        ~comparison["point_exact_match"] | ~comparison["quantile_exact_match"]
    ]
    assert set(changed["target_date"]) == {pd.Timestamp("2026-06-21")}
    counts = changed.groupby("candidate_id").size().to_dict()
    assert counts == {"F0_CURRENT_ORIGIN": 4, F6: 1}
    exact = comparison.groupby("candidate_id")[["point_exact_match", "quantile_exact_match"]].sum()
    assert int(exact.loc["F0_CURRENT_ORIGIN", "point_exact_match"]) == 1252
    assert int(exact.loc[F6, "point_exact_match"]) == 1255
