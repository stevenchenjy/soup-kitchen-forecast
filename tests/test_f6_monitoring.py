from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import prediction_logs
from src.f6_monitoring import (
    ActiveF6Package,
    TRAINING_PROVENANCE_KEY,
    active_f6_package,
    build_f6_monitoring_report,
    f6_training_status,
    filter_active_f6_rows,
    monitoring_stage,
)
from src.production_features import LOCKED_F6_FEATURE_ORDER_SHA256


PACKAGE_ID = "ny_12550_f6_2026-07-12_v1"
FEATURE_SET_ID = "F6_COMPACT_SELECTED"
POLICY_ID = "C0_EXISTING_RAW_QUANTILE"


def predictor(package_id: str = PACKAGE_ID, **overrides):
    values = {
        "package_id": package_id,
        "model_package_schema_version": 2,
        "uses_locked_f6": True,
        "feature_cols": [f"feature_{index}" for index in range(33)],
        "feature_contract": {
            "feature_set_id": FEATURE_SET_ID,
            "feature_order_sha256": LOCKED_F6_FEATURE_ORDER_SHA256,
        },
        "recommendation_policy_id": POLICY_ID,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def active_row(
    *,
    row_id: int = 1,
    service_date: str = "2026-07-18",
    predicted: float = 100.0,
    quantile: float = 115.0,
    suggested: int = 115,
    actual: int | None = 110,
    segment: str = "sat",
    horizon: int = 1,
    waste: float = 10.0,
    co2e: float = 17.1,
    package_id: str = PACKAGE_ID,
    **overrides,
) -> dict:
    row = {
        "id": row_id,
        "location_id": "ny_12550",
        "service_date": service_date,
        "prediction_created_at": f"{service_date}T12:00:00+00:00",
        "predicted_visitors": predicted,
        "predicted_quantile": quantile,
        "suggested_meals": suggested,
        "model_segment": segment,
        "package_id": package_id,
        "model_package_schema_version": 2,
        "feature_set_id": FEATURE_SET_ID,
        "feature_order_sha256": LOCKED_F6_FEATURE_ORDER_SHA256,
        "recommendation_policy_id": POLICY_ID,
        "service_horizon": horizon,
        "actual_visitors": actual,
        "absolute_error": None if actual is None else abs(actual - predicted),
        "waste_avoided_meals": waste,
        "estimated_co2e_reduction_kg": co2e,
        "created_at": f"{service_date}T12:00:00+00:00",
        "updated_at": f"{service_date}T12:00:00+00:00",
    }
    row.update(overrides)
    return row


def legacy_row(row_id: int, service_date: str) -> dict:
    return active_row(
        row_id=row_id,
        service_date=service_date,
        predicted=999.0,
        quantile=1000.0,
        suggested=1000,
        actual=1,
        waste=500.0,
        co2e=855.0,
        package_id="visitor_model_ny_12550.joblib",
        model_package_schema_version=1,
        feature_set_id=None,
        feature_order_sha256=None,
        recommendation_policy_id="OLDER_POLICY",
    )


def mixed_rows() -> list[dict]:
    rows = [
        legacy_row(index, f"2026-05-{index:02d}") for index in range(1, 11)
    ]
    rows.extend(
        [
            active_row(row_id=11),
            active_row(
                row_id=12,
                service_date="2026-07-19",
                predicted=150.0,
                quantile=145.0,
                suggested=145,
                actual=130,
                segment="sun",
                horizon=2,
                waste=12.0,
                co2e=20.52,
            ),
            active_row(
                row_id=13,
                service_date="2026-07-25",
                predicted=90.0,
                quantile=105.0,
                suggested=105,
                actual=80,
                package_id="ny_12550_f6_inactive_v1",
            ),
            active_row(
                row_id=14,
                service_date="2026-07-26",
                predicted=120.0,
                quantile=130.0,
                suggested=130,
                actual=None,
                segment="sun",
                horizon=5,
                waste=13.0,
                co2e=22.23,
            ),
        ]
    )
    return rows


@pytest.mark.parametrize(
    "updates",
    [
        {"package_id": "visitor_model_ny_12550.joblib"},
        {"package_id": None},
        {"package_id": "wrong-f6-package"},
        {"model_package_schema_version": 1},
        {"feature_set_id": "F0_CURRENT_ORIGIN"},
        {"feature_order_sha256": "wrong-hash"},
        {"feature_order_sha256": None, "feature_hash": "wrong-hash"},
        {"feature_hash": "wrong-hash"},
        {"recommendation_policy_id": "wrong-policy"},
    ],
)
def test_non_active_or_incomplete_provenance_is_excluded(updates: dict) -> None:
    contract = active_f6_package(predictor())
    row = active_row(**updates)
    assert filter_active_f6_rows([row], contract) == []


def test_feature_hash_is_optional_when_row_schema_does_not_expose_it() -> None:
    contract = active_f6_package(predictor())
    row = active_row(feature_order_sha256=None)
    assert filter_active_f6_rows([row], contract) == [row]


def test_all_available_feature_hash_fields_must_match() -> None:
    contract = active_f6_package(predictor())
    row = active_row(feature_hash="wrong-hash")
    assert filter_active_f6_rows([row], contract) == []


def test_active_f6_row_is_included() -> None:
    contract = active_f6_package(predictor())
    row = active_row()
    assert filter_active_f6_rows([row], contract) == [row]


def test_mixed_input_produces_active_f6_only_metrics_and_sustainability() -> None:
    report = build_f6_monitoring_report(predictor(), mixed_rows())

    assert report["integrity_ok"] is True
    assert report["prediction_count"] == 3
    assert report["reconciled_count"] == 2
    assert report["unreconciled_count"] == 1
    assert report["stage"] == "INSUFFICIENT_DATA"
    assert report["metrics"]["mae"] == pytest.approx(15.0)
    assert report["metrics"]["median_absolute_error"] == pytest.approx(15.0)
    assert report["metrics"]["rmse"] == pytest.approx(15.811388300841896)
    assert report["metrics"]["mean_signed_error"] == pytest.approx(5.0)
    assert report["metrics"]["underprediction_rate"] == pytest.approx(0.5)
    assert report["metrics"]["q80_empirical_coverage"] == pytest.approx(1.0)
    assert report["metrics"]["mean_over_preparation"] == pytest.approx(10.0)
    assert report["metrics"]["mean_under_preparation"] == pytest.approx(0.0)
    assert report["segments"]["Saturday"]["mae"] == pytest.approx(10.0)
    assert report["segments"]["Sunday"]["mae"] == pytest.approx(20.0)
    assert report["horizons"]["H1"]["mae"] == pytest.approx(10.0)
    assert report["horizons"]["H2"]["mae"] == pytest.approx(20.0)
    assert report["horizons"]["H5"]["mae"] is None
    assert report["sustainability"] == {
        "total_estimated_waste_avoided_meals": pytest.approx(35.0),
        "total_estimated_co2e_reduction_kg": pytest.approx(59.85),
    }


def test_latest_outcomes_contain_only_the_active_package() -> None:
    report = build_f6_monitoring_report(predictor(), mixed_rows())
    outcomes = report["latest_outcomes"]
    assert [row["id"] for row in outcomes] == [14, 12, 11]
    assert {row["package_id"] for row in outcomes} == {PACKAGE_ID}


def test_zero_f6_rows_never_falls_back_to_other_metrics() -> None:
    report = build_f6_monitoring_report(
        predictor(), [legacy_row(1, "2026-05-01")]
    )
    assert report["prediction_count"] == 0
    assert report["reconciled_count"] == 0
    assert report["unreconciled_count"] == 0
    assert report["stage"] == "INSUFFICIENT_DATA"
    assert all(value is None for value in report["metrics"].values())
    assert report["latest_outcomes"] == []
    assert report["sustainability"] == {
        "total_estimated_waste_avoided_meals": 0,
        "total_estimated_co2e_reduction_kg": 0,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_package_schema_version": 1, "uses_locked_f6": False},
        {"feature_contract": {"feature_set_id": "wrong", "feature_order_sha256": LOCKED_F6_FEATURE_ORDER_SHA256}},
        {"feature_contract": {"feature_set_id": FEATURE_SET_ID, "feature_order_sha256": "wrong"}},
        {"recommendation_policy_id": "wrong"},
    ],
)
def test_active_package_integrity_failure_suppresses_performance(
    overrides: dict,
) -> None:
    report = build_f6_monitoring_report(predictor(**overrides), [active_row()])
    assert report["integrity_ok"] is False
    assert report["metrics"] is None
    assert report["latest_outcomes"] == []
    assert report["sustainability"] is None


def test_active_package_change_updates_filter_automatically() -> None:
    newer_id = "ny_12550_f6_nightly_2026-07-19_v1"
    rows = [active_row(), active_row(row_id=2, package_id=newer_id)]

    old_report = build_f6_monitoring_report(predictor(), rows)
    new_report = build_f6_monitoring_report(predictor(newer_id), rows)

    assert [row["package_id"] for row in old_report["latest_outcomes"]] == [PACKAGE_ID]
    assert [row["package_id"] for row in new_report["latest_outcomes"]] == [newer_id]


def test_legacy_training_timestamp_is_not_presented_as_f6_training_time() -> None:
    contract = active_f6_package(predictor())
    old_run = {
        "status": "success",
        "finished_at": "2026-07-13T10:21:30+00:00",
        "metrics": {"overall": {"MAE": 10.0}},
    }
    status = f6_training_status(
        contract,
        retrain_state={"dirty": False},
        latest_run=old_run,
        latest_successful_run=old_run,
    )
    assert status["latest_successful_at"] is None
    assert status["status"] == "PENDING_FIRST_F6_RETRAIN"
    assert status["message"] == (
        "Activated from verified candidate; first genuine F6 retraining pending"
    )


def test_confirmed_active_f6_training_run_is_shown() -> None:
    contract = active_f6_package(predictor())
    provenance = {
        "package_id": PACKAGE_ID,
        "model_package_schema_version": 2,
        "feature_set_id": FEATURE_SET_ID,
        "feature_order_sha256": LOCKED_F6_FEATURE_ORDER_SHA256,
        "recommendation_policy_id": POLICY_ID,
    }
    run = {
        "status": "success",
        "finished_at": "2026-07-20T10:00:00+00:00",
        "attendance_rows": 362,
        "metrics": {TRAINING_PROVENANCE_KEY: provenance},
    }
    status = f6_training_status(
        contract,
        retrain_state={"dirty": False},
        latest_run=run,
        latest_successful_run=run,
    )
    assert status["latest_successful_at"] == "2026-07-20T10:00:00+00:00"
    assert status["status"] == "SUCCESS"
    assert status["attendance_rows"] == 362


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "INSUFFICIENT_DATA"), (3, "INSUFFICIENT_DATA"), (4, "EARLY_SIGNAL"), (7, "EARLY_SIGNAL"), (8, "INITIAL_REVIEW"), (11, "INITIAL_REVIEW"), (12, "STABLE_REVIEW")],
)
def test_monitoring_stage_boundaries(count: int, expected: str) -> None:
    assert monitoring_stage(count) == expected


def test_temporary_prediction_store_keeps_history_but_report_filters_it(
    tmp_path: Path,
) -> None:
    db = tmp_path / "prediction-logs.db"
    with patch.object(prediction_logs, "location_db_file", return_value=db), patch.object(
        prediction_logs, "_supabase_config", return_value=None
    ):
        with prediction_logs._connect("ny_12550") as conn:
            for row in [legacy_row(1, "2026-05-01"), active_row(row_id=2)]:
                columns = list(row)
                placeholders = ", ".join("?" for _ in columns)
                conn.execute(
                    f"INSERT INTO prediction_logs ({', '.join(columns)}) VALUES ({placeholders})",
                    [row[column] for column in columns],
                )
            conn.commit()
        loaded = prediction_logs.load_prediction_logs("ny_12550", limit=20)

    assert len(loaded) == 2
    report = build_f6_monitoring_report(predictor(), loaded)
    assert report["prediction_count"] == 1
    assert report["reconciled_count"] == 1
    assert [row["id"] for row in report["latest_outcomes"]] == [2]
