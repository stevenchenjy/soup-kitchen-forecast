from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src import prediction_logs
from src.f6_monitoring import (
    ActiveF6Package,
    BacktestChartError,
    BacktestSummaryError,
    TRAINING_PROVENANCE_KEY,
    active_f6_package,
    build_operational_impact,
    build_f6_monitoring_report,
    f6_training_status,
    filter_active_f6_rows,
    format_dashboard_date,
    load_verified_backtest_chart_series,
    load_verified_backtest_summary,
    monitoring_stage,
    retraining_status_label,
)
from src.production_features import LOCKED_F6_FEATURE_ORDER_SHA256


PACKAGE_ID = "ny_12550_f6_2026-07-12_v1"
FEATURE_SET_ID = "F6_COMPACT_SELECTED"
POLICY_ID = "C0_EXISTING_RAW_QUANTILE"
ROOT = Path(__file__).resolve().parents[1]
BACKTEST_SUMMARY = ROOT / "config/model_backtests" / f"{PACKAGE_ID}.json"
BACKTEST_CHART = (
    ROOT / "config/model_backtests" / f"{PACKAGE_ID}_predictions.csv"
)
BACKTEST_CHART_SHA256 = (
    "8cd73e08d215c2e428e4b99ce5fd972e8be3ac3ed9b6fb36e43aef1df956d982"
)
POINT_RESEARCH_METRICS = (
    ROOT
    / "artifacts/ny_12550/model_optimization/phase2b1_training_windows/06_training_window_metrics.csv"
)
OPERATIONAL_RESEARCH_METRICS = (
    ROOT
    / "artifacts/ny_12550/model_optimization/phase2c_lite_calibration/10_daytype_scenario_horizon.csv"
)
CHART_RESEARCH_PREDICTIONS = (
    ROOT
    / "artifacts/ny_12550/model_optimization/phase2c_lite_calibration/05_calibrated_predictions.csv"
)


def temporary_backtest_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    bundle_dir = tmp_path / "config/model_backtests"
    bundle_dir.mkdir(parents=True)
    summary_path = bundle_dir / BACKTEST_SUMMARY.name
    chart_path = bundle_dir / BACKTEST_CHART.name
    summary_path.write_bytes(BACKTEST_SUMMARY.read_bytes())
    chart_path.write_bytes(BACKTEST_CHART.read_bytes())
    monkeypatch.setattr("src.f6_monitoring.PROJECT_ROOT", tmp_path)
    return summary_path, chart_path


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


def test_verified_backtest_summary_loads_exact_active_package_metrics() -> None:
    summary = load_verified_backtest_summary(active_f6_package(predictor()))

    assert summary["attendance_cutoff"] == "2026-07-12"
    assert summary["evaluation"]["design"] == "origin_aware"
    assert summary["evaluation"]["row_count"] == 322
    assert summary["metrics"]["mae"] == pytest.approx(13.62696513246822)
    assert summary["metrics"]["rmse"] == pytest.approx(17.738338356170434)
    assert summary["metrics"]["median_absolute_error"] == pytest.approx(
        10.253146148989984
    )
    assert summary["metrics"]["mean_signed_error"] == pytest.approx(
        0.22609781014812802
    )
    assert summary["metrics"]["underprediction_rate"] == pytest.approx(
        0.577639751552795
    )
    assert summary["metrics"]["q80_empirical_coverage"] == pytest.approx(
        0.6832298136645962
    )
    assert summary["metrics"]["mean_over_preparation"] == pytest.approx(
        12.341614906832298
    )
    assert summary["metrics"]["mean_under_preparation"] == pytest.approx(
        3.1335403726708075
    )
    assert summary["segments"]["Saturday"]["mae"] == pytest.approx(
        13.622258066963107
    )
    assert summary["segments"]["Sunday"]["mae"] == pytest.approx(
        13.631790614715598
    )
    assert summary["horizons"]["H1"]["mae"] == pytest.approx(
        13.612895542207069
    )
    assert summary["horizons"]["H2"]["mae"] == pytest.approx(
        13.632493365962487
    )
    assert summary["horizons"]["H5"]["mae"] == pytest.approx(
        14.21230873930077
    )


def test_verified_backtest_summary_loads_without_deployed_source_artifacts(
    tmp_path: Path,
) -> None:
    with patch("src.f6_monitoring.PROJECT_ROOT", tmp_path):
        summary = load_verified_backtest_summary(active_f6_package(predictor()))

    assert summary["verification_status"] == "verified"
    assert summary["package_id"] == PACKAGE_ID


def test_tracked_backtest_chart_is_registered_with_matching_hash() -> None:
    summary = load_verified_backtest_summary(active_f6_package(predictor()))
    registration = summary["chart_dataset"]

    assert BACKTEST_CHART.is_file()
    assert registration["path"] == BACKTEST_CHART.relative_to(ROOT).as_posix()
    assert registration["sha256"] == BACKTEST_CHART_SHA256
    assert (
        hashlib.sha256(BACKTEST_CHART.read_bytes()).hexdigest()
        == BACKTEST_CHART_SHA256
    )
    assert registration["row_count"] == summary["evaluation"]["row_count"] == 322
    assert registration["scope_type"] == "scenario"
    assert registration["scope_value"] == summary["evaluation"]["primary_scope"]


def test_verified_backtest_chart_series_is_complete_finite_and_consistent() -> None:
    frame = load_verified_backtest_chart_series(active_f6_package(predictor()))
    required_columns = {
        "service_date",
        "actual_visitors",
        "point_prediction",
        "q80_prediction",
        "absolute_error",
        "model_segment",
        "service_horizon",
        "scenario",
    }
    numeric_columns = [
        "actual_visitors",
        "point_prediction",
        "q80_prediction",
        "absolute_error",
        "service_horizon",
    ]

    assert required_columns.issubset(frame.columns)
    assert len(frame) == 322
    assert frame["service_date"].min() == pd.Timestamp("2023-05-13")
    assert frame["service_date"].max() == pd.Timestamp("2026-07-12")
    assert (
        frame[numeric_columns]
        .map(lambda value: float("-inf") < value < float("inf"))
        .all()
        .all()
    )
    assert frame["absolute_error"].to_numpy() == pytest.approx(
        (frame["point_prediction"] - frame["actual_visitors"]).abs().to_numpy()
    )
    assert set(frame["model_segment"]) == {"sat", "sun"}
    assert set(frame["scenario"]) == {"S3_next_service"}
    assert frame["service_date"].is_monotonic_increasing


def test_verified_backtest_chart_loader_does_not_read_ignored_sources() -> None:
    with patch(
        "src.f6_monitoring._verified_source_path",
        side_effect=AssertionError("runtime attempted to read ignored research artifacts"),
    ):
        frame = load_verified_backtest_chart_series(active_f6_package(predictor()))

    assert len(frame) == 322


@pytest.mark.skipif(
    not CHART_RESEARCH_PREDICTIONS.is_file(),
    reason="Ignored research artifacts are not available in this checkout.",
)
def test_tracked_chart_rows_match_optional_authoritative_predictions() -> None:
    source = pd.read_csv(CHART_RESEARCH_PREDICTIONS)
    expected = source[
        source["feature_set_id"].eq(FEATURE_SET_ID)
        & source["training_window_id"].eq("TW_EXPANDING")
        & source["sample_weight_id"].eq("SW_UNIFORM")
        & source["calibration_policy_id"].eq(POLICY_ID)
        & source["scenario"].eq("S3_next_service")
    ][
        [
            "target_date",
            "actual",
            "point_prediction",
            "raw_quantile_prediction",
            "absolute_error",
            "model_segment",
            "service_horizon",
            "scenario",
        ]
    ].rename(
        columns={
            "target_date": "service_date",
            "actual": "actual_visitors",
            "raw_quantile_prediction": "q80_prediction",
        }
    )
    actual = pd.read_csv(BACKTEST_CHART)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_verified_backtest_chart_rejects_missing_tracked_csv(
    tmp_path: Path, monkeypatch
) -> None:
    summary_path, chart_path = temporary_backtest_bundle(tmp_path, monkeypatch)
    chart_path.unlink()

    with pytest.raises(BacktestChartError, match="dataset is missing"):
        load_verified_backtest_chart_series(
            active_f6_package(predictor()), summary_path=summary_path
        )


def test_verified_backtest_chart_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    summary_path, chart_path = temporary_backtest_bundle(tmp_path, monkeypatch)
    chart_path.write_bytes(chart_path.read_bytes() + b"\n")

    with pytest.raises(BacktestChartError, match="hash does not match"):
        load_verified_backtest_chart_series(
            active_f6_package(predictor()), summary_path=summary_path
        )


def test_verified_backtest_chart_rejects_summary_for_another_package(
    tmp_path: Path, monkeypatch
) -> None:
    summary_path, _ = temporary_backtest_bundle(tmp_path, monkeypatch)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["package_id"] = "another-f6-package-v1"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BacktestChartError, match="package binding"):
        load_verified_backtest_chart_series(
            active_f6_package(predictor()), summary_path=summary_path
        )


def _temporary_source_verification_fixture(
    tmp_path: Path,
) -> tuple[Path, list[Path]]:
    tracked_summary = (
        ROOT / "config/model_backtests/ny_12550_f6_2026-07-12_v1.json"
    )
    summary = json.loads(tracked_summary.read_text(encoding="utf-8"))
    sources: list[Path] = []
    source_records: list[dict[str, str]] = []
    for name, contents in (
        ("point_metrics.csv", "scope,mae\nS3_next_service,13.63\n"),
        ("operational_metrics.csv", "scope,mean_over_preparation\nS3_next_service,12.34\n"),
    ):
        source = tmp_path / "synthetic_sources" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(contents, encoding="utf-8")
        sources.append(source)
        source_records.append(
            {
                "path": source.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provides": f"synthetic verification fixture for {name}",
            }
        )
    summary["source_artifacts"] = source_records
    summary_path = tmp_path / "verified_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, sources


def test_verified_backtest_source_files_and_hashes_can_be_checked_explicitly(
    tmp_path: Path,
) -> None:
    summary_path, _ = _temporary_source_verification_fixture(tmp_path)

    with patch("src.f6_monitoring.PROJECT_ROOT", tmp_path):
        summary = load_verified_backtest_summary(
            active_f6_package(predictor()),
            summary_path=summary_path,
            verify_sources=True,
        )

    assert summary["verification_status"] == "verified"


def test_verified_backtest_source_verification_rejects_missing_file(
    tmp_path: Path,
) -> None:
    summary_path, sources = _temporary_source_verification_fixture(tmp_path)
    sources[0].unlink()

    with (
        patch("src.f6_monitoring.PROJECT_ROOT", tmp_path),
        pytest.raises(BacktestSummaryError, match="source is missing"),
    ):
        load_verified_backtest_summary(
            active_f6_package(predictor()),
            summary_path=summary_path,
            verify_sources=True,
        )


def test_verified_backtest_source_verification_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    summary_path, sources = _temporary_source_verification_fixture(tmp_path)
    sources[0].write_text("scope,mae\nchanged,999\n", encoding="utf-8")

    with (
        patch("src.f6_monitoring.PROJECT_ROOT", tmp_path),
        pytest.raises(BacktestSummaryError, match="source hash does not match"),
    ):
        load_verified_backtest_summary(
            active_f6_package(predictor()),
            summary_path=summary_path,
            verify_sources=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_id", "ny_12550_f6_wrong_v1"),
        ("feature_order_sha256", "wrong-feature-hash"),
    ],
)
def test_verified_backtest_summary_remains_bound_to_active_contract(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source = ROOT / "config/model_backtests/ny_12550_f6_2026-07-12_v1.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    summary[field] = value
    modified = tmp_path / "modified-summary.json"
    modified.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BacktestSummaryError, match=field):
        load_verified_backtest_summary(
            active_f6_package(predictor()), summary_path=modified
        )


def test_verified_backtest_summary_matches_tracked_candidate_metadata() -> None:
    summary = load_verified_backtest_summary(active_f6_package(predictor()))
    metadata = json.loads(
        (
            ROOT
            / "models/candidates/ny_12550_f6_2026-07-12_v1/metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["package_id"] == metadata["package_id"]
    assert (
        summary["model_package_schema_version"]
        == metadata["model_package_schema_version"]
    )
    assert summary["feature_set_id"] == metadata["feature_contract"]["feature_set_id"]
    assert (
        summary["feature_order_sha256"]
        == metadata["feature_contract"]["feature_order_sha256"]
    )
    assert (
        summary["recommendation_policy_id"]
        == metadata["recommendation_policy_id"]
    )
    assert summary["attendance_cutoff"] == metadata["attendance_input"][
        "maximum_service_date"
    ]
    assert summary["evaluation"]["row_count"] == metadata["metrics"]["overall"][
        "BacktestRows"
    ]
    assert summary["metrics"]["mae"] == pytest.approx(
        metadata["metrics"]["overall"]["MAE"]
    )
    assert summary["metrics"]["rmse"] == pytest.approx(
        metadata["metrics"]["overall"]["RMSE"]
    )
    assert summary["segments"]["Saturday"]["mae"] == pytest.approx(
        metadata["metrics"]["sat"]["MAE"]
    )
    assert summary["segments"]["Sunday"]["mae"] == pytest.approx(
        metadata["metrics"]["sun"]["MAE"]
    )


@pytest.mark.skipif(
    not POINT_RESEARCH_METRICS.is_file()
    or not OPERATIONAL_RESEARCH_METRICS.is_file(),
    reason="Ignored research artifacts are not available in this checkout.",
)
def test_verified_backtest_summary_matches_optional_research_artifacts() -> None:
    summary = load_verified_backtest_summary(active_f6_package(predictor()))
    point_metrics = pd.read_csv(POINT_RESEARCH_METRICS)
    operational_metrics = pd.read_csv(OPERATIONAL_RESEARCH_METRICS)
    primary = point_metrics[
        point_metrics["training_window_id"].eq("TW_EXPANDING")
        & point_metrics["evaluation_scope"].eq("scenario")
        & point_metrics["scope_value"].eq("S3_next_service")
    ].iloc[0]
    preparation = operational_metrics[
        operational_metrics["calibration_policy_id"].eq(POLICY_ID)
        & operational_metrics["scope_type"].eq("scenario")
        & operational_metrics["scope_value"].eq("S3_next_service")
    ].iloc[0]

    assert summary["metrics"]["median_absolute_error"] == pytest.approx(
        primary["median_absolute_error"]
    )
    assert summary["metrics"]["mean_signed_error"] == pytest.approx(
        primary["mean_signed_error"]
    )
    assert summary["metrics"]["underprediction_rate"] == pytest.approx(
        primary["underprediction_frequency"]
    )
    assert summary["metrics"]["q80_empirical_coverage"] == pytest.approx(
        primary["raw_quantile_coverage"]
    )
    assert summary["metrics"]["mean_over_preparation"] == pytest.approx(
        preparation["mean_over_preparation"]
    )
    assert summary["metrics"]["mean_under_preparation"] == pytest.approx(
        preparation["mean_under_preparation"]
    )
    for horizon in (1, 2, 5):
        row = point_metrics[
            point_metrics["training_window_id"].eq("TW_EXPANDING")
            & point_metrics["evaluation_scope"].eq("service_horizon")
            & point_metrics["scope_value"].astype(str).eq(str(horizon))
        ].iloc[0]
        assert summary["horizons"][f"H{horizon}"]["mae"] == pytest.approx(
            row["mae"]
        )


def test_verified_backtest_summary_uses_registered_reference_for_nightly_lineage() -> None:
    summary = load_verified_backtest_summary(
        active_f6_package(predictor("ny_12550_f6_nightly_2026-07-19_v1"))
    )

    assert summary["is_reference_backtest"] is True
    assert summary["package_id"] == PACKAGE_ID
    assert summary["active_package_id"] == "ny_12550_f6_nightly_2026-07-19_v1"


def test_verified_backtest_summary_rejects_unregistered_active_package() -> None:
    with pytest.raises(BacktestSummaryError, match="No verified historical backtest"):
        load_verified_backtest_summary(
            active_f6_package(predictor("unregistered_f6_package_v1"))
        )


def test_operational_impact_is_cumulative_across_all_packages() -> None:
    report = build_operational_impact(360, mixed_rows())

    assert report["attendance_row_count"] == 360
    assert report["prediction_log_count"] == 14
    assert report["reconciled_log_count"] == 13
    assert report["estimated_food_saved_meals"] == pytest.approx(5045.0)
    assert report["estimated_co2e_reduction_kg"] == pytest.approx(8626.95)


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
        "Verified candidate active. First production retraining has not completed."
    )


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("PENDING_FIRST_F6_RETRAIN", "Pending first retrain"),
        ("RETRAINING_REQUIRED", "Retraining required"),
        ("SUCCESS", "Up to date"),
        ("FAILED", "Last retrain failed"),
        ("DEPLOYMENT_MISMATCH", "Configuration mismatch"),
        ("unexpected_internal_value", "Status unavailable"),
        (None, "Status unavailable"),
    ],
)
def test_retraining_status_labels_are_plain_language(
    status: str | None,
    label: str,
) -> None:
    assert retraining_status_label(status) == label


@pytest.mark.parametrize(
    ("value", "formatted"),
    [
        ("2026-07-12", "Jul 12, 2026"),
        ("2026-07-12T18:42:11+00:00", "Jul 12, 2026"),
        ("2026-07-12T18:42:11Z", "Jul 12, 2026"),
        (None, "—"),
        ("not-a-date", "—"),
    ],
)
def test_dashboard_dates_are_compact_and_readable(
    value: str | None,
    formatted: str,
) -> None:
    assert format_dashboard_date(value) == formatted


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
