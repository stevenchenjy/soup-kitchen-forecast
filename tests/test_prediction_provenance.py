from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import prediction_logs
from src.model_publication import sha256_file
from src.predictor import VisitorPredictor


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "models/visitor_model_ny_12550.joblib"
LEGACY_BACKUP = (
    ROOT / "models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib"
)
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def test_active_f6_prediction_exposes_locked_model_provenance() -> None:
    predictor = VisitorPredictor(str(ACTIVE))
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        output = predictor.predict_next("2026-07-19")

    assert output.package_id == predictor.package_id
    assert output.model_package_schema_version == 2
    assert output.feature_set_id == "F6_COMPACT_SELECTED"
    assert output.feature_order_sha256 == (
        "dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419"
    )
    assert output.recommendation_policy_id == "C0_EXISTING_RAW_QUANTILE"
    assert output.forecast_origin.date().isoformat() == "2026-07-17"
    assert output.calendar_days_ahead == 2
    assert output.service_horizon == 2
    assert output.model_segment == "sun"


def test_legacy_prediction_provenance_is_backward_compatible() -> None:
    predictor = VisitorPredictor(str(LEGACY_BACKUP))
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)), patch(
        "src.predictor.WeatherClient.fetch_forecast_daily",
        return_value=__import__("pandas").DataFrame(
            [{
                "date": "2026-07-18",
                "temp_10_13": 72.0,
                "apparent_temp_10_13": 73.0,
                "humidity_10_13": 55.0,
                "wind_10_13": 6.0,
                "precip_10_13": 0.0,
            }]
        ),
    ):
        output = predictor.predict_next("2026-07-18")

    assert output.model_package_schema_version == 1
    assert output.feature_set_id is None
    assert output.feature_order_sha256 is None
    assert output.recommendation_policy_id == "LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS"
    assert output.calendar_days_ahead == 1
    assert output.service_horizon == 1


def _legacy_schema_sql() -> str:
    return """
    CREATE TABLE prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location_id TEXT NOT NULL,
        service_date TEXT NOT NULL,
        prediction_created_at TEXT NOT NULL,
        predicted_visitors REAL NOT NULL,
        predicted_quantile REAL,
        residual_buffer REAL,
        suggested_meals INTEGER NOT NULL,
        meal_buffer_pct REAL,
        model_segment TEXT,
        actual_visitors INTEGER,
        absolute_error REAL,
        baseline_meals_prepared INTEGER,
        waste_avoided_meals REAL,
        estimated_co2e_reduction_kg REAL,
        created_by TEXT,
        source_app TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """


def test_sqlite_migration_adds_nullable_columns_without_rewriting_rows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "prediction.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(_legacy_schema_sql())
        conn.execute(
            "INSERT INTO prediction_logs "
            "(location_id, service_date, prediction_created_at, predicted_visitors, "
            "suggested_meals, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ny_12550", "2026-07-12", "before", 100.0, 110, "before", "before"),
        )
        conn.commit()

    with patch.object(prediction_logs, "location_db_file", return_value=db):
        with prediction_logs._connect("ny_12550") as conn:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(prediction_logs)")
            }
            row = conn.execute("SELECT * FROM prediction_logs").fetchone()

    assert set(prediction_logs.PROVENANCE_COLUMN_TYPES).issubset(columns)
    assert row["predicted_visitors"] == 100.0
    assert all(row[column] is None for column in prediction_logs.PROVENANCE_COLUMN_TYPES)


def test_prediction_log_persists_f6_provenance_locally(tmp_path: Path) -> None:
    db = tmp_path / "prediction.db"
    predictor = VisitorPredictor(str(ACTIVE))
    with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        output = predictor.predict_next("2026-07-18")

    with patch.object(prediction_logs, "location_db_file", return_value=db), patch.object(
        prediction_logs, "_supabase_config", return_value=None
    ), patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
        prediction_logs.save_prediction_log("ny_12550", output, source_app="test")
        rows = prediction_logs.load_prediction_logs("ny_12550")

    assert len(rows) == 1
    row = rows[0]
    assert row["package_id"] == predictor.package_id
    assert row["model_package_schema_version"] == 2
    assert row["feature_set_id"] == "F6_COMPACT_SELECTED"
    assert row["recommendation_policy_id"] == "C0_EXISTING_RAW_QUANTILE"
    assert row["forecast_origin"] == "2026-07-17"
    assert row["calendar_days_ahead"] == 1
    assert row["service_horizon"] == 1


def test_old_prediction_output_without_provenance_still_serializes() -> None:
    old_output = SimpleNamespace(
        service_date="2026-07-18",
        predicted_visitors=100,
        predicted_quantile=110,
        residual_buffer=5,
        suggested_meals=115,
        meal_buffer_pct=0.08,
        model_segment="sat",
    )
    row = prediction_logs._row_from_prediction(
        "ny_12550", old_output, None, "legacy-test", None
    )
    assert all(row[column] is None for column in prediction_logs.PROVENANCE_COLUMN_TYPES)


def test_pending_prediction_dates_use_prediction_log_actual_as_completion_source() -> None:
    pending = prediction_logs.pending_prediction_service_dates(
        [
            {"service_date": "2026-07-11", "actual_visitors": None},
            {"service_date": "2026-07-12", "actual_visitors": None},
            {"service_date": "2026-07-13", "actual_visitors": 115},
            {"service_date": "2026-07-17", "actual_visitors": None},
            {"service_date": "not-a-date", "actual_visitors": None},
        ],
        date(2026, 7, 17),
    )

    assert pending == [date(2026, 7, 12), date(2026, 7, 11)]


def test_supabase_missing_columns_retry_without_provenance() -> None:
    payload = {
        "location_id": "ny_12550",
        "package_id": "package-v1",
        "feature_set_id": "F6_COMPACT_SELECTED",
    }
    with patch.object(
        prediction_logs,
        "_supabase_request",
        side_effect=[
            RuntimeError("Could not find the 'package_id' column in the schema cache"),
            [{"id": 1}],
        ],
    ) as request:
        result = prediction_logs._supabase_write_with_provenance_fallback(
            "POST", params=None, payload=payload, extra_headers=None
        )

    assert result == [{"id": 1}]
    assert request.call_count == 2
    assert request.call_args_list[0].kwargs["payload"]["package_id"] == "package-v1"
    assert request.call_args_list[1].kwargs["payload"] == {"location_id": "ny_12550"}


def test_supabase_non_schema_errors_are_not_retried() -> None:
    with patch.object(
        prediction_logs,
        "_supabase_request",
        side_effect=RuntimeError("Supabase request failed (401): unauthorized"),
    ) as request:
        with pytest.raises(RuntimeError, match="401"):
            prediction_logs._supabase_write_with_provenance_fallback(
                "POST",
                params=None,
                payload={"package_id": "package-v1"},
                extra_headers=None,
            )
    assert request.call_count == 1


def test_supabase_migration_is_additive_and_not_automatically_executed() -> None:
    path = ROOT / "supabase/migrations/20260716_add_prediction_model_provenance.sql"
    sql = path.read_text(encoding="utf-8").casefold()
    assert "add column if not exists package_id" in sql
    assert "add column if not exists service_horizon" in sql
    assert "drop " not in sql
    assert "delete " not in sql
    assert "update " not in sql
    assert sha256_file(path)
