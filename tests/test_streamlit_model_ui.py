from __future__ import annotations

import builtins
import io
import os
import warnings
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.f6_readiness import prediction_signature
from src.location_config import Location
from src.predictor import VisitorPredictor
from src.production_features import LOCKED_F6_FEATURE_ORDER_SHA256


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL = ROOT / "models/visitor_model_ny_12550.joblib"
LEGACY_BACKUP = (
    ROOT / "models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib"
)
LOCATION = Location(
    id="ny_12550",
    name="Newburgh, NY 12550",
    zip_code="12550",
)
ATTENDANCE = pd.DataFrame(
    [{"service_date": pd.Timestamp("2026-07-12"), "visitors": 100}]
)
SECRET_SENTINEL = "stage34-supabase-secret-must-not-render"
URL_SENTINEL = "https://stage34-ui-secret-url.invalid"
SECRET_SENTINELS = (SECRET_SENTINEL, URL_SENTINEL)
PACKAGE_ID = "ny_12550_f6_2026-07-12_v1"
FEATURE_SET_ID = "F6_COMPACT_SELECTED"
POLICY_ID = "C0_EXISTING_RAW_QUANTILE"
BLOCKED_SOURCE_ROOTS = (
    (ROOT / "data/updated").resolve(),
    (ROOT / "data/locations/ny_12550/Updated").resolve(),
)
ORIGINAL_BUILTIN_OPEN = builtins.open
ORIGINAL_IO_OPEN = io.open
ORIGINAL_OS_OPEN = os.open


class FakeWeatherClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def fetch_forecast_daily(self, target_date: date) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(target_date),
                    "temp_10_13": 72.0,
                    "apparent_temp_10_13": 73.0,
                    "humidity_10_13": 55.0,
                    "wind_10_13": 6.0,
                    "precip_10_13": 0.0,
                }
            ]
        )


def mutation_tripwire(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("Streamlit verification attempted a production mutation")


def weather_tripwire(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("F6/W0 prediction attempted to create a weather client")


def guarded_source_open(original):
    def guarded(file, *args: Any, **kwargs: Any):
        try:
            path = Path(file).resolve()
        except (TypeError, OSError):
            return original(file, *args, **kwargs)
        if any(path == root or root in path.parents for root in BLOCKED_SOURCE_ROOTS):
            raise AssertionError(f"UI attempted to read local attendance source: {path}")
        return original(file, *args, **kwargs)

    return guarded


def _user(role: str) -> dict[str, Any]:
    username = "admin-test" if role == "master" else "staff-test"
    return {
        "username": username,
        "role": role,
        "authorized_locations": ["*"] if role == "master" else ["ny_12550"],
        "password_hash": "not-a-real-hash",
        "salt": "not-a-real-salt",
        "iterations": 1,
    }


def _active_f6_row(
    *,
    row_id: int,
    service_date: str,
    predicted: float,
    quantile: float,
    suggested: int,
    actual: int | None,
    segment: str,
    horizon: int,
    package_id: str = PACKAGE_ID,
    schema_version: int = 2,
    feature_set_id: str | None = FEATURE_SET_ID,
    policy_id: str = POLICY_ID,
    waste: float = 10.0,
    co2e: float = 17.1,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "location_id": "ny_12550",
        "service_date": service_date,
        "prediction_created_at": f"{service_date}T12:00:00+00:00",
        "predicted_visitors": predicted,
        "predicted_quantile": quantile,
        "suggested_meals": suggested,
        "model_segment": segment,
        "package_id": package_id,
        "model_package_schema_version": schema_version,
        "feature_set_id": feature_set_id,
        "feature_order_sha256": (
            LOCKED_F6_FEATURE_ORDER_SHA256 if schema_version == 2 else None
        ),
        "recommendation_policy_id": policy_id,
        "service_horizon": horizon,
        "actual_visitors": actual,
        "waste_avoided_meals": waste,
        "estimated_co2e_reduction_kg": co2e,
    }


def _mixed_history_rows() -> list[dict[str, Any]]:
    rows = [
        _active_f6_row(
            row_id=index,
            service_date=f"2026-05-{index:02d}",
            predicted=999.0,
            quantile=1000.0,
            suggested=1000,
            actual=1,
            segment="sat",
            horizon=1,
            package_id="visitor_model_ny_12550.joblib",
            schema_version=1,
            feature_set_id=None,
            policy_id="OLDER_POLICY",
            waste=500.0,
            co2e=855.0,
        )
        for index in range(1, 11)
    ]
    rows.extend(
        [
            _active_f6_row(
                row_id=11,
                service_date="2026-07-18",
                predicted=100.0,
                quantile=115.0,
                suggested=115,
                actual=110,
                segment="sat",
                horizon=1,
                waste=10.0,
                co2e=17.1,
            ),
            _active_f6_row(
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
            _active_f6_row(
                row_id=13,
                service_date="2026-07-25",
                predicted=90.0,
                quantile=105.0,
                suggested=105,
                actual=80,
                segment="sat",
                horizon=1,
                package_id="ny_12550_f6_inactive_v1",
                waste=100.0,
                co2e=171.0,
            ),
            _active_f6_row(
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


def _element_text(app: AppTest) -> str:
    values: list[str] = []
    for collection_name in (
        "caption",
        "error",
        "exception",
        "header",
        "info",
        "markdown",
        "metric",
        "subheader",
        "success",
        "text",
        "title",
        "warning",
    ):
        collection = getattr(app, collection_name, ())
        for element in collection:
            for attribute in ("label", "value", "body"):
                value = getattr(element, attribute, None)
                if value is not None:
                    values.append(str(value))
    return "\n".join(values)


def _configure_common_patches(
    stack: ExitStack,
    *,
    user: dict[str, Any],
    model_path: Path,
    artifact_dir: Path,
    saved_logs: list[tuple[Any, ...]],
    f6: bool,
    prediction_rows: list[dict[str, Any]] | None = None,
    retrain_state: dict[str, Any] | None = None,
    latest_run: dict[str, Any] | None = None,
    latest_successful_run: dict[str, Any] | None = None,
) -> None:
    users = [user]
    stack.enter_context(
        patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": URL_SENTINEL,
                "SUPABASE_SERVICE_ROLE_KEY": SECRET_SENTINEL,
                "SUPABASE_ANON_KEY": SECRET_SENTINEL,
                "service_role_key": SECRET_SENTINEL,
                "anon_key": SECRET_SENTINEL,
                "key": SECRET_SENTINEL,
                "url": URL_SENTINEL,
                "LOKY_MAX_CPU_COUNT": "1",
            },
        )
    )
    stack.enter_context(
        patch("builtins.open", new=guarded_source_open(ORIGINAL_BUILTIN_OPEN))
    )
    stack.enter_context(patch("io.open", new=guarded_source_open(ORIGINAL_IO_OPEN)))
    stack.enter_context(patch("os.open", new=guarded_source_open(ORIGINAL_OS_OPEN)))
    stack.enter_context(patch("src.auth.get_user", return_value=user))
    stack.enter_context(patch("src.auth.load_users", return_value=users))
    stack.enter_context(patch("src.auth.user_store_mode", return_value="supabase"))
    stack.enter_context(
        patch(
            "src.auth.validate_user_record",
            return_value={"passed": True, "reason": "User record is valid."},
        )
    )
    stack.enter_context(patch("src.auth.save_users", side_effect=mutation_tripwire))
    stack.enter_context(patch("src.auth.delete_user", side_effect=mutation_tripwire))

    stack.enter_context(patch("src.location_config.list_locations", return_value=[LOCATION]))
    stack.enter_context(
        patch("src.location_config.save_locations", side_effect=mutation_tripwire)
    )
    stack.enter_context(
        patch("src.config.model_file_for_location", return_value=model_path)
    )
    stack.enter_context(
        patch("src.config.artifact_dir_for_location", return_value=artifact_dir)
    )
    stack.enter_context(
        patch("src.config.forecast_today", return_value=date(2026, 7, 17))
    )

    stack.enter_context(
        patch("src.data_admin.load_clean_data", return_value=ATTENDANCE.copy())
    )
    stack.enter_context(
        patch("src.data_admin.load_recent_attendance", return_value=ATTENDANCE.copy())
    )
    stack.enter_context(
        patch("src.data_admin.latest_staff_created_attendance", return_value=None)
    )
    for target in (
        "src.data_admin.upsert_record",
        "src.data_admin.save_clean_data",
        "src.data_admin.delete_record",
        "src.data_admin.delete_latest_staff_created_attendance",
    ):
        stack.enter_context(patch(target, side_effect=mutation_tripwire))
    stack.enter_context(
        patch("src.data_admin.attendance_store_mode", return_value="supabase")
    )

    stack.enter_context(
        patch(
            "src.prediction_logs.save_prediction_log",
            side_effect=lambda *args, **kwargs: saved_logs.append(args),
        )
    )
    stack.enter_context(
        patch(
            "src.prediction_logs.update_prediction_logs_with_actual",
            side_effect=mutation_tripwire,
        )
    )
    stack.enter_context(
        patch(
            "src.prediction_logs.cleanup_logs_without_attendance",
            side_effect=mutation_tripwire,
        )
    )
    stack.enter_context(
        patch("src.prediction_logs.prediction_log_store_mode", return_value="supabase")
    )
    stack.enter_context(
        patch("src.prediction_logs.summarize_monitoring", return_value={})
    )
    stack.enter_context(
        patch(
            "src.prediction_logs.load_prediction_logs",
            return_value=[] if prediction_rows is None else prediction_rows,
        )
    )

    stack.enter_context(
        patch(
            "src.model_training_runs.get_retrain_state",
            return_value={"dirty": False} if retrain_state is None else retrain_state,
        )
    )
    stack.enter_context(
        patch("src.model_training_runs.latest_training_run", return_value=latest_run)
    )
    stack.enter_context(
        patch(
            "src.model_training_runs.latest_successful_training_run",
            return_value=latest_successful_run,
        )
    )
    stack.enter_context(
        patch(
            "src.model_training_runs.model_training_run_store_mode",
            return_value="supabase",
        )
    )
    stack.enter_context(
        patch(
            "src.predictor.WeatherClient",
            new=weather_tripwire if f6 else FakeWeatherClient,
        )
    )


def _run_prediction_ui(
    *,
    app_filename: str,
    role: str,
    model_path: Path,
    target_date: str,
    f6: bool,
) -> dict[str, Any]:
    saved_logs: list[tuple[Any, ...]] = []
    with (
        TemporaryDirectory() as artifact_directory,
        ExitStack() as stack,
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        user = _user(role)
        _configure_common_patches(
            stack,
            user=user,
            model_path=model_path,
            artifact_dir=Path(artifact_directory),
            saved_logs=saved_logs,
            f6=f6,
        )

        direct_predictor = VisitorPredictor(str(model_path))
        direct = direct_predictor.predict_next(target_date, meal_buffer_pct=None)

        app = AppTest.from_file(str(ROOT / app_filename), default_timeout=60)
        app.session_state["user"] = {"username": user["username"]}
        app.run()
        assert not app.exception

        target_input = next(
            element
            for element in app.text_input
            if element.label.startswith("Target service date")
        )
        target_input.set_value(target_date)
        button_label = (
            "Generate prediction"
            if app_filename == "app.py"
            else "Get meal recommendation"
        )
        next(button for button in app.button if button.label == button_label).click()
        app.run()
        assert not app.exception
        assert not app.error
        assert len(saved_logs) == 1
        logged_prediction = saved_logs[0][1]
        assert prediction_signature(logged_prediction) == prediction_signature(direct)
        recommendation_metric = next(
            element
            for element in app.metric
            if element.label == "Raw Q80 recommended meals"
        )
        assert str(recommendation_metric.value) == str(direct.suggested_meals)

        rendered_text = _element_text(app)
        assert all(sentinel not in rendered_text for sentinel in SECRET_SENTINELS)
        return {
            "rendered_text": rendered_text,
            "slider_labels": [element.label for element in app.slider],
            "metric_labels": [element.label for element in app.metric],
            "prediction": prediction_signature(logged_prediction),
        }


def _run_admin_monitoring_ui(
    prediction_rows: list[dict[str, Any]],
    *,
    latest_run: dict[str, Any] | None = None,
    latest_successful_run: dict[str, Any] | None = None,
) -> AppTest:
    saved_logs: list[tuple[Any, ...]] = []
    with TemporaryDirectory() as artifact_directory, ExitStack() as stack:
        _configure_common_patches(
            stack,
            user=_user("master"),
            model_path=ACTIVE_MODEL,
            artifact_dir=Path(artifact_directory),
            saved_logs=saved_logs,
            f6=True,
            prediction_rows=prediction_rows,
            latest_run=latest_run,
            latest_successful_run=latest_successful_run,
        )
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.session_state["user"] = {"username": "admin-test"}
        app.run()

    assert not app.exception
    assert not app.error
    assert saved_logs == []
    return app


def _metric_values(app: AppTest) -> dict[str, str]:
    return {str(element.label): str(element.value) for element in app.metric}


def _rendered_dataframe(app: AppTest, required_columns: set[str]) -> pd.DataFrame:
    for element in app.dataframe:
        value = element.value
        if isinstance(value, pd.DataFrame) and required_columns.issubset(value.columns):
            return value
    raise AssertionError(f"No rendered dataframe has columns {sorted(required_columns)}")


@pytest.mark.parametrize("app_filename,role", [("app.py", "master"), ("app_staff.py", "staff")])
@pytest.mark.parametrize("target_date,segment", [("2026-07-18", "sat"), ("2026-07-19", "sun")])
def test_active_f6_admin_and_staff_prediction_paths(
    app_filename: str,
    role: str,
    target_date: str,
    segment: str,
) -> None:
    result = _run_prediction_ui(
        app_filename=app_filename,
        role=role,
        model_path=ACTIVE_MODEL,
        target_date=target_date,
        f6=True,
    )
    text = result["rendered_text"]
    assert "ny_12550_f6_2026-07-12_v1" in text
    assert "schema v2" in text
    assert "F6_COMPACT_SELECTED" in text
    assert "Recommendation policy: C0 raw Q80" in text
    assert all("buffer (%)" not in label.lower() for label in result["slider_labels"])
    assert result["prediction"]["model_segment"] == segment
    assert result["prediction"]["meal_buffer_pct"] == 0.0
    assert result["prediction"]["residual_buffer"] == 0.0
    assert "Raw Q80 recommended meals" in result["metric_labels"]
    forbidden = (
        "legacy",
        "fallback",
        "rollback",
        "schema v1",
        "percentage buffer",
        "residual buffer",
    )
    assert all(term not in text.lower() for term in forbidden)
    if app_filename == "app_staff.py":
        assert "waste avoided" not in text.lower()
        assert "co2e" not in text.lower()


def test_admin_monitoring_uses_only_active_f6_rows_from_mixed_history() -> None:
    old_run = {
        "status": "success",
        "finished_at": "2026-07-13T10:21:30+00:00",
        "attendance_rows": 360,
        "metrics": {"overall": {"MAE": 999.0}},
    }
    app = _run_admin_monitoring_ui(
        _mixed_history_rows(),
        latest_run=old_run,
        latest_successful_run=old_run,
    )
    metrics = _metric_values(app)
    assert metrics["F6 predictions"] == "3"
    assert metrics["F6 reconciled predictions"] == "2"
    assert metrics["F6 unreconciled predictions"] == "1"
    assert metrics["F6 review stage"] == "INSUFFICIENT_DATA"
    assert metrics["F6 MAE"] == "15.00"
    assert metrics["F6 median absolute error"] == "15.00"
    assert metrics["F6 RMSE"] == "15.81"
    assert metrics["F6 mean signed error"] == "5.00"
    assert metrics["F6 underprediction rate"] == "50.0%"
    assert metrics["F6 Q80 empirical coverage"] == "100.0%"
    assert metrics["F6 Saturday MAE"] == "10.00"
    assert metrics["F6 Sunday MAE"] == "20.00"
    assert metrics["F6 estimated waste avoided"] == "35.0 meals"
    assert metrics["F6 estimated CO2e reduction"] == "59.9 kg"

    text = _element_text(app)
    assert "Activated from verified candidate; first genuine F6 retraining pending" in text
    assert "2026-07-13T10:21:30+00:00" not in text
    assert "999.0" not in text
    assert "Live MAE" not in metrics
    assert "MAPE" not in metrics
    assert "P90AbsError" not in metrics

    outcomes = _rendered_dataframe(
        app,
        {"Service date", "Expected visitors", "Actual visitors", "Absolute error"},
    )
    assert outcomes["Service date"].tolist() == [
        "2026-07-26",
        "2026-07-19",
        "2026-07-18",
    ]
    assert outcomes["Expected visitors"].tolist() == [120.0, 150.0, 100.0]


def test_admin_monitoring_zero_f6_rows_shows_insufficient_data_without_fallback() -> None:
    app = _run_admin_monitoring_ui(_mixed_history_rows()[:10])
    metrics = _metric_values(app)
    assert metrics["F6 predictions"] == "0"
    assert metrics["F6 reconciled predictions"] == "0"
    assert metrics["F6 unreconciled predictions"] == "0"
    assert metrics["F6 review stage"] == "INSUFFICIENT_DATA"
    for label in (
        "F6 MAE",
        "F6 median absolute error",
        "F6 RMSE",
        "F6 mean signed error",
        "F6 underprediction rate",
        "F6 Q80 empirical coverage",
        "F6 mean over-preparation",
        "F6 mean under-preparation",
        "F6 Saturday MAE",
        "F6 Sunday MAE",
    ):
        assert metrics[label] == "—"
    text = _element_text(app)
    assert "F6 performance: Insufficient data" in text
    assert "No active-package F6 predictions are available yet." in text
    assert "fallback" not in text.lower()


@pytest.mark.parametrize("app_filename,role", [("app.py", "master"), ("app_staff.py", "staff")])
def test_non_f6_active_package_suppresses_production_prediction_ui(
    app_filename: str,
    role: str,
) -> None:
    saved_logs: list[tuple[Any, ...]] = []
    with TemporaryDirectory() as artifact_directory, ExitStack() as stack:
        _configure_common_patches(
            stack,
            user=_user(role),
            model_path=LEGACY_BACKUP,
            artifact_dir=Path(artifact_directory),
            saved_logs=saved_logs,
            f6=False,
        )
        app = AppTest.from_file(str(ROOT / app_filename), default_timeout=60)
        app.session_state["user"] = {
            "username": "admin-test" if role == "master" else "staff-test"
        }
        app.run()

    assert not app.exception
    text = _element_text(app)
    assert "F6 integrity error" in text
    assert not [
        item
        for item in app.text_input
        if item.label.startswith("Target service date")
    ]
    assert not [
        button
        for button in app.button
        if button.label in {"Generate prediction", "Get meal recommendation"}
    ]
    assert "F6 MAE" not in _metric_values(app)
    assert all(
        term not in text.lower()
        for term in ("legacy", "fallback", "rollback", "schema v1")
    )
    assert saved_logs == []
