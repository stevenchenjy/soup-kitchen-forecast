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


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PACKAGE = (
    ROOT
    / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
)
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
        patch("src.prediction_logs.load_prediction_logs", return_value=[])
    )

    stack.enter_context(
        patch("src.model_training_runs.get_retrain_state", return_value={"dirty": False})
    )
    stack.enter_context(
        patch("src.model_training_runs.latest_training_run", return_value=None)
    )
    stack.enter_context(
        patch("src.model_training_runs.latest_successful_training_run", return_value=None)
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
        direct = direct_predictor.predict_next(
            target_date,
            meal_buffer_pct=None if f6 else 0.30,
        )

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
        if not f6:
            buffer_slider = next(
                element
                for element in app.slider
                if "buffer (%)" in element.label.lower()
            )
            buffer_slider.set_value(30)
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
        recommendation_label = (
            "Raw 80th-percentile recommendation" if f6 else "Recommended Meals"
        )
        if app_filename == "app.py":
            prediction_message = next(
                element.value
                for element in app.success
                if str(element.value).startswith("Location: ny_12550")
            )
            assert (
                f"{recommendation_label}: {direct.suggested_meals}"
                in prediction_message
            )
        else:
            recommendation_metric = next(
                element
                for element in app.metric
                if element.label == recommendation_label
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
    assert (
        "F6/C0 uses the raw 80th-percentile recommendation without a percentage buffer."
        in text
    )
    assert all("buffer (%)" not in label.lower() for label in result["slider_labels"])
    assert result["prediction"]["model_segment"] == segment
    assert result["prediction"]["meal_buffer_pct"] == 0.0
    assert result["prediction"]["residual_buffer"] == 0.0
    if app_filename == "app_staff.py":
        assert "Raw 80th-percentile recommendation" in result["metric_labels"]
    else:
        assert "Raw 80th-percentile recommendation" in text


@pytest.mark.parametrize("app_filename,role", [("app.py", "master"), ("app_staff.py", "staff")])
def test_legacy_admin_and_staff_prediction_paths_remain_compatible(
    app_filename: str,
    role: str,
) -> None:
    result = _run_prediction_ui(
        app_filename=app_filename,
        role=role,
        model_path=LEGACY_BACKUP,
        target_date="2026-07-18",
        f6=False,
    )
    text = result["rendered_text"]
    assert "schema v1" in text
    assert "F6_COMPACT_SELECTED" not in text
    assert "raw 80th-percentile recommendation" not in text.lower()
    assert any("buffer (%)" in label.lower() for label in result["slider_labels"])
    assert result["prediction"]["model_segment"] == "sat"
    assert result["prediction"]["meal_buffer_pct"] == 0.30
