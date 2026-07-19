"""Active-package-only F6 monitoring calculations for production dashboards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

import pandas as pd

from src.config import PROJECT_ROOT
from src.feature_sets import F6
from src.production_features import (
    LOCKED_F6_FEATURE_ORDER_SHA256,
    MODEL_PACKAGE_SCHEMA_VERSION,
    RECOMMENDATION_POLICY_ID,
)


TRAINING_PROVENANCE_KEY = "_f6_package_provenance"
VERIFIED_BACKTEST_DIR = PROJECT_ROOT / "config" / "model_backtests"
_BACKTEST_METRIC_KEYS = {
    "mae",
    "median_absolute_error",
    "rmse",
    "mean_signed_error",
    "underprediction_rate",
    "q80_empirical_coverage",
    "mean_over_preparation",
    "mean_under_preparation",
}


@dataclass(frozen=True)
class ActiveF6Package:
    package_id: str
    schema_version: int
    feature_set_id: str
    feature_count: int
    feature_order_sha256: str
    recommendation_policy_id: str


class F6IntegrityError(ValueError):
    """Raised when the active model does not satisfy the locked F6 contract."""


class BacktestSummaryError(ValueError):
    """Raised when verified historical performance cannot be authenticated."""


class BacktestChartError(ValueError):
    """Raised when a tracked historical chart series cannot be authenticated."""


def active_f6_package(predictor: Any) -> ActiveF6Package:
    """Derive and validate the dashboard contract from the active model package."""

    if predictor is None:
        raise F6IntegrityError("The active F6 model could not be loaded.")

    schema_version = int(getattr(predictor, "model_package_schema_version", 0))
    feature_contract = getattr(predictor, "feature_contract", None)
    if schema_version != MODEL_PACKAGE_SCHEMA_VERSION:
        raise F6IntegrityError("The active model is not schema version 2.")
    if not bool(getattr(predictor, "uses_locked_f6", False)):
        raise F6IntegrityError("The active model is not a locked F6 package.")
    if not isinstance(feature_contract, Mapping):
        raise F6IntegrityError("The active F6 feature contract is unavailable.")

    feature_set_id = str(feature_contract.get("feature_set_id") or "")
    feature_hash = str(feature_contract.get("feature_order_sha256") or "")
    recommendation_policy_id = str(
        getattr(predictor, "recommendation_policy_id", "") or ""
    )
    feature_count = len(getattr(predictor, "feature_cols", ()))
    package_id = str(getattr(predictor, "package_id", "") or "")

    if not package_id:
        raise F6IntegrityError("The active F6 package ID is unavailable.")
    if feature_set_id != F6:
        raise F6IntegrityError("The active feature set is not F6_COMPACT_SELECTED.")
    if feature_count != 33:
        raise F6IntegrityError("The active F6 package does not contain 33 features.")
    if feature_hash != LOCKED_F6_FEATURE_ORDER_SHA256:
        raise F6IntegrityError("The active F6 feature hash does not match the lock.")
    if recommendation_policy_id != RECOMMENDATION_POLICY_ID:
        raise F6IntegrityError("The active F6 recommendation policy is not locked C0.")

    return ActiveF6Package(
        package_id=package_id,
        schema_version=schema_version,
        feature_set_id=feature_set_id,
        feature_count=feature_count,
        feature_order_sha256=feature_hash,
        recommendation_policy_id=recommendation_policy_id,
    )


def _schema_matches(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def row_matches_active_f6(
    row: Mapping[str, Any], contract: ActiveF6Package
) -> bool:
    """Return whether all available row provenance agrees with the active package."""

    if str(row.get("package_id") or "") != contract.package_id:
        return False
    if not _schema_matches(
        row.get("model_package_schema_version"), contract.schema_version
    ):
        return False
    if str(row.get("feature_set_id") or "") != contract.feature_set_id:
        return False
    if (
        str(row.get("recommendation_policy_id") or "")
        != contract.recommendation_policy_id
    ):
        return False

    for hash_field in ("feature_order_sha256", "feature_hash"):
        row_hash = row.get(hash_field)
        if (
            row_hash not in (None, "")
            and str(row_hash) != contract.feature_order_sha256
        ):
            return False
    return True


def filter_active_f6_rows(
    rows: Iterable[Mapping[str, Any]], contract: ActiveF6Package
) -> list[dict[str, Any]]:
    """Copy only rows that match the currently active F6 package contract."""

    return [dict(row) for row in rows if row_matches_active_f6(row, contract)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_source_path(value: Any) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise BacktestSummaryError("Verified backtest source path is invalid.")
    source = (PROJECT_ROOT / relative).resolve()
    if PROJECT_ROOT.resolve() not in source.parents:
        raise BacktestSummaryError("Verified backtest source is outside the project.")
    return source


def _registered_chart_path(value: Any) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise BacktestChartError("Historical chart dataset path is invalid.")
    chart_path = (PROJECT_ROOT / relative).resolve()
    if PROJECT_ROOT.resolve() not in chart_path.parents:
        raise BacktestChartError("Historical chart dataset is outside the project.")
    return chart_path


def load_verified_backtest_summary(
    contract: ActiveF6Package,
    *,
    summary_path: str | Path | None = None,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Load a package-bound historical summary, optionally checking source files.

    The tracked summary is the production runtime artifact. Its source-artifact
    records are provenance, while source existence and hashes are an opt-in
    generation/CI check because ignored research artifacts are not deployed.
    """

    if summary_path is None:
        if Path(contract.package_id).name != contract.package_id:
            raise BacktestSummaryError("Active package ID cannot identify a backtest summary.")
        path = VERIFIED_BACKTEST_DIR / f"{contract.package_id}.json"
    else:
        path = Path(summary_path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BacktestSummaryError(
            f"No verified historical backtest is registered for {contract.package_id}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BacktestSummaryError("Verified historical backtest could not be loaded.") from exc
    if not isinstance(summary, dict):
        raise BacktestSummaryError("Verified historical backtest must be a mapping.")
    if summary.get("summary_schema_version") != 1:
        raise BacktestSummaryError("Historical backtest summary schema is unsupported.")

    expected_contract = {
        "package_id": contract.package_id,
        "model_package_schema_version": contract.schema_version,
        "feature_set_id": contract.feature_set_id,
        "feature_order_sha256": contract.feature_order_sha256,
        "recommendation_policy_id": contract.recommendation_policy_id,
    }
    if summary.get("verification_status") != "verified":
        raise BacktestSummaryError("Historical backtest is not marked verified.")
    for field, expected in expected_contract.items():
        if summary.get(field) != expected:
            raise BacktestSummaryError(
                f"Historical backtest {field} does not match the active package."
            )

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or not _BACKTEST_METRIC_KEYS.issubset(metrics):
        raise BacktestSummaryError("Historical backtest metrics are incomplete.")
    if any(_finite_number(metrics.get(key)) is None for key in _BACKTEST_METRIC_KEYS):
        raise BacktestSummaryError("Historical backtest metrics contain invalid values.")
    try:
        date.fromisoformat(str(summary.get("attendance_cutoff") or ""))
    except ValueError as exc:
        raise BacktestSummaryError("Historical backtest attendance cutoff is invalid.") from exc
    segments = summary.get("segments")
    if not isinstance(segments, Mapping) or any(
        _finite_number((segments.get(label) or {}).get("mae")) is None
        for label in ("Saturday", "Sunday")
    ):
        raise BacktestSummaryError("Historical segment metrics are incomplete.")
    horizons = summary.get("horizons")
    if not isinstance(horizons, Mapping) or any(
        _finite_number((horizons.get(label) or {}).get("mae")) is None
        for label in ("H1", "H2", "H5")
    ):
        raise BacktestSummaryError("Historical horizon metrics are incomplete.")

    if verify_sources:
        sources = summary.get("source_artifacts")
        if not isinstance(sources, list) or not sources:
            raise BacktestSummaryError("Historical backtest source provenance is missing.")
        for source_record in sources:
            if not isinstance(source_record, Mapping):
                raise BacktestSummaryError("Historical backtest source record is invalid.")
            source = _verified_source_path(source_record.get("path"))
            if not source.is_file():
                raise BacktestSummaryError(f"Verified backtest source is missing: {source.name}")
            if _sha256_file(source) != str(source_record.get("sha256") or ""):
                raise BacktestSummaryError(
                    f"Verified backtest source hash does not match: {source.name}"
                )
    return summary


def load_verified_backtest_chart_series(
    contract: ActiveF6Package,
    *,
    summary_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and authenticate the tracked chart series for the active package."""

    try:
        summary = load_verified_backtest_summary(
            contract,
            summary_path=summary_path,
        )
    except BacktestSummaryError as exc:
        raise BacktestChartError(
            "Historical chart package binding could not be verified."
        ) from exc

    registration = summary.get("chart_dataset")
    if not isinstance(registration, Mapping):
        raise BacktestChartError("Historical chart dataset is not registered.")
    if registration.get("scope_type") != "scenario":
        raise BacktestChartError("Historical chart scope registration is invalid.")
    scope_value = str(registration.get("scope_value") or "")
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, Mapping) or scope_value != str(
        evaluation.get("primary_scope") or ""
    ):
        raise BacktestChartError("Historical chart scope does not match the summary.")

    chart_path = _registered_chart_path(registration.get("path"))
    if not chart_path.is_file():
        raise BacktestChartError("Tracked historical chart dataset is missing.")
    if _sha256_file(chart_path) != str(registration.get("sha256") or ""):
        raise BacktestChartError("Historical chart dataset hash does not match.")

    try:
        frame = pd.read_csv(chart_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise BacktestChartError("Historical chart dataset could not be loaded.") from exc

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
    if not required_columns.issubset(frame.columns):
        raise BacktestChartError("Historical chart dataset columns are incomplete.")

    try:
        registered_rows = int(registration.get("row_count"))
        evaluation_rows = int(evaluation.get("row_count"))
    except (TypeError, ValueError) as exc:
        raise BacktestChartError("Historical chart row-count registration is invalid.") from exc
    if registered_rows != evaluation_rows:
        raise BacktestChartError("Historical chart row count does not match the summary.")
    if len(frame) != registered_rows:
        raise BacktestChartError("Historical chart row count does not match.")

    parsed_dates = pd.to_datetime(
        frame["service_date"], format="%Y-%m-%d", errors="coerce"
    )
    if parsed_dates.isna().any():
        raise BacktestChartError("Historical chart service dates are invalid.")
    frame = frame.copy()
    frame["service_date"] = parsed_dates

    numeric_columns = (
        "actual_visitors",
        "point_prediction",
        "q80_prediction",
        "absolute_error",
        "service_horizon",
    )
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not converted.map(math.isfinite).all():
            raise BacktestChartError("Historical chart numeric values are invalid.")
        frame[column] = converted

    expected_error = (frame["point_prediction"] - frame["actual_visitors"]).abs()
    if not (frame["absolute_error"] - expected_error).abs().le(1e-9).all():
        raise BacktestChartError("Historical chart absolute errors are inconsistent.")

    if not frame["model_segment"].isin({"sat", "sun"}).all():
        raise BacktestChartError("Historical chart segment values are invalid.")
    if not frame["scenario"].eq(scope_value).all():
        raise BacktestChartError("Historical chart rows do not match the registered scope.")

    try:
        registered_minimum = pd.Timestamp(
            date.fromisoformat(str(registration.get("minimum_service_date") or ""))
        )
        registered_maximum = pd.Timestamp(
            date.fromisoformat(str(registration.get("maximum_service_date") or ""))
        )
        attendance_cutoff = pd.Timestamp(
            date.fromisoformat(str(summary.get("attendance_cutoff") or ""))
        )
    except ValueError as exc:
        raise BacktestChartError("Historical chart date registration is invalid.") from exc
    if (
        frame["service_date"].min() != registered_minimum
        or frame["service_date"].max() != registered_maximum
    ):
        raise BacktestChartError("Historical chart date range does not match.")
    if frame["service_date"].max() > attendance_cutoff:
        raise BacktestChartError("Historical chart exceeds the attendance cutoff.")

    return frame.sort_values("service_date", kind="stable").reset_index(drop=True)


def format_dashboard_date(value: Any) -> str:
    """Format a date or timestamp compactly for dashboard metric cards."""

    raw = str(value or "").strip()
    if not raw:
        return "—"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            return "—"
    return f"{parsed.strftime('%b')} {parsed.day}, {parsed.year}"


def retraining_status_label(value: Any) -> str:
    """Map internal retraining states to short administrator-facing labels."""

    return {
        "PENDING_FIRST_F6_RETRAIN": "Pending first retrain",
        "RETRAINING_REQUIRED": "Retraining required",
        "SUCCESS": "Up to date",
        "FAILED": "Last retrain failed",
    }.get(str(value or "").upper(), "Status unavailable")


def build_operational_impact(
    attendance_row_count: int | None,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize cumulative operational records without package filtering."""

    materialized = [dict(row) for row in rows]
    waste = [
        value
        for row in materialized
        if (value := _finite_number(row.get("waste_avoided_meals"))) is not None
    ]
    co2e = [
        value
        for row in materialized
        if (value := _finite_number(row.get("estimated_co2e_reduction_kg")))
        is not None
    ]
    return {
        "attendance_row_count": (
            None if attendance_row_count is None else max(int(attendance_row_count), 0)
        ),
        "prediction_log_count": len(materialized),
        "reconciled_log_count": sum(
            1 for row in materialized if _finite_number(row.get("actual_visitors")) is not None
        ),
        "estimated_food_saved_meals": sum(waste),
        "estimated_co2e_reduction_kg": sum(co2e),
    }


def monitoring_stage(reconciled_count: int) -> str:
    if reconciled_count <= 3:
        return "INSUFFICIENT_DATA"
    if reconciled_count <= 7:
        return "EARLY_SIGNAL"
    if reconciled_count <= 11:
        return "INITIAL_REVIEW"
    return "STABLE_REVIEW"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reconciled_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reconciled: list[dict[str, Any]] = []
    for row in rows:
        actual = _finite_number(row.get("actual_visitors"))
        predicted = _finite_number(row.get("predicted_visitors"))
        if actual is None or predicted is None:
            continue
        item = dict(row)
        item["_actual"] = actual
        item["_predicted"] = predicted
        item["_absolute_error"] = abs(predicted - actual)
        reconciled.append(item)
    return reconciled


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _mae(rows: Iterable[Mapping[str, Any]]) -> float | None:
    return _mean_or_none(float(row["_absolute_error"]) for row in rows)


def _quantile_coverage(rows: Iterable[Mapping[str, Any]]) -> float | None:
    covered: list[float] = []
    for row in rows:
        quantile = _finite_number(row.get("predicted_quantile"))
        if quantile is None:
            continue
        covered.append(1.0 if float(row["_actual"]) <= quantile else 0.0)
    return _mean_or_none(covered)


def _segment_rows(
    rows: Iterable[Mapping[str, Any]], segment: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("model_segment") or "") == segment]


def _horizon_rows(
    rows: Iterable[Mapping[str, Any]], horizon: int
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            value = int(row.get("service_horizon"))
        except (TypeError, ValueError):
            continue
        if value == horizon:
            selected.append(row)
    return selected


def _latest_rows(rows: Iterable[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("service_date") or ""),
            str(row.get("prediction_created_at") or ""),
            int(row.get("id") or 0),
        ),
        reverse=True,
    )
    return ordered[:limit]


def _empty_report(message: str) -> dict[str, Any]:
    return {
        "integrity_ok": False,
        "integrity_errors": [message],
        "contract": None,
        "prediction_count": None,
        "reconciled_count": None,
        "unreconciled_count": None,
        "stage": None,
        "metrics": None,
        "segments": None,
        "horizons": None,
        "latest_outcomes": [],
        "sustainability": None,
        "integrity_alerts": [],
        "performance_alerts": [],
    }


def build_f6_monitoring_report(
    predictor: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    load_error: Exception | None = None,
    latest_limit: int = 20,
) -> dict[str, Any]:
    """Build metrics and outcomes exclusively from the active F6 package rows."""

    if load_error is not None:
        return _empty_report("The active F6 model could not be loaded.")
    try:
        contract = active_f6_package(predictor)
    except (F6IntegrityError, TypeError, ValueError) as exc:
        return _empty_report(str(exc))

    active_rows = filter_active_f6_rows(rows, contract)
    reconciled = _reconciled_rows(active_rows)
    reconciled_count = len(reconciled)
    prediction_count = len(active_rows)
    unreconciled_count = prediction_count - reconciled_count
    stage = monitoring_stage(reconciled_count)

    absolute_errors = [float(row["_absolute_error"]) for row in reconciled]
    signed_errors = [
        float(row["_predicted"]) - float(row["_actual"]) for row in reconciled
    ]
    underprediction = [
        1.0 if float(row["_predicted"]) < float(row["_actual"]) else 0.0
        for row in reconciled
    ]
    over_preparation: list[float] = []
    under_preparation: list[float] = []
    for row in reconciled:
        meals = _finite_number(row.get("suggested_meals"))
        if meals is None:
            continue
        actual = float(row["_actual"])
        over_preparation.append(max(meals - actual, 0.0))
        under_preparation.append(max(actual - meals, 0.0))

    metrics = {
        "mae": _mean_or_none(absolute_errors),
        "median_absolute_error": median(absolute_errors) if absolute_errors else None,
        "rmse": (
            math.sqrt(fmean(error * error for error in absolute_errors))
            if absolute_errors
            else None
        ),
        "mean_signed_error": _mean_or_none(signed_errors),
        "underprediction_rate": _mean_or_none(underprediction),
        "q80_empirical_coverage": _quantile_coverage(reconciled),
        "mean_over_preparation": _mean_or_none(over_preparation),
        "mean_under_preparation": _mean_or_none(under_preparation),
    }

    segments: dict[str, dict[str, Any]] = {}
    for label, value in (("Saturday", "sat"), ("Sunday", "sun")):
        selected = _segment_rows(reconciled, value)
        segments[label] = {"reconciled_count": len(selected), "mae": _mae(selected)}

    horizons: dict[str, dict[str, Any]] = {}
    for horizon in (1, 2, 5):
        selected = _horizon_rows(reconciled, horizon)
        horizons[f"H{horizon}"] = {
            "reconciled_count": len(selected),
            "mae": _mae(selected),
            "q80_empirical_coverage": _quantile_coverage(selected),
        }

    missing_hash_count = sum(
        1
        for row in active_rows
        if row.get("feature_order_sha256") in (None, "")
        and row.get("feature_hash") in (None, "")
    )
    invalid_reconciled_count = sum(
        1
        for row in active_rows
        if row.get("actual_visitors") is not None
        and (
            _finite_number(row.get("actual_visitors")) is None
            or _finite_number(row.get("predicted_visitors")) is None
        )
    )
    integrity_alerts: list[str] = []
    if missing_hash_count:
        integrity_alerts.append(
            f"{missing_hash_count} active prediction row(s) do not contain feature-hash provenance."
        )
    if invalid_reconciled_count:
        integrity_alerts.append(
            f"{invalid_reconciled_count} active row(s) have invalid reconciliation values."
        )

    performance_alerts: list[str] = []
    if reconciled_count <= 3:
        performance_alerts.append(
            "Production performance is insufficient for operational conclusions."
        )
    else:
        coverage = metrics["q80_empirical_coverage"]
        if coverage is not None and coverage < 0.80:
            performance_alerts.append("Production Q80 empirical coverage is below 80%.")
        under_rate = metrics["underprediction_rate"]
        if under_rate is not None and under_rate > 0.50:
            performance_alerts.append(
                "Point predictions underpredict more than half of reconciled services."
            )

    waste_values = [
        value
        for row in active_rows
        if (value := _finite_number(row.get("waste_avoided_meals"))) is not None
    ]
    co2e_values = [
        value
        for row in active_rows
        if (value := _finite_number(row.get("estimated_co2e_reduction_kg")))
        is not None
    ]

    return {
        "integrity_ok": True,
        "integrity_errors": [],
        "contract": asdict(contract),
        "prediction_count": prediction_count,
        "reconciled_count": reconciled_count,
        "unreconciled_count": unreconciled_count,
        "stage": stage,
        "metrics": metrics,
        "segments": segments,
        "horizons": horizons,
        "latest_outcomes": _latest_rows(active_rows, latest_limit),
        "sustainability": {
            "total_estimated_waste_avoided_meals": sum(waste_values),
            "total_estimated_co2e_reduction_kg": sum(co2e_values),
        },
        "integrity_alerts": integrity_alerts,
        "performance_alerts": performance_alerts,
    }


def training_run_provenance(run: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(run, Mapping):
        return None
    metrics = run.get("metrics")
    if isinstance(metrics, str):
        return None
    if isinstance(metrics, Mapping):
        provenance = metrics.get(TRAINING_PROVENANCE_KEY)
        if isinstance(provenance, Mapping):
            return provenance
    direct = {
        key: run.get(key)
        for key in (
            "package_id",
            "model_package_schema_version",
            "feature_set_id",
            "feature_order_sha256",
            "recommendation_policy_id",
        )
    }
    return direct if any(value is not None for value in direct.values()) else None


def training_run_matches_active_f6(
    run: Mapping[str, Any] | None, contract: ActiveF6Package
) -> bool:
    provenance = training_run_provenance(run)
    if provenance is None:
        return False
    return (
        str(provenance.get("package_id") or "") == contract.package_id
        and _schema_matches(
            provenance.get("model_package_schema_version"), contract.schema_version
        )
        and str(provenance.get("feature_set_id") or "") == contract.feature_set_id
        and str(provenance.get("feature_order_sha256") or "")
        == contract.feature_order_sha256
        and str(provenance.get("recommendation_policy_id") or "")
        == contract.recommendation_policy_id
    )


def f6_training_status(
    contract: ActiveF6Package,
    *,
    retrain_state: Mapping[str, Any] | None,
    latest_run: Mapping[str, Any] | None,
    latest_successful_run: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Suppress unproven training runs and describe the active F6 status."""

    dirty = bool(retrain_state and retrain_state.get("dirty"))
    matching_latest = (
        dict(latest_run)
        if training_run_matches_active_f6(latest_run, contract)
        else None
    )
    matching_success = (
        dict(latest_successful_run)
        if training_run_matches_active_f6(latest_successful_run, contract)
        else None
    )

    if matching_success is None:
        message = "Activated from verified candidate. First production retraining pending."
        latest_successful_at = None
    else:
        message = "Confirmed production training run available."
        latest_successful_at = matching_success.get("finished_at")

    if dirty:
        status = "RETRAINING_REQUIRED"
    elif matching_latest is not None:
        status = str(matching_latest.get("status") or "UNKNOWN").upper()
    elif matching_success is not None:
        status = "SUCCESS"
    else:
        status = "PENDING_FIRST_F6_RETRAIN"

    return {
        "needs_retraining": dirty,
        "status": status,
        "latest_successful_at": latest_successful_at,
        "attendance_rows": (
            matching_latest.get("attendance_rows")
            if matching_latest is not None
            else matching_success.get("attendance_rows")
            if matching_success is not None
            else None
        ),
        "message": message,
    }


def f6_training_provenance_payload(package: Mapping[str, Any]) -> dict[str, Any]:
    """Build JSON-safe package provenance for future training-run records."""

    feature_contract = package.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise F6IntegrityError("F6 training package feature contract is unavailable.")
    return {
        "package_id": package.get("package_id"),
        "model_package_schema_version": package.get(
            "model_package_schema_version"
        ),
        "feature_set_id": feature_contract.get("feature_set_id"),
        "feature_order_sha256": feature_contract.get("feature_order_sha256"),
        "recommendation_policy_id": package.get("recommendation_policy_id"),
    }
