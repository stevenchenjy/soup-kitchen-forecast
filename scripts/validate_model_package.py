from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import warnings
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.model_publication import (
    sha256_file,
    validate_f6_package_file,
    validate_legacy_package_file,
)
from src.predictor import VisitorPredictor


def _parse_smoke(value: str) -> tuple[str, str]:
    try:
        target, origin = value.split(":", 1)
        return pd.Timestamp(target).date().isoformat(), pd.Timestamp(origin).date().isoformat()
    except Exception as exc:
        raise argparse.ArgumentTypeError("smoke must be TARGET_DATE:FORECAST_ORIGIN") from exc


def _fake_weather(target_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "date": pd.Timestamp(target_date),
            "temp_10_13": 72.0,
            "apparent_temp_10_13": 73.0,
            "humidity_10_13": 55.0,
            "wind_10_13": 6.0,
            "precip_10_13": 0.0,
        }]
    )


def _signature(output) -> dict[str, object]:
    return {
        "service_date": output.service_date.date().isoformat(),
        "predicted_visitors": output.predicted_visitors,
        "predicted_quantile": output.predicted_quantile,
        "residual_buffer": output.residual_buffer,
        "suggested_meals": output.suggested_meals,
        "meal_buffer_pct": output.meal_buffer_pct,
        "model_segment": output.model_segment,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a model package in a fresh process.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-schema", type=int, choices=(1, 2), required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--package-id")
    parser.add_argument("--smoke", action="append", type=_parse_smoke, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.package).resolve()
    if sha256_file(path) != args.expected_sha256:
        raise ValueError("Package hash differs from --expected-sha256")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if args.expected_schema == 2:
            validation = validate_f6_package_file(
                path,
                expected_sha256=args.expected_sha256,
                expected_package_id=args.package_id,
            )
        else:
            if args.package_id:
                raise ValueError("--package-id is only valid for schema-v2 validation")
            validation = validate_legacy_package_file(
                path, expected_sha256=args.expected_sha256
            )
        predictor = VisitorPredictor(str(path))

    signatures = []
    for target, origin in args.smoke:
        with patch("src.config.forecast_today", return_value=pd.Timestamp(origin).date()), patch(
            "src.predictor.WeatherClient.fetch_forecast_daily",
            side_effect=lambda requested: _fake_weather(requested),
        ):
            signatures.append(_signature(predictor.predict_next(target)))
    print(json.dumps({
        "validation": validation,
        "load_warning_count": len(caught),
        "smoke_predictions": signatures,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
