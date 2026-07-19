from __future__ import annotations

import pandas as pd

from src.f6_charts import (
    build_absolute_error_chart,
    build_actual_vs_predicted_chart,
)


def chart_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_date": pd.to_datetime(["2026-07-11", "2026-07-12"]),
            "actual_visitors": [93.0, 170.0],
            "point_prediction": [104.3, 157.0],
            "q80_prediction": [116.2, 167.2],
            "absolute_error": [11.3, 13.0],
            "model_segment": ["sat", "sun"],
            "service_horizon": [1, 1],
            "scenario": ["S3_next_service", "S3_next_service"],
        }
    )


def _tooltip_fields(spec: dict) -> set[str]:
    return {item["field"] for item in spec["encoding"]["tooltip"]}


def test_actual_vs_predicted_chart_receives_all_three_verified_series() -> None:
    chart = build_actual_vs_predicted_chart(chart_frame())
    spec = chart.to_dict()

    assert {
        "Service Date",
        "Actual Visitors",
        "Expected Visitors",
        "Q80 Recommendation",
        "Saturday/Sunday",
        "Service Horizon",
    }.issubset(chart.data.columns)
    assert spec["transform"][0]["fold"] == [
        "Actual Visitors",
        "Expected Visitors",
        "Q80 Recommendation",
    ]
    assert _tooltip_fields(spec) == {
        "Service Date",
        "Actual Visitors",
        "Expected Visitors",
        "Q80 Recommendation",
        "Saturday/Sunday",
        "Service Horizon",
    }
    assert spec["encoding"]["y"]["title"] == "Visitors"
    assert spec["height"] == 320
    assert spec["params"]


def test_absolute_error_chart_uses_point_error_without_q80_error() -> None:
    chart = build_absolute_error_chart(chart_frame())
    spec = chart.to_dict()

    assert "Absolute Error" in chart.data.columns
    assert spec["encoding"]["y"]["field"] == "Absolute Error"
    assert _tooltip_fields(spec) == {
        "Service Date",
        "Actual Visitors",
        "Expected Visitors",
        "Absolute Error",
        "Saturday/Sunday",
        "Service Horizon",
    }
    assert "Q80 Recommendation" not in _tooltip_fields(spec)
    assert spec["height"] == 280
    assert spec["params"]
