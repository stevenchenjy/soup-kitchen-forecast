"""Altair charts for authenticated F6 historical performance rows."""

from __future__ import annotations

import altair as alt
import pandas as pd


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["Saturday/Sunday"] = display["model_segment"].map(
        {"sat": "Saturday", "sun": "Sunday"}
    )
    return display.rename(
        columns={
            "service_date": "Service Date",
            "actual_visitors": "Actual Visitors",
            "point_prediction": "Expected Visitors",
            "q80_prediction": "Q80 Recommendation",
            "absolute_error": "Absolute Error",
            "service_horizon": "Service Horizon",
        }
    )


def build_actual_vs_predicted_chart(frame: pd.DataFrame) -> alt.Chart:
    """Plot actual, expected, and Q80 recommendation series over service dates."""

    display = _display_frame(frame)
    return (
        alt.Chart(display)
        .transform_fold(
            ["Actual Visitors", "Expected Visitors", "Q80 Recommendation"],
            as_=["Series", "Visitors"],
        )
        .mark_line(point=False, strokeWidth=2)
        .encode(
            x=alt.X("Service Date:T", title="Service date"),
            y=alt.Y("Visitors:Q", title="Visitors", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                title=None,
                sort=[
                    "Actual Visitors",
                    "Expected Visitors",
                    "Q80 Recommendation",
                ],
            ),
            tooltip=[
                alt.Tooltip(
                    "Service Date:T", title="Service date", format="%b %d, %Y"
                ),
                alt.Tooltip("Actual Visitors:Q", format=".1f"),
                alt.Tooltip("Expected Visitors:Q", format=".1f"),
                alt.Tooltip("Q80 Recommendation:Q", format=".1f"),
                alt.Tooltip("Saturday/Sunday:N"),
                alt.Tooltip("Service Horizon:Q", format=".0f"),
            ],
        )
        .properties(height=320)
        .interactive(bind_y=False)
    )


def build_absolute_error_chart(frame: pd.DataFrame) -> alt.Chart:
    """Plot absolute point-prediction error over service dates."""

    display = _display_frame(frame)
    return (
        alt.Chart(display)
        .mark_line(point=False, strokeWidth=2, color="#d95f02")
        .encode(
            x=alt.X("Service Date:T", title="Service date"),
            y=alt.Y(
                "Absolute Error:Q",
                title="Absolute error (visitors)",
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip(
                    "Service Date:T", title="Service date", format="%b %d, %Y"
                ),
                alt.Tooltip("Actual Visitors:Q", format=".1f"),
                alt.Tooltip("Expected Visitors:Q", format=".1f"),
                alt.Tooltip("Absolute Error:Q", format=".1f"),
                alt.Tooltip("Saturday/Sunday:N"),
                alt.Tooltip("Service Horizon:Q", format=".0f"),
            ],
        )
        .properties(height=280)
        .interactive(bind_y=False)
    )
