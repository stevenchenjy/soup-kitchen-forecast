from __future__ import annotations

from types import SimpleNamespace
from datetime import date
from unittest.mock import patch

from src.production_features import RECOMMENDATION_POLICY_ID
from src.recommendation_ui import recommendation_ui_policy


def test_f6_c0_ui_hides_percentage_buffer_and_labels_raw_quantile() -> None:
    predictor = SimpleNamespace(
        uses_locked_f6=True,
        recommendation_policy_id=RECOMMENDATION_POLICY_ID,
        model_package_schema_version=2,
        package_id="f6-candidate-v1",
        feature_contract={"feature_set_id": "F6_COMPACT_SELECTED"},
    )
    policy = recommendation_ui_policy(predictor)
    assert policy.show_percentage_buffer is False
    assert policy.recommendation_label == "Raw 80th-percentile recommendation"
    assert policy.package_caption == (
        "Model package: f6-candidate-v1 · schema v2 · F6_COMPACT_SELECTED"
    )


def test_legacy_ui_retains_percentage_buffer_and_identifies_schema() -> None:
    predictor = SimpleNamespace(
        uses_locked_f6=False,
        recommendation_policy_id="LEGACY_MAX_OF_POINT_QUANTILE_AND_BUFFERS",
        model_package_schema_version=1,
        package_id="visitor_model_ny_12550.joblib",
    )
    policy = recommendation_ui_policy(predictor)
    assert policy.show_percentage_buffer is True
    assert policy.recommendation_label == "Recommended Meals"
    assert policy.package_caption == (
        "Model package: visitor_model_ny_12550.joblib · schema v1"
    )


def test_equal_weight_ui_describes_the_effective_plan_without_relabelling_raw_q80():
    from src.meal_policy import load_preparation_policy

    predictor = SimpleNamespace(
        uses_locked_f6=True,
        recommendation_policy_id=RECOMMENDATION_POLICY_ID,
        model_package_schema_version=2,
        package_id="f6-candidate-v1",
        feature_contract={"feature_set_id": "F6_COMPACT_SELECTED"},
        preparation_policy=load_preparation_policy("ny_12550"),
    )
    with patch("src.config.forecast_today", return_value=date(2026, 10, 26)):
        current = recommendation_ui_policy(predictor)
    assert current.show_percentage_buffer is False
    assert current.recommendation_label == "Equal-weight meal recommendation"
    assert current.preparation_label == "Equal-weight meal plan"
    assert current.recommendation_caption == (
        "Meal recommendations balance excess meals and shortages equally."
    )
    assert "P1_EQUAL_WEIGHT_Q80_MINUS_7" in current.package_caption
    with patch("src.config.forecast_today", return_value=date(2026, 10, 19)):
        previous = recommendation_ui_policy(predictor)
    assert previous.preparation_label == "Raw Q80 recommendation"
    assert previous.recommendation_label == "Raw 80th-percentile recommendation"
