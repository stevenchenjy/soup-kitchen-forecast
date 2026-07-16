from __future__ import annotations

from types import SimpleNamespace

from src.production_features import RECOMMENDATION_POLICY_ID
from src.recommendation_ui import recommendation_ui_policy


def test_f6_c0_ui_hides_percentage_buffer_and_labels_raw_quantile() -> None:
    predictor = SimpleNamespace(
        uses_locked_f6=True,
        recommendation_policy_id=RECOMMENDATION_POLICY_ID,
        model_package_schema_version=2,
        package_id="f6-candidate-v1",
    )
    policy = recommendation_ui_policy(predictor)
    assert policy.show_percentage_buffer is False
    assert policy.recommendation_label == "Raw 80th-percentile recommendation"
    assert policy.package_caption == "Model package: f6-candidate-v1 · schema v2"


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
