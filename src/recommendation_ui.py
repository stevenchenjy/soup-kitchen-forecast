from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import src.config as config
from src.production_features import RECOMMENDATION_POLICY_ID


@dataclass(frozen=True)
class RecommendationUiPolicy:
    show_percentage_buffer: bool
    recommendation_label: str
    package_caption: str
    recommendation_caption: str = (
        "Recommended meals include a built-in safety margin based on expected attendance."
    )
    preparation_label: str = "Raw Q80 recommendation"


def next_service_preparation_policy(predictor: Any):
    """Use the predictor's policy snapshot for the next eligible service date."""
    policy = getattr(predictor, "preparation_policy", None)
    if policy is None:
        return None
    target = config.forecast_today(getattr(predictor, "weather_timezone", "America/New_York"))
    while target.weekday() not in (5, 6):
        target += timedelta(days=1)
    return policy.for_date(target)


def recommendation_ui_policy(predictor: Any) -> RecommendationUiPolicy:
    is_f6_c0 = bool(getattr(predictor, "uses_locked_f6", False)) and (
        getattr(predictor, "recommendation_policy_id", None) == RECOMMENDATION_POLICY_ID
    )
    schema_version = int(getattr(predictor, "model_package_schema_version", 1))
    package_id = str(getattr(predictor, "package_id", "unknown"))
    feature_set_id = ""
    feature_contract = getattr(predictor, "feature_contract", None)
    if isinstance(feature_contract, dict):
        feature_set_id = str(feature_contract.get("feature_set_id", ""))
    package_caption = f"Model package: {package_id} · schema v{schema_version}"
    if is_f6_c0 and feature_set_id:
        package_caption += f" · {feature_set_id}"
    preparation = next_service_preparation_policy(predictor)
    uses_equal_weight = is_f6_c0 and preparation is not None and preparation.reduction_meals > 0
    if uses_equal_weight:
        package_caption += f" · preparation {preparation.policy_id}"
    return RecommendationUiPolicy(
        show_percentage_buffer=not is_f6_c0,
        recommendation_label=(
            "Equal-weight meal recommendation" if uses_equal_weight
            else "Raw 80th-percentile recommendation" if is_f6_c0 else "Recommended Meals"
        ),
        package_caption=package_caption,
        recommendation_caption=(
            "Meal recommendations balance excess meals and shortages equally."
            if uses_equal_weight else
            "Recommended meals include a built-in safety margin based on expected attendance."
        ),
        preparation_label=("Equal-weight meal plan" if uses_equal_weight else "Raw Q80 recommendation"),
    )
