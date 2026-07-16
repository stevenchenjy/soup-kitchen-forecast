from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.production_features import RECOMMENDATION_POLICY_ID


@dataclass(frozen=True)
class RecommendationUiPolicy:
    show_percentage_buffer: bool
    recommendation_label: str
    package_caption: str


def recommendation_ui_policy(predictor: Any) -> RecommendationUiPolicy:
    is_f6_c0 = bool(getattr(predictor, "uses_locked_f6", False)) and (
        getattr(predictor, "recommendation_policy_id", None) == RECOMMENDATION_POLICY_ID
    )
    schema_version = int(getattr(predictor, "model_package_schema_version", 1))
    package_id = str(getattr(predictor, "package_id", "unknown"))
    return RecommendationUiPolicy(
        show_percentage_buffer=not is_f6_c0,
        recommendation_label=(
            "Raw 80th-percentile recommendation" if is_f6_c0 else "Recommended Meals"
        ),
        package_caption=f"Model package: {package_id} · schema v{schema_version}",
    )
