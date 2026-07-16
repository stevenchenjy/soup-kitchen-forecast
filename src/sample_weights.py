"""Centralized sample-weight policies and Phase 2B2-Lite decision rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleWeightPolicy:
    sample_weight_id: str
    half_life_rows: int | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SW_UNIFORM = SampleWeightPolicy(
    "SW_UNIFORM", None, "Every expanding-history segment row receives weight 1.0."
)
SW_HL104 = SampleWeightPolicy(
    "SW_HL104", 104, "Segment-row recency half-life of 104 rows."
)
SW_HL52 = SampleWeightPolicy(
    "SW_HL52", 52, "Segment-row recency half-life of 52 rows."
)
SAMPLE_WEIGHT_CANDIDATES: tuple[SampleWeightPolicy, ...] = (
    SW_UNIFORM,
    SW_HL104,
    SW_HL52,
)
SAMPLE_WEIGHTS_BY_ID = {
    item.sample_weight_id: item for item in SAMPLE_WEIGHT_CANDIDATES
}


def resolve_sample_weight_policy(
    value: SampleWeightPolicy | str,
) -> SampleWeightPolicy:
    if isinstance(value, SampleWeightPolicy):
        canonical = SAMPLE_WEIGHTS_BY_ID.get(value.sample_weight_id)
        if canonical != value:
            raise ValueError(f"Non-canonical sample-weight policy: {value.sample_weight_id}")
        return value
    try:
        return SAMPLE_WEIGHTS_BY_ID[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported sample-weight policy: {value}") from exc


def segment_age_ranks(row_count: int) -> np.ndarray:
    """Return oldest-to-newest ages with the newest segment row at age zero."""

    count = int(row_count)
    if count < 0:
        raise ValueError("row_count must be nonnegative")
    return np.arange(count - 1, -1, -1, dtype=float)


def generate_sample_weights(
    policy: SampleWeightPolicy | str,
    row_count: int,
) -> np.ndarray:
    """Generate positive monotone segment-local weights in training-row order."""

    resolved = resolve_sample_weight_policy(policy)
    ages = segment_age_ranks(row_count)
    if resolved.half_life_rows is None:
        weights = np.ones(len(ages), dtype=float)
    else:
        weights = np.power(0.5, ages / float(resolved.half_life_rows))
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise AssertionError("Sample weights must be positive and finite")
    if len(weights) and weights[-1] != 1.0:
        raise AssertionError("Newest segment training row must have weight 1.0")
    if len(weights) > 1 and (np.diff(weights) < 0).any():
        raise AssertionError("Weights must be monotone nondecreasing toward the newest row")
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Effective sample size requires a nonempty one-dimensional vector")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Effective sample size requires positive finite weights")
    return float(np.square(values.sum()) / np.square(values).sum())


DEVELOPMENT_SELECTION_COLUMNS = [
    "sample_weight_id",
    "development_macro_mae",
    "development_micro_mae",
    "development_recent_tail_mae",
    "development_s2_mae",
    "development_saturday_mae",
    "development_sunday_mae",
    "development_p90_absolute_error",
    "development_bias",
]


def select_development_policy(
    development_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Apply only the preregistered development qualification rules.

    Confirmation columns, if present in the caller's frame, are never copied into
    the selection view and therefore cannot affect the decision.
    """

    missing = set(DEVELOPMENT_SELECTION_COLUMNS).difference(development_metrics.columns)
    if missing:
        raise ValueError(f"Development metrics missing columns: {sorted(missing)}")
    table = development_metrics[DEVELOPMENT_SELECTION_COLUMNS].copy()
    if set(table["sample_weight_id"]) != set(SAMPLE_WEIGHTS_BY_ID) or len(table) != 3:
        raise ValueError("Development table must contain exactly the three locked policies")
    if table["sample_weight_id"].duplicated().any():
        raise ValueError("Development table contains duplicate sample-weight policies")
    uniform = table[table["sample_weight_id"] == SW_UNIFORM.sample_weight_id].iloc[0]
    table["macro_mae_improvement"] = (
        uniform["development_macro_mae"] - table["development_macro_mae"]
    )
    table["recent_tail_mae_improvement"] = (
        uniform["development_recent_tail_mae"]
        - table["development_recent_tail_mae"]
    )
    table["s2_mae_change"] = table["development_s2_mae"] - uniform["development_s2_mae"]
    table["saturday_mae_change"] = (
        table["development_saturday_mae"] - uniform["development_saturday_mae"]
    )
    table["sunday_mae_change"] = (
        table["development_sunday_mae"] - uniform["development_sunday_mae"]
    )
    table["p90_change"] = (
        table["development_p90_absolute_error"]
        - uniform["development_p90_absolute_error"]
    )
    weighted = table["sample_weight_id"] != SW_UNIFORM.sample_weight_id
    table["passes_macro_threshold"] = weighted & table["macro_mae_improvement"].ge(0.25)
    table["passes_recent_tail_threshold"] = weighted & table[
        "recent_tail_mae_improvement"
    ].ge(0.15)
    table["passes_s2_guardrail"] = weighted & table["s2_mae_change"].le(0.25)
    table["passes_saturday_guardrail"] = weighted & table["saturday_mae_change"].le(0.40)
    table["passes_sunday_guardrail"] = weighted & table["sunday_mae_change"].le(0.40)
    table["passes_p90_guardrail"] = weighted & table["p90_change"].le(0.50)
    rules = [
        "passes_macro_threshold",
        "passes_recent_tail_threshold",
        "passes_s2_guardrail",
        "passes_saturday_guardrail",
        "passes_sunday_guardrail",
        "passes_p90_guardrail",
    ]
    table["development_qualified"] = table[rules].all(axis=1)
    qualified = table[table["development_qualified"]].copy()
    if qualified.empty:
        selected = SW_UNIFORM.sample_weight_id
    elif len(qualified) == 1:
        selected = str(qualified.iloc[0]["sample_weight_id"])
    else:
        best = qualified.sort_values(
            ["development_macro_mae", "sample_weight_id"], kind="stable"
        ).iloc[0]
        hl104 = qualified[
            qualified["sample_weight_id"] == SW_HL104.sample_weight_id
        ]
        if not hl104.empty and abs(
            float(hl104.iloc[0]["development_macro_mae"])
            - float(best["development_macro_mae"])
        ) <= 0.10:
            selected = SW_HL104.sample_weight_id
        else:
            selected = str(best["sample_weight_id"])
    table["development_selected"] = table["sample_weight_id"].eq(selected)
    return selected, table


CONFIRMATION_GUARDRAIL_COLUMNS = [
    "sample_weight_id",
    "confirmation_mae",
    "confirmation_recent_52_mae",
    "confirmation_s2_mae",
    "confirmation_saturday_mae",
    "confirmation_sunday_mae",
    "confirmation_p90_absolute_error",
]


def apply_confirmation_guardrail(
    locked_development_policy: str,
    confirmation_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Apply post-lock confirmation guardrails without searching another policy."""

    locked = resolve_sample_weight_policy(locked_development_policy).sample_weight_id
    missing = set(CONFIRMATION_GUARDRAIL_COLUMNS).difference(confirmation_metrics.columns)
    if missing:
        raise ValueError(f"Confirmation metrics missing columns: {sorted(missing)}")
    table = confirmation_metrics[CONFIRMATION_GUARDRAIL_COLUMNS].copy()
    if set(table["sample_weight_id"]) != set(SAMPLE_WEIGHTS_BY_ID) or len(table) != 3:
        raise ValueError("Confirmation table must contain exactly the three locked policies")
    uniform = table[table["sample_weight_id"] == SW_UNIFORM.sample_weight_id].iloc[0]
    changes = {
        "confirmation_mae_change": ("confirmation_mae", 0.20),
        "confirmation_recent_52_mae_change": ("confirmation_recent_52_mae", 0.20),
        "confirmation_s2_mae_change": ("confirmation_s2_mae", 0.50),
        "confirmation_saturday_mae_change": ("confirmation_saturday_mae", 0.50),
        "confirmation_sunday_mae_change": ("confirmation_sunday_mae", 0.50),
        "confirmation_p90_change": ("confirmation_p90_absolute_error", 1.00),
    }
    for output, (source, threshold) in changes.items():
        table[output] = table[source] - uniform[source]
        table[f"passes_{output}"] = table[output].le(threshold)
    pass_columns = [f"passes_{column}" for column in changes]
    table["passes_all_confirmation_guardrails"] = table[pass_columns].all(axis=1)
    if locked == SW_UNIFORM.sample_weight_id:
        final = locked
        table["guardrail_applied_to_locked_policy"] = False
    else:
        locked_row = table[table["sample_weight_id"] == locked].iloc[0]
        final = locked if bool(locked_row["passes_all_confirmation_guardrails"]) else SW_UNIFORM.sample_weight_id
        table["guardrail_applied_to_locked_policy"] = table["sample_weight_id"].eq(locked)
    table["final_selected"] = table["sample_weight_id"].eq(final)
    return final, table


def candidate_registry_rows() -> list[dict[str, Any]]:
    return [item.to_dict() for item in SAMPLE_WEIGHT_CANDIDATES]
