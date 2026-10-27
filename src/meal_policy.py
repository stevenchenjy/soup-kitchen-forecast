"""Versioned preparation decisions applied separately from the raw F6 forecast.

Model packages and their historical C0 backtests retain their original meaning.
The tracked location/date policy changes only the final number of meals to prepare.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.config import PROJECT_ROOT


RAW_Q80_POLICY_ID = "C0_EXISTING_RAW_QUANTILE"
MEAL_POLICY_CONFIG_PATH = PROJECT_ROOT / "config" / "meal_policies.json"


def _service_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("Meal policy service date must use YYYY-MM-DD.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Meal policy service date must use YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise ValueError("Meal policy service date must use YYYY-MM-DD.")
    return parsed


@dataclass(frozen=True)
class MealPreparationPolicy:
    policy_id: str = RAW_Q80_POLICY_ID
    reduction_meals: int = 0
    effective_from_service_date: date | None = None
    config_fingerprint: str | None = None

    def for_date(self, service_date: Any) -> MealPreparationPolicy:
        target = _service_date(service_date)
        if (
            self.effective_from_service_date is not None
            and target < self.effective_from_service_date
        ):
            return replace(
                self,
                policy_id=RAW_Q80_POLICY_ID,
                reduction_meals=0,
                effective_from_service_date=None,
            )
        return self

    def apply(self, raw_meals: int) -> int:
        if type(raw_meals) is not int:
            raise ValueError("Raw recommended meals must be a whole meal count.")
        # Preserve the original C0 behavior when no preparation adjustment applies.
        if self.reduction_meals == 0:
            return raw_meals
        return max(0, raw_meals - self.reduction_meals)


def raw_preparation_policy() -> MealPreparationPolicy:
    return MealPreparationPolicy()


def _config_bytes(path: str | Path | None) -> bytes:
    source = Path(path) if path is not None else MEAL_POLICY_CONFIG_PATH
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ValueError("Meal preparation policy configuration is unavailable.") from exc


def meal_policy_config_fingerprint(*, path: str | Path | None = None) -> str:
    """Hash exact config bytes so a running API notices replacements and edits."""

    return hashlib.sha256(_config_bytes(path)).hexdigest()


def load_preparation_policy(
    location_id: str | None, *, path: str | Path | None = None
) -> MealPreparationPolicy:
    """Read and validate one immutable policy/config snapshot.

    Unknown locations receive raw C0. An invalid configured policy raises instead
    of silently changing the intended production preparation rule.
    """

    raw = _config_bytes(path)
    fingerprint = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Meal preparation policy configuration is invalid JSON.") from exc
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("Meal preparation policy schema_version must be 1.")
    if payload["schema_version"] != 1:
        raise ValueError("Meal preparation policy schema_version must be 1.")
    policies = payload.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("Meal preparation policies must be a location mapping.")

    validated: dict[str, MealPreparationPolicy] = {}
    for configured_location, specification in policies.items():
        if not isinstance(configured_location, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", configured_location
        ):
            raise ValueError("Meal preparation policy location ID is invalid.")
        if not isinstance(specification, dict):
            raise ValueError("Meal preparation policy must be a mapping.")
        if specification.get("base_policy_id") != RAW_Q80_POLICY_ID:
            raise ValueError("Meal preparation policy must preserve the raw C0 base.")
        reduction = specification.get("reduction_meals")
        if type(reduction) is not int or not 1 <= reduction <= 10:
            raise ValueError(
                "Meal preparation reduction_meals must be a positive integer from 1 to 10."
            )
        expected_id = f"P1_EQUAL_WEIGHT_Q80_MINUS_{reduction}"
        if specification.get("policy_id") != expected_id:
            raise ValueError("Meal preparation policy ID does not identify its reduction.")
        effective_from = _service_date(specification.get("effective_from_service_date"))
        validated[configured_location] = MealPreparationPolicy(
            policy_id=expected_id,
            reduction_meals=reduction,
            effective_from_service_date=effective_from,
            config_fingerprint=fingerprint,
        )

    return validated.get(
        location_id,
        MealPreparationPolicy(config_fingerprint=fingerprint),
    )
