from datetime import date
import hashlib
import json

import pytest

from src.meal_policy import (
    RAW_Q80_POLICY_ID,
    load_preparation_policy,
    meal_policy_config_fingerprint,
)


def configuration():
    return {
        "schema_version": 1,
        "policies": {
            "ny_12550": {
                "policy_id": "P1_EQUAL_WEIGHT_Q80_MINUS_7",
                "base_policy_id": RAW_Q80_POLICY_ID,
                "reduction_meals": 7,
                "effective_from_service_date": "2026-10-31",
            }
        },
    }


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(configuration()), encoding="utf-8")
    return path


def test_tracked_policy_is_the_approved_equal_weight_plan():
    policy = load_preparation_policy("ny_12550")
    assert policy.policy_id == "P1_EQUAL_WEIGHT_Q80_MINUS_7"
    assert policy.reduction_meals == 7
    assert policy.effective_from_service_date == date(2026, 10, 31)


def test_start_date_changes_preparation_but_preserves_prior_c0(policy_file):
    policy = load_preparation_policy("ny_12550", path=policy_file)
    before = policy.for_date("2026-10-25")
    assert before.policy_id == RAW_Q80_POLICY_ID
    assert before.reduction_meals == 0
    assert before.apply(121) == 121
    assert policy.for_date("2026-10-31").apply(121) == 114
    assert policy.for_date(date(2026, 11, 1)).policy_id == policy.policy_id


@pytest.mark.parametrize("raw_meals,expected", [(0, 0), (6, 0), (7, 0), (8, 1), (121, 114)])
def test_reduced_count_cannot_be_negative(policy_file, raw_meals, expected):
    policy = load_preparation_policy("ny_12550", path=policy_file)
    assert policy.apply(raw_meals) == expected


@pytest.mark.parametrize("location_id", ["nyc_10001", None])
def test_other_or_unknown_location_keeps_c0(policy_file, location_id):
    policy = load_preparation_policy(location_id, path=policy_file)
    assert policy.for_date("2026-10-31").policy_id == RAW_Q80_POLICY_ID
    assert policy.apply(121) == 121


def test_policy_and_fingerprint_are_one_immutable_snapshot(policy_file):
    original = load_preparation_policy("ny_12550", path=policy_file)
    original_bytes = policy_file.read_bytes()
    assert original.config_fingerprint == hashlib.sha256(original_bytes).hexdigest()
    changed = configuration()
    changed["policies"]["ny_12550"].update(
        reduction_meals=6, policy_id="P1_EQUAL_WEIGHT_Q80_MINUS_6"
    )
    policy_file.write_text(json.dumps(changed), encoding="utf-8")
    replacement = load_preparation_policy("ny_12550", path=policy_file)
    assert original.apply(121) == 114
    assert replacement.apply(121) == 115
    assert replacement.config_fingerprint != original.config_fingerprint
    assert replacement.config_fingerprint == meal_policy_config_fingerprint(path=policy_file)


@pytest.mark.parametrize("reduction", [-1, 0, 11, 7.5, True, "7", None])
def test_invalid_reduction_fails_instead_of_falling_back(policy_file, reduction):
    payload = configuration()
    payload["policies"]["ny_12550"]["reduction_meals"] = reduction
    policy_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_preparation_policy("ny_12550", path=policy_file)


@pytest.mark.parametrize(
    "changes,expected_error",
    [
        ({"policy_id": "C0_EXISTING_RAW_QUANTILE"}, "identify its reduction"),
        ({"base_policy_id": "other"}, "raw C0 base"),
        ({"effective_from_service_date": "2026-02-30"}, "YYYY-MM-DD"),
        ({"effective_from_service_date": "20261031"}, "YYYY-MM-DD"),
        ({"effective_from_service_date": None}, "YYYY-MM-DD"),
    ],
)
def test_inconsistent_policy_metadata_is_rejected(policy_file, changes, expected_error):
    payload = configuration()
    payload["policies"]["ny_12550"].update(changes)
    policy_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=expected_error):
        load_preparation_policy("ny_12550", path=policy_file)


@pytest.mark.parametrize("payload", [{}, {"schema_version": True}, {"schema_version": 2}, []])
def test_unsupported_configuration_schema_is_rejected(policy_file, payload):
    policy_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_preparation_policy("ny_12550", path=policy_file)


def test_unreadable_or_invalid_config_is_not_silently_ignored(policy_file):
    policy_file.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_preparation_policy("ny_12550", path=policy_file)
    policy_file.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        load_preparation_policy("ny_12550", path=policy_file)
