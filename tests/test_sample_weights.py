from __future__ import annotations

import numpy as np
import pandas as pd

from src.sample_weights import (
    SAMPLE_WEIGHT_CANDIDATES,
    SW_HL104,
    SW_HL52,
    SW_UNIFORM,
    apply_confirmation_guardrail,
    effective_sample_size,
    generate_sample_weights,
    select_development_policy,
)


def test_uniform_weights_are_all_ones_and_keep_every_row() -> None:
    weights = generate_sample_weights(SW_UNIFORM, 137)
    assert len(weights) == 137
    np.testing.assert_array_equal(weights, np.ones(137))
    assert effective_sample_size(weights) == 137.0


def test_half_life_formulas_are_exact() -> None:
    for policy, half_life in [(SW_HL104, 104), (SW_HL52, 52)]:
        weights = generate_sample_weights(policy, 9)
        ages = np.arange(8, -1, -1, dtype=float)
        np.testing.assert_allclose(weights, 0.5 ** (ages / half_life), rtol=0, atol=0)


def test_newest_weight_is_one_and_weights_are_positive_finite_monotone() -> None:
    for policy in SAMPLE_WEIGHT_CANDIDATES:
        weights = generate_sample_weights(policy, 250)
        assert weights[-1] == 1.0
        assert np.isfinite(weights).all()
        assert (weights > 0).all()
        assert (np.diff(weights) >= 0).all()


def test_saturday_and_sunday_segment_ages_are_independent() -> None:
    saturday = generate_sample_weights(SW_HL52, 40)
    sunday = generate_sample_weights(SW_HL52, 27)
    assert saturday[-1] == sunday[-1] == 1.0
    assert saturday[0] == 0.5 ** (39 / 52)
    assert sunday[0] == 0.5 ** (26 / 52)


def development_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_weight_id": "SW_UNIFORM",
                "development_macro_mae": 10.00,
                "development_micro_mae": 10.00,
                "development_recent_tail_mae": 9.00,
                "development_s2_mae": 10.00,
                "development_saturday_mae": 10.00,
                "development_sunday_mae": 10.00,
                "development_p90_absolute_error": 20.00,
                "development_bias": 0.00,
            },
            {
                "sample_weight_id": "SW_HL104",
                "development_macro_mae": 9.75,
                "development_micro_mae": 9.90,
                "development_recent_tail_mae": 8.85,
                "development_s2_mae": 10.25,
                "development_saturday_mae": 10.40,
                "development_sunday_mae": 10.40,
                "development_p90_absolute_error": 20.50,
                "development_bias": 0.10,
            },
            {
                "sample_weight_id": "SW_HL52",
                "development_macro_mae": 9.70,
                "development_micro_mae": 9.80,
                "development_recent_tail_mae": 8.80,
                "development_s2_mae": 10.20,
                "development_saturday_mae": 10.30,
                "development_sunday_mae": 10.30,
                "development_p90_absolute_error": 20.40,
                "development_bias": 0.10,
            },
        ]
    )


def test_development_thresholds_are_inclusive_and_weaker_decay_wins_close_tie() -> None:
    selected, audit = select_development_policy(development_rows())
    assert selected == "SW_HL104"
    assert audit.loc[audit["sample_weight_id"] != "SW_UNIFORM", "development_qualified"].all()


def test_development_selection_cannot_read_confirmation_metrics() -> None:
    first = development_rows().assign(confirmation_mae=[999, -999, -999])
    second = development_rows().assign(confirmation_mae=[-999, 999, 999])
    selected_first, audit_first = select_development_policy(first)
    selected_second, audit_second = select_development_policy(second)
    assert selected_first == selected_second == "SW_HL104"
    assert "confirmation_mae" not in audit_first.columns
    pd.testing.assert_frame_equal(audit_first, audit_second)


def confirmation_rows(weighted_mae_change: float = 0.20) -> pd.DataFrame:
    base = {
        "confirmation_mae": 10.0,
        "confirmation_recent_52_mae": 10.0,
        "confirmation_s2_mae": 10.0,
        "confirmation_saturday_mae": 10.0,
        "confirmation_sunday_mae": 10.0,
        "confirmation_p90_absolute_error": 20.0,
    }
    rows = [{"sample_weight_id": "SW_UNIFORM", **base}]
    for policy in ["SW_HL104", "SW_HL52"]:
        row = {"sample_weight_id": policy, **base}
        row["confirmation_mae"] += weighted_mae_change
        row["confirmation_recent_52_mae"] += 0.20
        row["confirmation_s2_mae"] += 0.50
        row["confirmation_saturday_mae"] += 0.50
        row["confirmation_sunday_mae"] += 0.50
        row["confirmation_p90_absolute_error"] += 1.00
        rows.append(row)
    return pd.DataFrame(rows)


def test_confirmation_guardrails_are_inclusive_and_failure_reverts_to_uniform() -> None:
    selected, passing = apply_confirmation_guardrail("SW_HL104", confirmation_rows())
    assert selected == "SW_HL104"
    assert passing.loc[passing["sample_weight_id"] == "SW_HL104", "passes_all_confirmation_guardrails"].item()
    selected, failing = apply_confirmation_guardrail(
        "SW_HL104", confirmation_rows(weighted_mae_change=0.200001)
    )
    assert selected == "SW_UNIFORM"
    assert not failing.loc[failing["sample_weight_id"] == "SW_HL104", "passes_all_confirmation_guardrails"].item()

