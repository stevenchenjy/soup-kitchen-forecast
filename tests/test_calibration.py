from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import (
    C0, C1, C2, C3, C4,
    apply_confirmation_calibration_guardrails,
    build_calibration_history_cache,
    calibrate_predictions,
    collapse_residual_observations,
    conformal_empirical_quantile,
    select_development_calibration,
)


def synthetic_predictions(n_dates: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-06", periods=n_dates, freq="7D")
    rows = []
    row_id = 0
    for index, target in enumerate(dates):
        day_type = "Saturday" if index % 2 == 0 else "Sunday"
        for scenario in ("S1", "S2"):
            rows.append({
                "calibration_row_id": row_id,
                "target_date": target,
                "forecast_origin": target - pd.Timedelta(days=1),
                "day_type": day_type,
                "scenario": scenario,
                "actual": 100.0 + index % 9,
                "point_prediction": 96.0 + (scenario == "S2"),
                "raw_quantile_prediction": 103.25,
            })
            row_id += 1
    return pd.DataFrame(rows)


def test_conformal_quantile_uses_clipped_ceil_order_statistic() -> None:
    residuals = [0.0, 1.0, 2.0, 9.0]
    value, rank = conformal_empirical_quantile(residuals, 0.8)
    assert (value, rank) == (9.0, 4)
    value, rank = conformal_empirical_quantile(residuals, 0.2)
    assert (value, rank) == (0.0, 1)


def test_residuals_are_one_maximum_observation_per_target() -> None:
    observations = collapse_residual_observations(synthetic_predictions(5))
    assert len(observations) == 5
    assert observations["target_date"].is_unique
    assert observations["source_scenario_count"].eq(2).all()
    assert observations.iloc[0]["residual_observation"] == 4.0


def test_history_is_strictly_prior_and_origin_available() -> None:
    frame = synthetic_predictions(30)
    cache = build_calibration_history_cache(frame)
    row = frame.iloc[-1]
    history = cache.histories_by_row_id[int(row["calibration_row_id"])]
    used_dates = cache.observations.iloc[history.pooled_indices]["target_date"]
    assert (used_dates < row["target_date"]).all()
    assert (used_dates <= row["forecast_origin"]).all()
    assert row["target_date"] not in set(used_dates)
    assert cache.build_count == 1


def test_daytype_windows_and_pooled_fallback_are_deterministic() -> None:
    frame = synthetic_predictions(60)
    cache = build_calibration_history_cache(frame)
    outputs = {
        policy.calibration_policy_id: calibrate_predictions(
            frame, target_coverage=0.8, calibration_policy=policy, history_cache=cache
        )
        for policy in (C1, C2, C3, C4)
    }
    last_id = int(frame.iloc[-1]["calibration_row_id"])
    last = {key: value[value["calibration_row_id"] == last_id].iloc[0] for key, value in outputs.items()}
    assert last[C1.calibration_policy_id]["calibration_history_count"] == 29
    assert last[C2.calibration_policy_id]["calibration_history_count"] == 29
    assert last[C3.calibration_policy_id]["calibration_history_count"] == 26
    assert last[C4.calibration_policy_id]["calibration_history_count"] == 59
    fallback = outputs[C1.calibration_policy_id]
    fallback = fallback[fallback["fallback_used"] == "pooled_expanding"]
    assert not fallback.empty
    assert fallback["calibration_history_count"].ge(20).all()


def test_calibration_preserves_point_and_uses_ceiling_only_for_meals() -> None:
    frame = synthetic_predictions(30)
    out = calibrate_predictions(frame, target_coverage=0.8, calibration_policy=C1)
    np.testing.assert_array_equal(out["point_prediction"], frame["point_prediction"])
    eligible = out["recommended_meals"].notna()
    np.testing.assert_array_equal(
        out.loc[eligible, "recommended_meals"],
        np.ceil(out.loc[eligible, "unrounded_upper_recommendation"]),
    )
    assert out["provenance_valid"].all()
    assert not out["legacy_percentage_buffer_added"].any()
    assert not out["saved_residual_buffer_added"].any()


def development_table() -> pd.DataFrame:
    values = {
        C0.calibration_policy_id: (0.68, 8.0, 4.0, 0.65, 0.65),
        C1.calibration_policy_id: (0.80, 12.0, 3.0, 0.76, 0.76),
        C2.calibration_policy_id: (0.81, 11.5, 3.2, 0.77, 0.75),
        C3.calibration_policy_id: (0.79, 11.0, 3.1, 0.75, 0.76),
        C4.calibration_policy_id: (0.81, 13.0, 3.0, 0.78, 0.78),
    }
    return pd.DataFrame([{
        "calibration_policy_id": policy,
        "coverage_target": 0.8,
        "empirical_coverage": metrics[0],
        "mean_over_preparation": metrics[1],
        "mean_under_preparation": metrics[2],
        "sunday_coverage": metrics[3],
        "s2_coverage": metrics[4],
    } for policy, metrics in values.items()])


def test_development_selection_uses_preregistered_tolerance_and_order() -> None:
    selected, audit, path = select_development_calibration(development_table())
    assert selected == C1.calibration_policy_id
    assert audit.loc[audit["development_selected"], "calibration_policy_id"].item() == selected
    assert "one_meal" in path


def test_confirmation_guardrail_falls_back_only_to_c0() -> None:
    table = development_table().rename(columns={"mean_under_preparation": "unused"})
    table["p90_over_preparation"] = 20.0
    table.loc[table["calibration_policy_id"] == C0.calibration_policy_id, "p90_over_preparation"] = 10.0
    table.loc[table["calibration_policy_id"] == C1.calibration_policy_id, "empirical_coverage"] = 0.73
    final, audit = apply_confirmation_calibration_guardrails(C1.calibration_policy_id, table)
    assert final == C0.calibration_policy_id
    assert audit.loc[audit["final_recommended_policy"], "calibration_policy_id"].item() == C0.calibration_policy_id
