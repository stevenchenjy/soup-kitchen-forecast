from __future__ import annotations

import pandas as pd
import pytest

from src.training_windows import (
    TRAINING_WINDOW_CANDIDATES,
    TW_26,
    TW_52,
    TW_EXPANDING,
    apply_training_window,
    resolve_training_window,
    select_training_window,
    validate_candidate_ids,
)


def segment_frame(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_date": pd.date_range("2023-01-01", periods=count, freq="7D"),
            "visitors": range(count),
            "segment": "sun",
        }
    )


def test_locked_registry_has_exact_candidates_and_order() -> None:
    ids = tuple(item.training_window_id for item in TRAINING_WINDOW_CANDIDATES)
    assert ids == ("TW_EXPANDING", "TW_104", "TW_78", "TW_52", "TW_26")
    validate_candidate_ids(ids)
    with pytest.raises(ValueError, match="Candidate order"):
        validate_candidate_ids(reversed(ids))


def test_window_retains_latest_segment_rows_after_upstream_filtering() -> None:
    frame = segment_frame(40)
    result = apply_training_window(frame, TW_26)
    assert result.available_segment_training_rows == 40
    assert result.retained_segment_training_rows == 26
    assert result.window_constrained is True
    assert result.frame["service_date"].tolist() == frame.tail(26)["service_date"].tolist()
    assert result.retained_training_start_date == frame.iloc[-26]["service_date"]
    assert result.retained_training_end_date == frame.iloc[-1]["service_date"]


def test_short_history_is_unconstrained_and_expanding_is_exact() -> None:
    frame = segment_frame(18)
    short = apply_training_window(frame, TW_52)
    expanding = apply_training_window(frame, None)
    assert short.window_constrained is False
    assert short.retained_segment_training_rows == 18
    pd.testing.assert_frame_equal(short.frame, frame)
    pd.testing.assert_frame_equal(expanding.frame, frame)
    assert resolve_training_window(None) == TW_EXPANDING


def test_duplicate_segment_target_dates_are_rejected() -> None:
    frame = pd.concat([segment_frame(2), segment_frame(2).iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        apply_training_window(frame, TW_26)


def test_selection_uses_hierarchy_then_longer_window_tolerance() -> None:
    rows = []
    for index, candidate in enumerate(TRAINING_WINDOW_CANDIDATES):
        rows.append(
            {
                "training_window_id": candidate.training_window_id,
                "development_macro_s1_s5_mae": 10.5,
                "development_recent_tail_mae": 10 + index,
                "development_s2_mae": 11 + index,
                "development_p90_absolute_error": 20 + index,
                "development_mean_signed_error": index - 2,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[frame["training_window_id"] == "TW_52", "development_macro_s1_s5_mae"] = 10.0
    frame.loc[frame["training_window_id"] == "TW_78", "development_macro_s1_s5_mae"] = 10.08
    selected, audit = select_training_window(frame)
    assert selected == "TW_78"
    assert audit.loc[audit["training_window_id"] == "TW_78", "selected"].item()


def test_selection_contract_rejects_confirmation_only_or_missing_candidates() -> None:
    columns = {
        "training_window_id": ["TW_EXPANDING"],
        "development_macro_s1_s5_mae": [1.0],
        "development_recent_tail_mae": [1.0],
        "development_s2_mae": [1.0],
        "development_p90_absolute_error": [1.0],
        "development_mean_signed_error": [0.0],
        "confirmation_mae": [-999.0],
    }
    with pytest.raises(ValueError, match="each locked candidate"):
        select_training_window(pd.DataFrame(columns))

