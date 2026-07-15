"""Centralized Phase 2B1 training-window definitions and selection logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd

from src.config import DATE_COL


@dataclass(frozen=True)
class TrainingWindowDefinition:
    """One segment-level model-fitting window candidate."""

    training_window_id: str
    configured_window_rows: int | None
    approximate_years: float | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingWindowResult:
    """Diagnostics returned with a retained segment training frame."""

    frame: pd.DataFrame
    available_segment_training_rows: int
    retained_segment_training_rows: int
    window_constrained: bool
    retained_training_start_date: pd.Timestamp | None
    retained_training_end_date: pd.Timestamp | None
    effective_window_days: int | None
    effective_window_years: float | None


TW_EXPANDING = TrainingWindowDefinition(
    "TW_EXPANDING",
    None,
    None,
    "All origin-available segment training examples after weekday-policy filtering.",
)
TW_104 = TrainingWindowDefinition("TW_104", 104, 2.0, "Latest 104 segment training examples.")
TW_78 = TrainingWindowDefinition("TW_78", 78, 1.5, "Latest 78 segment training examples.")
TW_52 = TrainingWindowDefinition("TW_52", 52, 1.0, "Latest 52 segment training examples.")
TW_26 = TrainingWindowDefinition("TW_26", 26, 0.5, "Latest 26 segment training examples.")

TRAINING_WINDOW_CANDIDATES: tuple[TrainingWindowDefinition, ...] = (
    TW_EXPANDING,
    TW_104,
    TW_78,
    TW_52,
    TW_26,
)
TRAINING_WINDOWS_BY_ID = {
    item.training_window_id: item for item in TRAINING_WINDOW_CANDIDATES
}

if len(TRAINING_WINDOWS_BY_ID) != len(TRAINING_WINDOW_CANDIDATES):
    raise AssertionError("Training-window candidate identifiers must be unique")


def resolve_training_window(
    value: TrainingWindowDefinition | str | None,
) -> TrainingWindowDefinition:
    """Normalize an optional public API value; None preserves expanding behavior."""

    if value is None:
        return TW_EXPANDING
    if isinstance(value, TrainingWindowDefinition):
        if value.training_window_id not in TRAINING_WINDOWS_BY_ID:
            raise ValueError(f"Unsupported training-window definition: {value.training_window_id}")
        canonical = TRAINING_WINDOWS_BY_ID[value.training_window_id]
        if value != canonical:
            raise ValueError(f"Non-canonical training-window definition: {value.training_window_id}")
        return value
    try:
        return TRAINING_WINDOWS_BY_ID[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported training-window id: {value}") from exc


def apply_training_window(
    segment_training_frame: pd.DataFrame,
    definition: TrainingWindowDefinition | str | None,
    *,
    date_col: str = DATE_COL,
) -> TrainingWindowResult:
    """Retain the latest N origin-available examples within one model segment.

    The caller is responsible for applying weekday-policy filtering, origin
    availability, and segment assignment before this function. Feature construction
    is deliberately outside this operation and therefore retains full origin history.
    """

    resolved = resolve_training_window(definition)
    if date_col not in segment_training_frame.columns:
        raise ValueError(f"Training frame does not contain date column {date_col!r}")
    frame = segment_training_frame.copy()
    if not frame.empty:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
        if frame[date_col].duplicated().any():
            raise ValueError("Segment training frame contains duplicate target dates")
        frame = frame.sort_values(date_col, kind="stable").reset_index(drop=True)
    available = int(len(frame))
    limit = resolved.configured_window_rows
    constrained = bool(limit is not None and available > limit)
    retained = frame if limit is None or available <= limit else frame.tail(limit).reset_index(drop=True)
    retained_count = int(len(retained))
    if retained.empty:
        start = end = None
        days = None
        years = None
    else:
        start = pd.Timestamp(retained[date_col].iloc[0]).normalize()
        end = pd.Timestamp(retained[date_col].iloc[-1]).normalize()
        days = int((end - start).days)
        years = float(days / 365.2425)
    return TrainingWindowResult(
        frame=retained,
        available_segment_training_rows=available,
        retained_segment_training_rows=retained_count,
        window_constrained=constrained,
        retained_training_start_date=start,
        retained_training_end_date=end,
        effective_window_days=days,
        effective_window_years=years,
    )


def registry_rows() -> list[dict[str, Any]]:
    return [item.to_dict() for item in TRAINING_WINDOW_CANDIDATES]


def select_training_window(
    development_rows: pd.DataFrame,
    *,
    macro_tolerance: float = 0.10,
) -> tuple[str, pd.DataFrame]:
    """Apply the locked development-only lexicographic selection policy.

    Required inputs are deliberately limited to candidate identity and development
    metrics. Confirmation values cannot affect this function.
    """

    required = {
        "training_window_id",
        "development_macro_s1_s5_mae",
        "development_recent_tail_mae",
        "development_s2_mae",
        "development_p90_absolute_error",
        "development_mean_signed_error",
    }
    missing = required.difference(development_rows.columns)
    if missing:
        raise ValueError(f"Development selection table is missing columns: {sorted(missing)}")
    table = development_rows[list(required)].copy()
    if set(table["training_window_id"]) != set(TRAINING_WINDOWS_BY_ID):
        raise ValueError("Development selection table must contain each locked candidate exactly once")
    if table["training_window_id"].duplicated().any() or len(table) != len(TRAINING_WINDOWS_BY_ID):
        raise ValueError("Development selection table contains duplicate candidates")
    numeric = sorted(required.difference({"training_window_id"}))
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
    if table[numeric].isna().any().any():
        raise ValueError("Development selection metrics must be complete")

    best_macro = float(table["development_macro_s1_s5_mae"].min())
    table["within_macro_tolerance"] = (
        table["development_macro_s1_s5_mae"] <= best_macro + float(macro_tolerance)
    )
    # A candidate outside tolerance cannot win. Within the tolerance band, the locked
    # preference is the longer window; remaining metrics provide deterministic order
    # for auditing and for the (normally unreachable) equal-length case.
    length_rank = {"TW_EXPANDING": 0, "TW_104": 1, "TW_78": 2, "TW_52": 3, "TW_26": 4}
    table["longer_window_rank"] = table["training_window_id"].map(length_rank).astype(int)
    eligible = table[table["within_macro_tolerance"]].copy()
    eligible["absolute_bias"] = eligible["development_mean_signed_error"].abs()
    eligible = eligible.sort_values(
        [
            "longer_window_rank",
            "development_recent_tail_mae",
            "development_s2_mae",
            "development_p90_absolute_error",
            "absolute_bias",
            "training_window_id",
        ],
        kind="stable",
    )
    selected = str(eligible.iloc[0]["training_window_id"])

    # Also expose the strict hierarchy rank before the tolerance override.
    ranked = table.assign(absolute_bias=table["development_mean_signed_error"].abs()).sort_values(
        [
            "development_macro_s1_s5_mae",
            "development_recent_tail_mae",
            "development_s2_mae",
            "development_p90_absolute_error",
            "absolute_bias",
            "training_window_id",
        ],
        kind="stable",
    )
    strict_rank = {window: rank for rank, window in enumerate(ranked["training_window_id"], 1)}
    table["strict_hierarchy_rank"] = table["training_window_id"].map(strict_rank).astype(int)
    table["selected"] = table["training_window_id"].eq(selected)
    return selected, table.sort_values("longer_window_rank", kind="stable").reset_index(drop=True)


def validate_candidate_ids(candidate_ids: Iterable[str]) -> None:
    if tuple(candidate_ids) != tuple(item.training_window_id for item in TRAINING_WINDOW_CANDIDATES):
        raise ValueError("Candidate order or membership differs from the locked Phase 2B1 registry")
