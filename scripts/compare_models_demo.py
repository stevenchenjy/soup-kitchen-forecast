from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_admin
from src.config import DATE_COL, TARGET_COL, location_db_file
from src.data_processing import add_basic_calendar_features
from src.features import add_lag_features, build_training_frame


OUTPUT_DIR = ROOT / "experiments" / "model_comparison"
MEANINGFUL_IMPROVEMENT_PERCENT = 5.0


def _readonly_connect(location_id: str) -> sqlite3.Connection:
    """Open the tracked attendance database without schema/write side effects."""
    database = location_db_file(location_id).resolve()
    if not database.exists():
        raise FileNotFoundError(f"No local attendance database found for '{location_id}': {database}")
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def load_local_clean_data(location_id: str) -> pd.DataFrame:
    """Use the project loader while forcing its local, read-only code path."""
    with (
        patch.object(data_admin, "_supabase_config", return_value=None),
        patch.object(data_admin, "_connect", side_effect=_readonly_connect),
    ):
        return data_admin.load_clean_data(location_id)


def build_leakage_safe_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reuse project features but leave missing-value fitting inside each fold."""
    canonical_bundle = build_training_frame(raw_df)
    frame = add_basic_calendar_features(raw_df)
    frame = add_lag_features(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[TARGET_COL, "lag1", "lag2", "lag3"])
    frame = frame.sort_values(DATE_COL).reset_index(drop=True)
    return frame, canonical_bundle.feature_cols


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "Random Forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Extra Trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Gradient Boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=3,
                loss="huber",
                random_state=42,
            ),
        ),
        "Ridge Regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "HistGradientBoostingRegressor": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def expanding_window_splits(
    row_count: int, min_train_size: int, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if min_train_size < 2:
        raise ValueError("--min-train-size must be at least 2.")
    remaining = row_count - min_train_size
    if remaining < n_splits:
        raise ValueError(
            f"Need at least {min_train_size + n_splits} engineered rows for "
            f"{n_splits} backtest folds; found {row_count}."
        )

    test_blocks = np.array_split(np.arange(min_train_size, row_count), n_splits)
    return [(np.arange(block[0]), block) for block in test_blocks]


def _historical_average_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    overall_mean = float(train[TARGET_COL].mean())
    if "is_sun" not in train.columns:
        return np.full(len(test), overall_mean)

    daytype_means = train.groupby("is_sun")[TARGET_COL].mean().to_dict()
    return test["is_sun"].map(daytype_means).fillna(overall_mean).to_numpy(dtype=float)


def run_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str],
    min_train_size: int,
    n_splits: int,
) -> tuple[pd.DataFrame, list[str]]:
    splits = expanding_window_splits(len(frame), min_train_size, n_splits)
    factories = model_factories()
    prediction_rows: list[dict] = []
    skipped_models: list[str] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        fold_predictions: dict[str, np.ndarray] = {
            "Historical Average": _historical_average_predictions(train, test)
        }

        for model_name, factory in factories.items():
            if model_name in skipped_models:
                continue
            try:
                model = factory()
                model.fit(train[feature_cols], train[TARGET_COL])
                fold_predictions[model_name] = model.predict(test[feature_cols])
            except (ImportError, TypeError, ValueError) as exc:
                skipped_models.append(model_name)
                print(f"Skipping incompatible model {model_name}: {exc}", file=sys.stderr)

        train_end_date = pd.to_datetime(train[DATE_COL].iloc[-1]).date().isoformat()
        for model_name, predictions in fold_predictions.items():
            for row_position, prediction in enumerate(predictions):
                test_row = test.iloc[row_position]
                actual = float(test_row[TARGET_COL])
                prediction_rows.append(
                    {
                        DATE_COL: pd.to_datetime(test_row[DATE_COL]).date().isoformat(),
                        "fold": fold,
                        "train_end_date": train_end_date,
                        "train_size": len(train),
                        "model": model_name,
                        "actual": actual,
                        "prediction": float(prediction),
                        "absolute_error": abs(actual - float(prediction)),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    return predictions.sort_values(["model", DATE_COL]).reset_index(drop=True), skipped_models


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model_predictions in predictions.groupby("model", sort=False):
        actual = model_predictions["actual"]
        predicted = model_predictions["prediction"]
        nonzero = actual != 0
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        rows.append(
            {
                "model": model_name,
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mape_percent": mape,
                "bias": float(np.mean(predicted - actual)),
                "r2": float(r2_score(actual, predicted)),
                "n_predictions": len(model_predictions),
            }
        )

    metrics = pd.DataFrame(rows)
    baseline_mae = float(metrics.loc[metrics["model"] == "Historical Average", "mae"].iloc[0])
    metrics["improvement_vs_baseline_percent"] = (baseline_mae - metrics["mae"]) / baseline_mae * 100
    metrics = metrics.sort_values(["mae", "model"]).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    return metrics


def write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    ordered = metrics.sort_values("mae", ascending=True)
    colors = ["tab:green" if rank == 1 else "tab:blue" for rank in ordered["rank"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ordered["model"], ordered["mae"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Model comparison: time-based backtest MAE")
    ax.set_xlabel("Mean absolute error (visitors; lower is better)")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(ordered["mae"]):
        ax.text(value, index, f" {value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_mae.png", dpi=160)
    plt.close(fig)

    best_model = str(metrics.iloc[0]["model"])
    best = predictions[predictions["model"] == best_model].copy()
    best[DATE_COL] = pd.to_datetime(best[DATE_COL])
    best = best.sort_values(DATE_COL)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(best[DATE_COL], best["actual"], label="Actual", color="black", linewidth=1.8)
    ax.plot(best[DATE_COL], best["prediction"], label=best_model, color="tab:green", linewidth=1.5)
    ax.set_title(f"Actual vs predicted: {best_model}")
    ax.set_xlabel("Service date")
    ax.set_ylabel("Visitors")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_pred_best_model.png", dpi=160)
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    frame: pd.DataFrame,
    feature_cols: list[str],
    min_train_size: int,
    n_splits: int,
    skipped_models: list[str],
    output_dir: Path,
) -> None:
    best = metrics.iloc[0]
    best_stronger = metrics[metrics["model"] != "Historical Average"].iloc[0]
    meaningful = best_stronger["improvement_vs_baseline_percent"] >= MEANINGFUL_IMPROVEMENT_PERCENT
    conclusion = (
        f"Yes. **{best_stronger['model']}** reduced MAE by "
        f"{best_stronger['improvement_vs_baseline_percent']:.1f}% versus the baseline."
        if meaningful
        else f"No. The best stronger model, **{best_stronger['model']}**, changed MAE by "
        f"{best_stronger['improvement_vs_baseline_percent']:.1f}% versus the baseline, below the "
        f"{MEANINGFUL_IMPROVEMENT_PERCENT:.0f}% experiment threshold."
    )
    display_metrics = metrics[
        ["rank", "model", "mae", "rmse", "mape_percent", "bias", "improvement_vs_baseline_percent"]
    ].copy()
    for column in ["mae", "rmse", "mape_percent", "bias", "improvement_vs_baseline_percent"]:
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.2f}")

    headers = [str(column) for column in display_metrics.columns]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display_metrics.itertuples(index=False, name=None)
    )
    metrics_table = "\n".join(markdown_rows)

    skipped_text = ", ".join(skipped_models) if skipped_models else "None"
    summary = f"""# Offline model comparison: `{argsafe(frame.attrs.get('location_id', 'unknown'))}`

## Result

The best model by MAE was **{best['model']}** at **{best['mae']:.2f} visitors**.

Did a stronger model improve meaningfully? {conclusion}

## Method

- Data: {len(frame)} engineered attendance rows from {pd.to_datetime(frame[DATE_COL]).min().date()} through {pd.to_datetime(frame[DATE_COL]).max().date()}.
- Evaluation: {n_splits} expanding-window, chronological backtest folds; the first {min_train_size} rows formed the initial training window.
- Leakage control: every test block occurs strictly after its training window, and missing-value preprocessing is fit on training rows only.
- Baseline: mean of prior attendance for the matching Saturday/Sunday day type, with the overall prior mean as fallback.
- Features ({len(feature_cols)}): {', '.join(feature_cols)}.
- Meaningful improvement threshold: at least {MEANINGFUL_IMPROVEMENT_PERCENT:.0f}% lower MAE than the historical-average baseline.
- Models skipped as incompatible: {skipped_text}.

## Metrics

{metrics_table}

This is an offline experiment. It does not train, replace, or write any production model or artifact.
"""
    (output_dir / "model_comparison_summary.md").write_text(summary, encoding="utf-8")


def argsafe(value: object) -> str:
    return str(value).replace("`", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local regression models with time-based backtesting.")
    parser.add_argument("--location", required=True, help="Location ID, for example ny_12550")
    parser.add_argument("--min-train-size", type=int, default=104)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_df = load_local_clean_data(args.location)
    if raw_df.empty:
        raise ValueError(f"No local historical data available for location '{args.location}'.")

    frame, feature_cols = build_leakage_safe_frame(raw_df)
    frame.attrs["location_id"] = args.location
    predictions, skipped_models = run_backtest(
        frame,
        feature_cols,
        min_train_size=args.min_train_size,
        n_splits=args.n_splits,
    )
    metrics = calculate_metrics(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(OUTPUT_DIR / "model_comparison_predictions.csv", index=False, float_format="%.6f")
    write_plots(metrics, predictions, OUTPUT_DIR)
    write_summary(
        metrics,
        frame,
        feature_cols,
        args.min_train_size,
        args.n_splits,
        skipped_models,
        OUTPUT_DIR,
    )

    best = metrics.iloc[0]
    print(f"Compared {len(metrics)} models for {args.location} with {len(predictions) // len(metrics)} test rows each.")
    print(f"Best model by MAE: {best['model']} ({best['mae']:.2f} visitors)")
    print(f"Outputs written to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
