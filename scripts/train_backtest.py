from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATE_COL, TARGET_COL, artifact_dir_for_location, location_weather_file, model_file_for_location
from src.data_admin import bootstrap_location_from_csv, load_clean_data
from src.features import build_training_frame
from src.location_config import get_location
from src.modeling import fit_final_models_by_daytype, rolling_backtest_by_daytype
from src.weather import build_weather_dataset


def _load_or_build_weather(location_id: str, service_dates: pd.Series) -> pd.DataFrame | None:
    location = get_location(location_id)
    weather_path = location_weather_file(location_id)

    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        return weather if not weather.empty else None

    try:
        return build_weather_dataset(
            service_dates,
            out_csv=weather_path,
            zip_code=location.zip_code,
            country=location.country_code,
            timezone=location.timezone,
        )
    except Exception as exc:
        print(f"Warning: weather fetch failed for {location_id}; training without weather features: {exc}")
        return None


def _write_predictions(outputs: dict, artifact_dir: Path) -> pd.DataFrame:
    overall = outputs.get("overall", {}).get("predictions", pd.DataFrame())
    if not overall.empty:
        overall.to_csv(artifact_dir / "backtest_predictions.csv", index=False)

    for key in ["sat", "sun"]:
        preds = outputs.get(key, {}).get("predictions", pd.DataFrame())
        if not preds.empty:
            preds.to_csv(artifact_dir / f"backtest_predictions_{key}.csv", index=False)

    return overall


def _write_metrics(outputs: dict, artifact_dir: Path) -> dict:
    metrics = {
        key: outputs.get(key, {}).get("metrics", {"BacktestRows": 0})
        for key in ["overall", "sat", "sun"]
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _write_plots(predictions: pd.DataFrame, artifact_dir: Path) -> None:
    if predictions.empty:
        return

    plot_df = predictions.copy()
    plot_df[DATE_COL] = pd.to_datetime(plot_df[DATE_COL])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df[DATE_COL], plot_df["actual"], label="Actual", linewidth=2)
    ax.plot(plot_df[DATE_COL], plot_df["pred"], label="Predicted", linewidth=2)
    if "pred_q" in plot_df.columns:
        ax.plot(plot_df[DATE_COL], plot_df["pred_q"], label="Predicted quantile", linestyle="--")
    ax.set_title("Backtest actual vs predicted")
    ax.set_xlabel("Service date")
    ax.set_ylabel("Visitors")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(artifact_dir / "backtest_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df[DATE_COL], plot_df["abs_error"], label="Absolute error", color="tab:red")
    ax.set_title("Backtest absolute error")
    ax.set_xlabel("Service date")
    ax.set_ylabel("Visitors")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(artifact_dir / "backtest_abs_error.png", dpi=150)
    plt.close(fig)


def _residual_buffer(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return 0.0
    under_prediction = np.maximum(predictions["actual"] - predictions["pred"], 0)
    return float(np.quantile(under_prediction, 0.8))


def train_location(location_id: str, min_train_size: int = 18, quantile: float = 0.8) -> Path:
    bootstrap_location_from_csv(location_id)
    raw_df = load_clean_data(location_id)
    if raw_df.empty:
        raise ValueError(f"No historical data available for location '{location_id}'.")

    weather_df = _load_or_build_weather(location_id, raw_df[DATE_COL])
    bundle = build_training_frame(raw_df, weather_df)
    if bundle.df.empty:
        raise ValueError(f"Not enough historical data to train location '{location_id}'.")

    outputs = rolling_backtest_by_daytype(
        bundle.df,
        bundle.feature_cols,
        min_train_size=min_train_size,
        quantile=quantile,
    )
    models, quantile_models = fit_final_models_by_daytype(bundle.df, bundle.feature_cols, quantile=quantile)
    if not models:
        raise ValueError(f"No final models were trained for location '{location_id}'.")

    artifact_dir = artifact_dir_for_location(location_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions = _write_predictions(outputs, artifact_dir)
    metrics = _write_metrics(outputs, artifact_dir)
    _write_plots(predictions, artifact_dir)

    location = get_location(location_id)
    residual_buffer_by_day = {
        key: _residual_buffer(outputs.get(key, {}).get("predictions", pd.DataFrame()))
        for key in ["sat", "sun"]
    }
    model_path = model_file_for_location(location_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "quantile_models": quantile_models,
            "feature_cols": bundle.feature_cols,
            "history_df": bundle.df.copy(),
            "default_meal_buffer_pct": 0.08,
            "residual_buffer_by_day": residual_buffer_by_day,
            "weather_context": {
                "zip_code": location.zip_code,
                "country_code": location.country_code,
                "timezone": location.timezone,
            },
            "metrics": metrics,
        },
        model_path,
    )
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and backtest one location.")
    parser.add_argument("--location", required=True, help="Location ID from data/locations.json")
    parser.add_argument("--min-train-size", type=int, default=18)
    parser.add_argument("--quantile", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = train_location(
        location_id=args.location,
        min_train_size=args.min_train_size,
        quantile=args.quantile,
    )
    print(f"Training completed for {args.location}: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
