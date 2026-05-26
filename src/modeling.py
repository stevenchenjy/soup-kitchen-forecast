from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from src.config import DATE_COL, TARGET_COL


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict



def make_point_model(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_state,
    )



def make_quantile_model(quantile: float = 0.8, random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_depth=4,
        max_iter=500,
        random_state=random_state,
    )



def _calc_metrics(pred_df: pd.DataFrame) -> dict:
    return {
        "MAE": float(mean_absolute_error(pred_df["actual"], pred_df["pred"])),
        "RMSE": float(root_mean_squared_error(pred_df["actual"], pred_df["pred"])),
        "MAPE": float(mean_absolute_percentage_error(pred_df["actual"], pred_df["pred"])),
        "Bias": float(np.mean(pred_df["pred"] - pred_df["actual"])),
        "P90AbsError": float(np.quantile(pred_df["abs_error"], 0.9)),
        "BacktestRows": int(len(pred_df)),
    }



def rolling_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_size: int = 36,
    quantile: float = 0.8,
) -> BacktestResult:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    rows = []

    for i in range(min_train_size, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i : i + 1]

        point_model = make_point_model()
        q_model = make_quantile_model(quantile=quantile)

        point_model.fit(train[feature_cols], train[TARGET_COL])
        q_model.fit(train[feature_cols], train[TARGET_COL])

        pred = float(point_model.predict(test[feature_cols])[0])
        pred_q = float(q_model.predict(test[feature_cols])[0])
        actual = float(test[TARGET_COL].iloc[0])

        rows.append(
            {
                DATE_COL: test[DATE_COL].iloc[0],
                "actual": actual,
                "pred": pred,
                "pred_q": pred_q,
                "abs_error": abs(actual - pred),
                "is_sun": int(test["is_sun"].iloc[0]) if "is_sun" in test.columns else None,
            }
        )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError("Backtest has no rows. Reduce min_train_size or provide more data.")

    metrics = _calc_metrics(pred_df)
    return BacktestResult(predictions=pred_df, metrics=metrics)



def rolling_backtest_by_daytype(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_size: int = 18,
    quantile: float = 0.8,
) -> dict:
    if "is_sun" not in df.columns:
        raise ValueError("Column 'is_sun' is required for daytype split backtest")

    out = {}
    all_preds = []
    for key, flag in [("sat", 0), ("sun", 1)]:
        part = df[df["is_sun"] == flag].sort_values(DATE_COL).reset_index(drop=True)
        if len(part) <= min_train_size:
            out[key] = {"metrics": {"BacktestRows": 0}, "predictions": pd.DataFrame()}
            continue
        bt = rolling_backtest(part, feature_cols, min_train_size=min_train_size, quantile=quantile)
        out[key] = {"metrics": bt.metrics, "predictions": bt.predictions}
        all_preds.append(bt.predictions)

    if all_preds:
        merged = pd.concat(all_preds, ignore_index=True).sort_values(DATE_COL)
        out["overall"] = {"metrics": _calc_metrics(merged), "predictions": merged}
    else:
        out["overall"] = {"metrics": {"BacktestRows": 0}, "predictions": pd.DataFrame()}

    return out



def fit_final_models_by_daytype(
    df: pd.DataFrame,
    feature_cols: list[str],
    quantile: float = 0.8,
) -> tuple[dict, dict]:
    models = {}
    q_models = {}
    for key, flag in [("sat", 0), ("sun", 1)]:
        part = df[df["is_sun"] == flag].sort_values(DATE_COL).reset_index(drop=True)
        if part.empty:
            continue

        m = make_point_model()
        qm = make_quantile_model(quantile=quantile)
        m.fit(part[feature_cols], part[TARGET_COL])
        qm.fit(part[feature_cols], part[TARGET_COL])
        models[key] = m
        q_models[key] = qm

    return models, q_models
