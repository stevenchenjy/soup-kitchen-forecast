from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL
from src.data_processing import add_basic_calendar_features


@dataclass
class FeatureBundle:
    df: pd.DataFrame
    feature_cols: list[str]



def add_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    out["lag1"] = out[target_col].shift(1)
    out["lag2"] = out[target_col].shift(2)
    out["lag3"] = out[target_col].shift(3)
    out["rolling_mean_3"] = out[target_col].shift(1).rolling(3).mean()
    out["rolling_std_3"] = out[target_col].shift(1).rolling(3).std()
    if "is_sun" in out.columns:
        out["lag_same_daytype_1"] = out.groupby("is_sun")[target_col].shift(1)
        out["rolling_mean_daytype_3"] = out.groupby("is_sun")[target_col].transform(lambda s: s.shift(1).rolling(3).mean())
        out["rolling_std_daytype_3"] = out.groupby("is_sun")[target_col].transform(lambda s: s.shift(1).rolling(3).std())
    if "slot_num" in out.columns:
        out["lag_same_slot_1"] = out.groupby("slot_num")[target_col].shift(1)
        out["rolling_mean_slot_3"] = out.groupby("slot_num")[target_col].transform(lambda s: s.shift(1).rolling(3).mean())
        out["rolling_std_slot_3"] = out.groupby("slot_num")[target_col].transform(lambda s: s.shift(1).rolling(3).std())

    return out


def merge_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = weather_df.copy()
    out["date"] = pd.to_datetime(out[DATE_COL]).dt.date
    w["date"] = pd.to_datetime(w["date"]).dt.date
    out = out.merge(w, on="date", how="left")
    out = out.drop(columns=["date"])
    return out


def build_training_frame(raw_df: pd.DataFrame, weather_df: pd.DataFrame | None = None) -> FeatureBundle:
    df = raw_df.copy()
    df = add_basic_calendar_features(df)
    df = add_lag_features(df)

    if weather_df is not None and not weather_df.empty:
        df = merge_weather_features(df, weather_df)

    numeric_cols = [
        "year_num",
        "month_num",
        "day_num",
        "weekday_num",
        "weekofyear",
        "is_weekend",
        "month_sin",
        "month_cos",
        "slot_num",
        "is_sun",
        "lag1",
        "lag2",
        "lag3",
        "rolling_mean_3",
        "rolling_std_3",
        "lag_same_daytype_1",
        "rolling_mean_daytype_3",
        "rolling_std_daytype_3",
        "lag_same_slot_1",
        "rolling_mean_slot_3",
        "rolling_std_slot_3",
        "temp_10_13",
        "apparent_temp_10_13",
        "humidity_10_13",
        "wind_10_13",
        "precip_10_13",
    ]
    feature_cols = [c for c in numeric_cols if c in df.columns]

    for c in feature_cols:
        df[c] = df[c].fillna(df[c].median())

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL, "lag1", "lag2", "lag3"]).reset_index(drop=True)

    return FeatureBundle(df=df, feature_cols=feature_cols)
