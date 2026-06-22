# Offline model comparison: `ny_12550`

## Result

The best model by MAE was **Random Forest** at **13.31 visitors**.

Did a stronger model improve meaningfully? Yes. **Random Forest** reduced MAE by 27.5% versus the baseline.

## Method

- Data: 321 engineered attendance rows from 2023-01-14 through 2026-02-15.
- Evaluation: 5 expanding-window, chronological backtest folds; the first 104 rows formed the initial training window.
- Leakage control: every test block occurs strictly after its training window, and missing-value preprocessing is fit on training rows only.
- Baseline: mean of prior attendance for the matching Saturday/Sunday day type, with the overall prior mean as fallback.
- Features (21): year_num, month_num, day_num, weekday_num, weekofyear, is_weekend, month_sin, month_cos, slot_num, is_sun, lag1, lag2, lag3, rolling_mean_3, rolling_std_3, lag_same_daytype_1, rolling_mean_daytype_3, rolling_std_daytype_3, lag_same_slot_1, rolling_mean_slot_3, rolling_std_slot_3.
- Meaningful improvement threshold: at least 5% lower MAE than the historical-average baseline.
- Models skipped as incompatible: None.

## Metrics

| rank | model | mae | rmse | mape_percent | bias | improvement_vs_baseline_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Random Forest | 13.31 | 16.80 | 9.97 | -1.67 | 27.54 |
| 2 | Extra Trees | 13.62 | 16.90 | 10.19 | -2.03 | 25.88 |
| 3 | Gradient Boosting | 14.35 | 18.23 | 10.63 | -2.04 | 21.88 |
| 4 | HistGradientBoostingRegressor | 14.35 | 18.06 | 10.72 | -1.88 | 21.87 |
| 5 | Ridge Regression | 15.04 | 19.71 | 11.67 | 6.14 | 18.14 |
| 6 | Historical Average | 18.37 | 22.67 | 13.13 | -4.88 | 0.00 |

This is an offline experiment. It does not train, replace, or write any production model or artifact.
