# Weather and attendance model audit — September 13, 2026

**Recommendation: consider weather as a candidate improvement, with priority on snowfall and travel disruption. Keep the current production model until an evaluation using forecasts available before service demonstrates a useful benefit. The evidence does not support a blanket attendance reduction on extreme-weather days.**

The current model ignores weather. In a new offline comparison, adding historical 11 a.m.–1 p.m. weather reduced mean absolute error by 0.49 visitors (3.7%), but uncertainty includes no improvement. Explicit adverse-weather flags contributed essentially no additional accuracy. Errors on snowfall days remain large.

## Scope and evidence

This audit covers all recorded service days for Newburgh, NY 12550; its results do not establish performance at other locations. The intended operational forecast cutoff was not specified, so the new experiment follows the existing previous-recorded-service backtest. Day-before, Friday-for-Sunday, and morning-of-service forecasts still require separate evaluation.

| Evidence | Scope | Appropriate interpretation |
|---|---|---|
| Active model/code inspection | September 7 package; history through September 6, 2026 | What this workspace currently predicts |
| Old reference predictions | 322 unique service dates through July 12, 2026 | Historical error diagnosis, with older attendance labels |
| New paired experiment | 168 service dates, January 4, 2025–September 6, 2026 | Exploratory usefulness of realized weather |
| Weather source | 32,280 hourly timestamps, January 1, 2023–September 6, 2026 | Gridded historical estimates; not on-site measurements or issue-time forecasts |

The active package contains 375 attendance rows, all on weekends, with no duplicate dates or missing/nonpositive attendance values. Its embedded backtest reports MAE 13.53 visitors and RMSE 17.61 across 338 folds. These metrics cover a different period from the 168-date comparison below.

## What the current prediction actually uses

The active package is `ny_12550_f6_nightly_2026-09-06_7cdff7ba0b22_r34125933626a1_v1`, schema 2. Its F6 contract has 33 calendar, recent-attendance, monthly-slot, and forecast-horizon features. It explicitly sets `weather_policy: W0_no_weather` in [the contract](../config/model_contracts/f6_v1.json).

[The predictor](../src/predictor.py) returns the F6 feature row before reaching the weather client. With date and attendance history held fixed, discovering snow, heat, rain, or wind cannot change the current output. An in-memory check replaced the weather forecast method with an exception: three predictions still succeeded and made zero weather calls.

The point model is a 400-tree random forest; the meal recommendation comes from a separate 80th-percentile gradient boosting model, rounded upward. The current recommendation is `ceil(raw Q80)`. Adding a column or changing a weather constant cannot teach these fitted models a weather response. A weather candidate needs retraining and a new feature contract.

## The requested weather window needs precise definitions

[The legacy weather aggregator](../src/weather.py) selects timestamps 10:00, 11:00, 12:00, and 13:00. It averages temperature, apparent temperature, humidity, and wind, and sums precipitation. It retains neither raw hours nor snowfall, snow cover, gust maxima, forecast issue times, or completeness diagnostics.

Open-Meteo timestamps temperature at the stated hour, while precipitation and snowfall describe the preceding hour; gusts are preceding-hour maxima. Thus, summing precipitation stamped 10 through 13 represents approximately **9 a.m.–1 p.m.** For the requested **11 a.m.–1 p.m. interval**, this audit sums the precipitation/snowfall values stamped **12:00 and 13:00**, takes gust maxima over those two intervals, and summarizes instantaneous conditions at **11:00, 12:00, and 13:00**, in America/New_York. [Official hourly definitions](https://open-meteo.com/en/docs).

Weather was requested for Newburgh at 41.50343, −74.01042. The returned grid point was 41.51142, −74.04898. The archive used default Best Match, whose documented sources include IFS, ERA5, and ERA5-Land. Local precipitation and snow conditions can differ from gridded estimates. [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

## New paired model comparison

All variants use identical attendance labels, dates, expanding training windows, separate Saturday/Sunday segments, fold-fitted median imputation, and the production estimator settings. Earlier attendance alone supplies attendance features. Weather variants additionally use the **realized target-day weather**, including at each evaluation date. This is deliberately a hindsight diagnostic.

The six added continuous fields are apparent-temperature minimum/maximum, precipitation total, maximum gust, snowfall total, and maximum snow depth. Five optional flags were fixed before fitting: apparent temperature ≥32°C, apparent temperature ≤−10°C, precipitation ≥5 mm in two hours, gust ≥40 km/h, and snowfall >0 or snow cover ≥1 cm. These are exploratory screening thresholds, not official severe-weather definitions. They mark 87 of 168 evaluation days and should not be described as 87 extreme-weather events.

| Variant | MAE, visitors ↓ | RMSE ↓ | Average prediction minus attendance | Q80 pinball loss ↓ |
|---|---:|---:|---:|---:|
| F6 baseline | 13.382 | 17.424 | +1.816 | 4.310 |
| F6 + 11–13 weather | 12.893 | 17.167 | +2.676 | 4.200 |
| F6 + 11–13 weather + flags | 12.895 | 17.144 | +2.652 | 4.213 |
| F6 + same six weather summaries over old timestamps | 12.997 | 17.175 | +2.763 | 4.192 |

The final row isolates a broader timestamp window; it does not reproduce the old five-feature weather model.

The 11–13 weather improvement is **0.489 visitors per prediction**. A paired bootstrap resampling 85 whole service weekends gives a 95% interval of **−0.251 to +1.205 visitors of improvement**. Resampling 21 calendar months gives **−0.102 to +1.075**. Both include zero. These are exploratory uncertainty estimates conditional on the chosen features and historical predictions, without adjustment for multiple comparisons or model-selection uncertainty.

Adding the flags to continuous weather changes MAE by approximately **+0.001 visitors**, effectively no difference. Its paired weekend interval for improvement is −0.049 to +0.045. Restricting the window to 11–13 improves MAE by only 0.104 relative to the wider-window candidate; that interval also includes zero.

Weather changed individual point predictions by an average absolute 3.29 visitors; 33 of 168 moved at least five visitors. Changing a prediction is different from improving its accuracy.

![Weather comparison and uncertainty](../artifacts/ny_12550/weather_audit_2026-09-13/weather_comparison.png)

| Slice | Dates | F6 MAE | With 11–13 weather | Implication |
|---|---:|---:|---:|---|
| Saturday | 85 | 13.022 | 12.203 | Larger estimated gain |
| Sunday | 83 | 13.751 | 13.601 | Small estimated gain |
| 2025 | 104 | 13.337 | 12.981 | Small gain |
| 2026 through September 6 | 64 | 13.457 | 12.752 | Larger gain, with increased positive bias |
| Any screening flag | 87 | 14.286 | 14.233 | Almost no average error reduction |
| No screening flag | 81 | 12.412 | 11.454 | Most of the aggregate improvement |
| Falling snow during 11–13 | 6 | 33.937 | 31.751 | Large misses persist; very few events |

The falling-snow subset has baseline average overprediction of **33.00 visitors**, falling to **30.01** with weather. For example, January 18, 2026 attendance was 106: the baseline predicted 163.84 and the weather candidate 159.13. This warrants investigation, but the simple added features did not resolve the miss. Falling snow and existing snow cover are distinct: the broader snow-or-cover screen contains 37 test dates.

Only one test date had ≥5 mm precipitation during the exact two-hour window, so this evaluation cannot establish performance on heavy rain. It also cannot establish tornado, lightning, flood, freezing-rain, or other rare-hazard effects. Those conditions are not sufficiently represented or labeled.

## Meal preparation matters as much as point error

The following numbers simulate the existing `ceil(Q80)` recommendation against recorded attendance; they are not measured kitchen waste or actual historical shortages.

| Simulated preparation result, 168 services | F6 | + 11–13 weather | + weather and flags |
|---|---:|---:|---:|
| Days with fewer recommended meals than visitors | 41 | 42 | 41 |
| Total shortfall meals | 317 | 309 | 308 |
| Mean surplus meals per service | 13.85 | 13.40 | 13.50 |

Changes are small and uncertain. For continuous weather, Saturday shortfall totals decrease from 145 to 114, but Sunday totals increase from 172 to 195. A point-MAE improvement therefore does not establish a safer preparation policy. The historical raw Q80 also does not consistently achieve 80% coverage; it needs explicit calibration checks at the intended forecast horizon.

## Data and workspace findings that affect the decision

1. **Missing service status can hide the most disruptive weather.** Ten expected weekend dates are absent from active attendance: April 30, 2023; January 7, 2024; January 25, February 21/22/28, March 1, April 12, and May 23/24, 2026. January 25, 2026 has estimated snowfall of 3.78 cm during 11–13 and apparent temperature around −18.8°C, yet no attendance row. It is unknown whether the kitchen was closed or attendance was unrecorded. Keep missing attendance distinct from zero, and record whether service operated and why it closed. Evaluate attendance conditional on being open separately from closure decisions.

2. **Local data sources are not synchronized.** The local SQLite has 324 attendance records through February 15, 2026; the legacy CSV has 323; the export has 360 through July 12; the active package has 375 through September 6. The active package excludes the export's Tuesday April 14 entry and revises May 31 attendance from 177 to 148. The stored weather cache ends February 15. The fresh hourly retrieval covers all active dates. New experiment labels come from the active package; July diagnostics retain their older labels and are not mixed into the new metrics.

3. **Prior weather results do not settle the F6 question.** Old F0 next-service MAE fell from 13.591 to 13.346 across 315 dates with realized weather. The often cited 1,256 scenario rows represent overlapping predictions for those dates, not independent services. The later [F6 W1 diagnostic](../artifacts/ny_12550/model_optimization/phase2a_feature_repair/14_w1_secondary_diagnostic.md) explicitly kept F6 weather-free; it was an inert-policy check. The new experiment above actually adds weather features.

4. **Issue-time evidence is currently missing.** [Prediction logging](../src/prediction_logs.py) overwrites the prior forecast for a location/date. Exact earlier prediction versions and associated weather forecasts cannot be reconstructed from that store. The predictor also uses date-level origins, clamping same-day targets to the previous day, so an 11 a.m. update requires timestamp-aware weather availability.

5. **A separate API freshness issue exists.** [The API](../src/api.py) caches predictors by location without checking the model file for a nightly replacement. A long-running worker can retain an older package. This does not explain the weather experiment, but should be addressed in a deployment maintenance task.

## Recommended next step

Proceed with a weather candidate evaluated alongside the existing prediction, while keeping production on F6. Prioritize continuous snowfall, snow cover, precipitation intensity, gusts, and apparent temperature. Compare the travel/arrival period before 11 a.m. separately from service time; this audit does not establish that 11–13 is the optimal causal or predictive window.

1. **Choose the preparation cutoff.** Define the actual decision timestamp for each operational use, such as Friday morning for the weekend or service-day morning. At that timestamp, use observations already available plus forecasts of the remaining 11–13 period. Realized future weather must never enter a deployable backtest.
2. **Retain forecasts and predictions without overwriting them.** Save location, issue/availability timestamp, model run, valid hour, timezone, units, weather features, attendance cutoff, package/feature-contract version, prediction, and meal recommendation. Record missing weather and use a tested fallback to F6. Record service status, closure reason, access disruptions, attendance revisions, and relevant special events.
3. **Recover suitable forecast history where available.** Open-Meteo's Historical Forecast API stitches the first hours of successive runs; it does not by itself reconstruct a forecast known on a specific earlier morning. Its Single Runs option is designed for particular initialization times. The Previous Runs API offers fixed 1–7 day offsets, with model-dependent coverage. Check actual publication availability against the kitchen cutoff. [Historical Forecast documentation](https://open-meteo.com/en/docs/historical-forecast-api), [Previous Runs documentation](https://open-meteo.com/en/docs/previous-runs-api).
4. **Evaluate paired forecasts on new dates.** Keep F6 as the comparator. Fix a compact weather feature set and thresholds before testing; retain a later confirmation period, report individual event counts, and group uncertainty by weather event/weekend. Include both Saturday and Sunday and the actual operational leads. Report point error, bias, Q80 calibration, shortage frequency/size, and surplus. Avoid selecting a model using a single rare event or the best of many thresholds.
5. **Promote only after useful operational gains are demonstrated.** Require a practically meaningful improvement with uncertainty assessed, stable behavior across service types, and acceptable shortfall outcomes. Agree the kitchen's shortage-versus-surplus priorities before selecting the recommendation policy. Version the expanded feature contract and validate fallback/parity; do not modify the locked F6 feature list in place.

The current data justify collecting and testing weather, especially snow and access disruption. They do not justify a fixed “subtract 25 or 33 visitors” rule, a universal extreme-weather flag, or an immediate production change.

## Material Passport and reproducibility

Status: **ANALYZED**. Offline experiments were executed and independently reviewed; these are new exploratory results, not a replicated confirmatory study. All attendance analysis stayed local. Production code, configuration, databases, and model packages were left unchanged.

The 2025+ period was fixed before this audit's model fitting, but overlaps prior F6 feature-selection work. Historical attendance revisions and availability timestamps are not fully recoverable. Weather is hindsight information. Results are conditional on recorded service dates, one kitchen, the selected features, and the existing estimator families.

Eleven statistical pitfalls were checked:

| Check | Assessment |
|---|---|
| Simpson's paradox | Year and Saturday/Sunday slices reported; gains differ by segment. No claim that the pooled result holds for each segment. |
| Ecological inference | Conclusions concern service-day visitor counts, not individual visitor behavior. |
| Selection/Berkson bias | Recorded services are selected; absent/closed days may differ systematically. |
| Collider bias | No causal adjustment claim; selection on an operating/recorded service remains a limitation. |
| Base rates | Event counts included; one heavy-rain and six falling-snow test dates are insufficient for general hazard conclusions. |
| Regression to the mean | Paired fixed-date model comparisons used; no before/after claim based on selecting worst errors. |
| Survivorship bias | Ten missing expected service dates identified, without imputing attendance zero. |
| Look-elsewhere effect | All four fixed variants and adverse/ordinary results reported; no selective significant-result claim. |
| Forking paths | Features/thresholds fixed before fitting in this audit; prior F6 selection and exploratory subgroup analysis disclosed. |
| Correlation versus causation | Weather associations and forecasting gains are not causal effects. |
| Reverse causality | Attendance does not plausibly cause regional weather; timing/availability leakage is the relevant predictive threat. |

Verification: 40 existing tests plus nine subtests passed across production F6 integration, origin features, and forecast validation. Independent review checked fold cutoffs, imputation, paired comparisons, interval alignment, and integer meal calculations. Hourly data contain the required window records without duplicates or nulls.

Saved local analysis: [artifact directory](../artifacts/ny_12550/weather_audit_2026-09-13/). Key files are [methodology](../artifacts/ny_12550/weather_audit_2026-09-13/methodology.json), [input hashes](../artifacts/ny_12550/weather_audit_2026-09-13/audit_manifest.json), [paired predictions](../artifacts/ny_12550/weather_audit_2026-09-13/paired_predictions.csv), [metrics](../artifacts/ny_12550/weather_audit_2026-09-13/paired_metrics.csv), [confidence intervals](../artifacts/ny_12550/weather_audit_2026-09-13/paired_mae_gain_intervals.csv), and [meal outcomes](../artifacts/ny_12550/weather_audit_2026-09-13/paired_operational_metrics.csv). The hourly response, request metadata, engineered features, scripts, and run log are also retained there. This generated directory is ignored by Git.

The copied scripts use their artifact directory for outputs. To rerun against the same source/package hashes:

```bash
.venv/bin/python artifacts/ny_12550/weather_audit_2026-09-13/paired_weather_backtest.py
.venv/bin/python artifacts/ny_12550/weather_audit_2026-09-13/paired_operational_metrics.py
```

Future package or code changes require matching the saved input hashes or treating the rerun as a new experiment. Runtime/library versions and exact estimator parameters are recorded in the methodology file.
