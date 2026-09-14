# Prospective weather comparison alongside F6

The weather candidate is an opt-in comparison model. Staff/admin predictions and meal recommendations continue to use active F6. Running the comparison records F6 and the candidate together, with the exact weather forecast available before the kitchen's preparation cutoff. Nothing in this workflow activates a model automatically.

The initial Newburgh candidate is `models/candidates/ny_12550_weather_shadow_2026-09-13_v1/weather_candidate.joblib`. It adds six fields to the same 33 F6 attendance/calendar features:

| Field | Local 11 a.m.–1 p.m. definition | Unit |
|---|---|---|
| Apparent temperature minimum and maximum | Instantaneous readings at 11:00, 12:00, 13:00 | °C |
| Precipitation total | Preceding-hour values stamped 12:00 and 13:00 | mm |
| Maximum gust | Preceding-hour maxima stamped 12:00 and 13:00 | km/h |
| Snowfall total | Preceding-hour values stamped 12:00 and 13:00 | cm |
| Maximum snow cover depth | Instantaneous readings at 11:00, 12:00, 13:00 | m |

These interval definitions follow the [Open-Meteo hourly API](https://open-meteo.com/en/docs). The capture rejects incomplete hours, incorrect units/timezones, duplicate records, invalid values, and location mismatches. It does not substitute historical weather when a live forecast fails.

## Current status

The candidate is trained on the active F6 attendance snapshot through September 6, 2026, and the historical hourly weather retrieved during the audit. It is labeled `realized_historical_bootstrap` and `shadow_only_not_validated`. This is an initial model for prospective testing, not a candidate already validated on historical forecasts issued before preparation.

The confirmed preparation rule is **12 elapsed hours before 10 a.m. on the service date**, in `America/New_York`. It is saved in [the study configuration](../config/weather_shadow_studies.json). Collection starts five minutes before cutoff, leaving time for network and database requests.

| Service | Normal preparation cutoff | Scheduled capture start |
|---|---|---|
| Saturday | Friday 10 p.m. | Friday 9:55 p.m. |
| Sunday | Saturday 10 p.m. | Saturday 9:55 p.m. |

The calculation subtracts twelve hours in UTC from local 10 a.m. On the spring daylight-saving Sunday, the cutoff is Saturday **9 p.m. EST**, with capture at 8:55 p.m. On the fall transition Sunday, it is Saturday **11 p.m. EDT**, with capture at 10:55 p.m. This preserves twelve elapsed hours instead of fixing the cutoff to a wall-clock hour. The first configured captures after setup are September 18 and 19, 2026, at 9:55 p.m. EDT, for September 19 and 20 service. No prospective service pair has yet been captured at setup.

A local Codex scheduled task, `newburgh-preparation-weather-study`, is active in this conversation. It checks Friday and Saturday at 7:55, 8:55, 9:55, and 10:55 p.m. local time. The first check allows candidate maintenance; the later checks cover the normal and DST capture times. `capture-due` decides whether a forecast can be collected at the actual execution time. A queued or delayed task is not evidence of an on-time capture. The computer must be on and the app running for [local scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app). Routine successful checks remain quiet; missed captures and meaningful failures are reported.

No existing production prediction path, active model, nightly retraining workflow, or attendance table is changed. Shared Supabase storage requires the separate migration below; it has not been applied by this implementation.

## Capture a pair at the real preparation cutoff

Run from the project root. Inspect the computed schedule without requesting weather:

```bash
.venv/bin/python scripts/run_weather_shadow.py schedule --location ny_12550
```

The scheduler entry point only captures between five minutes before the computed cutoff and the cutoff itself. Outside that interval it reports `not_due` without requesting weather or writing a study record. It skips a pair already saved before the same cutoff:

```bash
.venv/bin/python scripts/run_weather_shadow.py capture-due --location ny_12550
```

For a manual capture, supply the actual service date. The cutoff, candidate, study ID, and local store come from the study configuration:

```bash
.venv/bin/python scripts/run_weather_shadow.py capture \
  --location ny_12550 \
  --service-date YYYY-MM-DD
```

An optional explicit `--cutoff-at` must identify exactly the same instant as the configured rule; a different cutoff is rejected. The service date must be Saturday or Sunday within the existing 16-day forecast range. Manual calls more than 15 minutes early or starting after cutoff fail before requesting weather. A missed scheduled capture must be reported as missed; never backfill it with later or historical weather.

Each capture saves a new UUID and retains:

- Preparation cutoff, capture start, forecast request/receipt, prediction generation timestamps, and a database-generated insertion receipt (`persisted_at`).
- Raw hourly weather JSON, all six derived features, explicit units/timezone, request and response-grid coordinates, and a SHA-256 covering the snapshot.
- F6 and candidate package IDs and file hashes, attendance-history fingerprint, attendance cutoff, horizon, and the exact shared raw F6 feature row.
- Both point predictions, raw Q80 predictions, and integer `ceil(Q80)` meal recommendations.
- Failures and whether weather or the candidate was unavailable; these runs retain the F6 result and are excluded from paired accuracy estimates.

The live seamless weather endpoint does not supply one reliable model issue timestamp, so `provider_issued_at` is null. Its recorded request/receipt times establish when this process obtained the forecast; they must not be described as provider issue times.

Records are append-only. Rerunning capture preserves the previous version. Evaluation uses the most recently retrieved eligible pair for each study/location/service/cutoff, so reruns do not count as independent services. A database write completed after cutoff is excluded using its insertion receipt, even if computation finished before cutoff. Local/server clocks must be synchronized for this evidence to be meaningful.

Exit status is 0 for a generated paired record, 2 for a saved weather/candidate/cutoff failure, and 1 for a rejected request or persistence error. A generated pair still needs its database receipt checked at evaluation; a slow write may make it ineligible. Storage errors surface rather than silently changing storage backend.

## Evaluate after attendance is recorded

Supply a CSV with `service_date,visitors`, optionally `location_id,service_status`. The latter may be `open`, `closed`, or `unknown`; a supplied missing/unknown status is excluded. Without a status column, recorded attendance is treated as an open service. Record closures explicitly rather than inventing zero attendance. Duplicate date/location labels must be reconciled first; absent counts remain pending.

```bash
.venv/bin/python scripts/run_weather_shadow.py evaluate \
  --location ny_12550 \
  --attendance-csv path/to/current_attendance.csv \
  --local-store data/weather_shadow/ny_12550.sqlite \
  --output-dir artifacts/ny_12550/weather_shadow_evaluation/run_001
```

The output directory must be new. `evaluation.json`, `metrics.csv`, and `paired_predictions.csv` include eligibility exclusions and the attendance input hash. Scores are reported separately by cutoff schedule, Saturday/Sunday, and both fitted model hashes. Metrics include MAE, RMSE, signed error, Q80 pinball loss/coverage, shortfall days/meals, and surplus per service. Closed services and services that have not yet completed 1 p.m. are excluded. Saved weather checksums, extracted features, and internal/outer timestamps are verified before scoring.

`strategy_metrics.csv` additionally follows the paired strategy across candidate/F6 refreshes, with the number of fitted versions explicitly reported. It keeps cutoff schedules and day types separate. This is the performance of the ongoing matched-refit strategy, not an estimate for one fixed fitted model; per-version metrics remain available for diagnosing regressions.

There are no prospective accuracy claims until eligible captures have completed service and actual attendance. Promotion remains a separate decision based on sufficient events, both service types, the operational forecast lead, and acceptable shortage/surplus outcomes. The six-day historical snowfall finding is a reason to gather evidence, not a fixed downward attendance correction.

## Refresh the candidate after F6 retraining

The candidate is tied to its exact F6 source-package hash and attendance history. If nightly retraining changes F6, capture still saves F6 and weather, but marks the candidate unavailable until a new version is trained from that baseline. This prevents a silent comparison between models fitted to different attendance snapshots.

Obtain complete historical hourly weather through the new attendance cutoff, then train a new version. Never reuse an existing candidate directory.

```bash
.venv/bin/python scripts/train_weather_candidate.py \
  --location ny_12550 \
  --baseline-model models/visitor_model_ny_12550.joblib \
  --weather-input path/to/complete_historical_hourly_weather.json \
  --package-id ny_12550_weather_shadow_YYYY-MM-DD_v2 \
  --latitude 41.50343 \
  --longitude=-74.01042
```

For the initial version, the input was `artifacts/ny_12550/weather_audit_2026-09-13/hourly_weather.json`. The trainer uses exactly the existing RF400 and HGB Q80 estimators and separate segment imputers. It checks historical coverage, baseline and weather hashes, coordinates, feature order, and fitted dimensions. It writes only versioned packages under `models/candidates`, with metadata/checksums. After successful validation, update only the study configuration's `candidate` path to the new version. The distinct weather package schema is rejected by production `VisitorPredictor` if loaded accidentally.

Future candidates trained from accumulated issue-time snapshots will need a separately versioned training design. This initial trainer deliberately does not relabel realized historical weather as archived forecasts.

## Shared deployment storage

The configured Newburgh study explicitly uses local `data/weather_shadow/ny_12550.sqlite`. A CLI `--local-store` overrides that path. For a deployed server, apply [the dedicated migration](../supabase/migrations/20260913_weather_shadow_snapshots.sql), configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`, and set the study's `local_store` to null. With no explicit local path and no shared settings, the storage library defaults to `data/weather_shadow/<location>.sqlite`; partial shared configuration fails explicitly.

The new `weather_shadow_runs` table permits the server to insert and read records, but not update/delete them or provide the database receipt timestamp. Existing staff-facing prediction logs continue their current behavior. Raw snapshots and local candidate packages are excluded from Git. Keep the capture process separate from staff-facing requests so weather latency/failure cannot delay service recommendations.

Account for DST and scheduler delay; do not run a missed cutoff later and present it as an on-time forecast. Local collection requires the computer and scheduler to be available before cutoff. A persistent deployed scheduler can use the same `capture-due` entry point and configuration.
