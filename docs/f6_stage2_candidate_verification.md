# F6 Stage 2 Candidate Verification

## Scope

- Base commit: `6fba3568292d820aab7111cff459aba74e712bf0`
- Branch: `f6-stage2-candidate`
- Location: `ny_12550`
- Candidate package ID: `ny_12550_f6_2026-07-12_v1`
- Candidate created at: `2026-07-16T19:21:43.757711+00:00`
- Candidate status: `candidate_not_active`

No active-model path, nightly workflow, Supabase data, model contract, model
class, model hyperparameter, or recommendation policy was changed.

## Attendance input validation

The local source remains ignored by Git under `data/updated/*.csv` and was not
rewritten or added to the branch.

- Path: `data/updated/attendance_rows.csv`
- SHA-256: `7bd8c4bf341b516a3370ffbe37eb517698ea68192656a03473a120cd50d603db`
- Size: 30,620 bytes
- Encoding: US-ASCII (valid UTF-8 subset), without a byte-order mark
- Delimiter: comma
- Columns: `location_id`, `service_date`, `visitors`, `created_at`, `updated_at`
- Canonical training fields: `service_date`, `visitors`
- Raw rows: 360
- Minimum service date: `2023-01-01`
- Maximum service date: `2026-07-12`
- Duplicate service dates: 0
- Missing service dates: 0
- Invalid service dates: 0
- Missing attendance values: 0
- Invalid attendance values: 0
- Source location IDs: `ny_12550`
- Non-weekend records: 1 (`2026-04-14`)
- T1-eligible weekend rows used by the feature builder: 359

The original 360-row source was validated without modification. The locked T1
weekday policy excluded the one Tuesday observation when building the 359-row
candidate history.

## Locked contract verification

- Tracked contract file SHA-256:
  `fa63dfe645490071d6712169a0b723d0807c50320fcc072f32b1992fd1cfdfe1`
- Feature set: `F6_COMPACT_SELECTED`
- Feature count: 33
- Feature-order SHA-256:
  `dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419`
- Training window: `TW_EXPANDING`
- Sample weighting: `SW_UNIFORM`
- Weekday policy: `T1_valid_weekends`
- Weather policy: `W0_no_weather`
- Segmentation: separate Saturday and Sunday
- Recommendation policy: `C0_EXISTING_RAW_QUANTILE`
- Point model: `RandomForestRegressor`, 400 trees, depth 8, minimum leaf 2,
  random seed 42
- Quantile model: `HistGradientBoostingRegressor`, quantile 0.8, learning rate
  0.05, depth 4, 500 iterations, random seed 42

## Candidate package

Package directory:
`models/candidates/ny_12550_f6_2026-07-12_v1/`

- `model_package.joblib`: 5,933,909 bytes,
  SHA-256 `9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397`
- `metadata.json`: 4,647 bytes,
  SHA-256 `c5bdfb1396919342d2736ab1579dd3de7fc5f88c4619da7c4afacda2529b9854`
- `checksums.json`: 230 bytes,
  SHA-256 `3a8cf555ccb0e17e8f30d3d6b10ab1ba4f7d439d8254a6a449de795e42541275`

The checksum manifest matches the package and metadata files. The schema-v2
package loads through `VisitorPredictor`, has Saturday and Sunday point,
quantile, and preprocessing components, and contains 359 weekend history rows
from `2023-01-01` through `2026-07-12`.

## Backtest results

| Segment | Rows | MAE | RMSE | MAPE | Bias | P90 absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 322 | 13.6270 | 17.7383 | 10.4206% | 0.2261 | 29.4720 |
| Saturday | 163 | 13.6223 | 17.3566 | 11.8922% | 1.3888 | 29.8218 |
| Sunday | 159 | 13.6318 | 18.1213 | 8.9119% | -0.9659 | 28.5870 |

These metrics verify package generation; they do not reopen feature,
hyperparameter, calibration, or policy selection.

## Prediction smoke test

For Saturday `2026-07-18`:

- Point prediction: `100.9658`
- Raw 80th-percentile prediction: `115.5408`
- C0 suggested meals: `116`
- Percentage buffer: `0.0`
- Residual buffer: `0.0`

## Active-model protection

Before and after candidate training:

- `models/visitor_model_ny_12550.joblib`:
  `ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0`
- `models/visitor_model.joblib`:
  `cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920`
- Active schema: v1
- Active feature count: 26
- Active recommendation policy: legacy max-of-point/quantile/buffers
- Nightly workflow:
  `fb6df61cbccbaea809146c87f24a81f342642f1fc695fcace8f1fa50a0f19225`

The guarded command also refused the now-existing candidate destination.
Reserved or unversioned package IDs and active model paths are covered by
production integration tests.

## Verification

- Candidate and production integration tests: 20 passed
- Complete suite: 115 passed, 30 optional research tests skipped, 9 subtests
  passed
- `git diff --check`: passed

The candidate was trained and verified only. It was not activated or deployed.
