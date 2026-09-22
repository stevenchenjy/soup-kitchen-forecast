# Newburgh holistic model verification — September 22, 2026

The supported release decision is to retain the locked F6 algorithm, restore
nightly publication, and refresh its training data through September 20. Neither
the fixed point-prediction blend nor the weather candidate has sufficient evidence
for replacing F6. Two recent Sunday errors are not the basis for this decision.

## Inputs and evaluation design

Read-only CSV exports from the configured Supabase project supplied 380 attendance
records through September 20 and 31 saved prediction records. There were no
duplicate dates or missing counts. The known April 14 Tuesday record was excluded
under F6's existing weekend-only rule, leaving 379 services. All dates shared with
the current F6 training snapshot had identical counts. Absent service dates remain
unknown; the source has no explicit closure-status column.

The attendance export SHA-256 is
`2cc91ce232c2cbf5e108937122e3e4d172d624c16c6c0f8643d030f4234835c5`.
Raw exports, reproducible analysis code, the evaluation plan, and detailed outputs
are retained locally in `artifacts/ny_12550/holistic_audit_2026-09-22/` and excluded
from this release.

The chronological replay covers 275 recorded service dates from January 2024
through September 20, 2026, at preparation, seven-day, and fifteen-day origins:
825 scenarios, not 825 independent services. Each fitted segment model and its
imputer use only training targets available by the origin. It uses the production
33-feature builder, RF400 point estimator, HGB Q80 estimator, expanding history,
and separate Saturday/Sunday models. No future target attendance is included.

Historical replay assumes counts were available by the end of their service date;
it cannot recover historical entry delays or revisions. It is therefore separated
from actual logged forecasts and from immutable, pre-cutoff weather pairs. Earlier
periods were used in previous model research. The 20-service post-lock extension
is a temporal check, not a wholly untouched benchmark.

## Point prediction results

Mean absolute error at the preparation origin, in visitors:

| Model | All 275 services | Recent 52 | Post-lock 20 |
|---|---:|---:|---:|
| Locked F6 | 12.71 | 11.81 | 11.13 |
| Previous matching weekday | 16.75 | 18.33 | 16.90 |
| Last-four matching-weekday mean | 13.65 | 12.81 | 13.09 |
| Last-four matching-weekday median | 13.74 | 13.38 | 14.08 |
| Previous matching weekday/month slot | 16.27 | 15.50 | 12.60 |
| Fixed 50/50 F6/last-four-median blend | 12.64 | 12.34 | 12.53 |

The fixed blend's overall gain is 0.066 visitors, with a paired whole-weekend
95% bootstrap interval of -0.47 to +0.65. A four-week moving-block sensitivity
gives -0.46 to +0.57. On the newest 20 services it is worse by 1.40 visitors;
the whole-weekend interval for its improvement is -2.70 to -0.17. It fails the
predeclared practical-gain and consistency gates. These intervals are descriptive
and conditional on the tested candidates, not adjusted proof after a model search.

F6 preparation MAE is 12.70 on Saturdays and 12.72 on Sundays across the full
replay. On the most recent 52 services it is 12.86 and 10.76 respectively. Across
all dates, seven-day MAE is 12.74 and fifteen-day MAE is 13.41.

## Meal recommendations and observed operation

F6 raw Q80 covers 71.3% of historical replay outcomes, below its nominal 80%.
Coverage is 80.9% in 2026 and 78.8% over the most recent 52 services. The recent
52-service simulation has 11 shortfall dates, 67 total shortfall meals, and 14.65
surplus meals per service. These are recommendation-versus-attendance calculations,
not measured kitchen waste or confirmed real shortages. The point blend leaves
Q80 unchanged and consequently cannot repair its calibration.

There are 18 logged services after F6 activation with its zero-buffer policy.
Two forecasts were saved after preparation cutoff and are excluded from the strict
preparation subset. The remaining 16 have MAE 11.07: Saturday 11.67, Sunday 10.48.
Their recommendations cover 12 of 16 services, with 15 total shortfall meals and
10.56 surplus meals per service. This small sample does not precisely establish
long-run coverage. The remote prediction table lacks model-provenance columns,
so these are post-release operational logs, not proof of each exact package hash.

The earlier training-window, sample-weight, and Q80 calibration experiments were
also reviewed. Finite windows and recency weights had not established a stable
gain; the selected calibration alternative failed its surplus guardrail. Their
reused confirmation periods do not become fresh evidence in this audit.

## Weather comparison

Both saved September 19/20 forecast pairs pass receipt, prediction, and persistence
before the actual preparation cutoff, with matched F6/candidate attendance inputs.
They can now be scored against authoritative actual attendance.

| Service | F6 absolute error | Weather absolute error | F6 recommendation surplus | Weather surplus |
|---|---:|---:|---:|---:|
| September 19, Saturday | 3.38 | 1.16 | 14 | 11 |
| September 20, Sunday | 29.48 | 23.86 | 43 | 40 |

The weather candidate was closer on both services, but two services and no extreme
events cannot establish reliable benefit. The prior 168-service realized-weather
study showed a 0.49-visitor average gain with uncertainty spanning zero; realized
weather is not the forecast available before preparation. Continue the prospective
comparison and keep weather out of production until that evidence supports it.

## Release repairs and limits

The September 21 and 22 nightly runs trained successfully but failed publication:
14 weather tests replayed a fixed September 18 origin against the changing active
package. When training advanced through September 20, the correct leakage guard
rejected that future history. Fixed-clock tests now use the immutable July F6
fixture, and an explicit regression still requires future-trained baselines to fail
before any weather request or storage write. The production guard is unchanged.

A long-running FastAPI process also cached models indefinitely by location. It now
invalidates its predictor after a package replacement and refuses to cache a model
if the file changes during loading. Staff/admin Streamlit requests already reload
the package; this closes the corresponding API release gap.

The release retains F6/C0, its rollback mechanism, and the weather study's isolation.
Refreshing the same algorithm with new attendance is not a newly validated model
architecture. Full-suite CI, nightly publication, package identity, current input
parity, and application verification are required before calling the refresh live.

## Interpretation audit

All 11 methodological checks were considered: aggregation versus day-type results;
ecological inference; selection and collider bias; event base rates; regression to
the mean; missing-service survivorship; multiple comparisons; researcher choices;
causal claims; and reverse causality. Material limitations are reused historical
data, unknown closures/entry-time vintages, correlated services, a short post-lock
period, missing remote model provenance, and sparse extreme weather. No causal
effect of weather or independently guaranteed improvement is claimed.
