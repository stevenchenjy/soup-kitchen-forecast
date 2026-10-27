# Equal-weight meal preparation policy

Newburgh services from October 31, 2026 use
`P1_EQUAL_WEIGHT_Q80_MINUS_7`: `max(0, ceil(raw Q80) - 7)`.
The user explicitly chose equal weight for potential excess meals and potential
shortfalls on October 26. This replaces the operating preference inferred from
the previous Q80 target. It does not change the visitor forecast, the quantile
model, or the package's locked F6/C0 contract.

## Evidence and tradeoff

The bounded evaluation compared the current target with fixed reductions of
1–10 meals. Selection used 207 preparation services from 2024–2025; temporal
checking used 70 services from January through September 27, 2026, with seven
newer October services kept separate. The objective was mean potential excess
plus mean potential shortfall. Ties preferred the smaller reduction.

Seven fewer meals was the development winner. On the 70 checking services:

| Measure | Raw C0 target | Equal-weight target |
|---|---:|---:|
| Mean equal-cost loss | 18.47 | 14.93 |
| Total potential excess | 1,206 | 837 |
| Total potential shortfall | 87 | 208 |
| Services below attendance | 14 | 23 |
| Coverage | 80.0% | 67.1% |

The equal-cost score improved 19.18%. The paired 95% improvement intervals were
2.28–4.77 loss units per service when resampling whole weekends, and 2.36–4.51
with four-weekend moving blocks. Both weekdays, both longer forecast horizons,
and the seven-service extension improved. The candidate passed the fixed 5%
practical-gain, interval, day-type, horizon, and newest-data gates.

The former 75% minimum coverage gate expressed the previous operating preference
and was removed before this equal-weight recomputation at the user's direction.
Coverage and shortfall magnitude remain explicit outcomes. A lower total loss
does not mean shortages are unchanged: potential shortfall increased by 121
meals in exchange for 369 fewer excess meals in this checking period.

The [tracked release evidence](../config/meal_policy_evaluations/equal_weight_2026-10-26.json)
contains aggregate results, all gates, intervals, and input/plan/script hashes.
The source attendance export had 389 records through October 25, including the
known Tuesday excluded under F6's weekend-only rule. October 24 was absent and
remained unknown. Fresh saved-forecast exports were checked separately against
current authoritative attendance.

These are retrospective target-versus-attendance calculations, not measured
discarded food or confirmed unmet demand. The histories were reused in prior
research, and the earlier equal-weight sensitivity winner was already known.
The revised plan discloses that knowledge. Historical entry/revision vintages
and service closures cannot be reconstructed. Seven newer services are a small
temporal check. The result is the best tested fixed decrement under equal costs,
not a universal optimum or a guarantee for every service.

## Runtime and provenance

The location/date policy is applied centrally in `VisitorPredictor`, so the staff
dashboard, admin dashboard, and API use the same integer target. Packages without
the explicit Newburgh location, legacy packages, and services before the
effective date retain their original behavior. Output metadata distinguishes
the effective preparation policy from the unchanged base model policy.

The API invalidates its predictor cache when either the model or policy
configuration changes, and refuses to cache a mixed snapshot during replacement.
Raw package validation, historical research, and weather-shadow baselines use an
explicit raw-policy mode. Existing model backtests are still labeled C0; they
are not relabeled as evidence for the adjusted preparation count.

Prediction logs store the effective policy ID using the existing additive
provenance migration. Historical unknown policy IDs are not backfilled by guess.
Monitoring matches both package identity and the policy appropriate to each
service date. Nightly model refreshes preserve the tracked preparation policy.

## Release verification

Before release, require the independent outcome checks, relevant regression
checks, full publication test gate, and a fresh-process prediction using the
published model. After release, inspect real staff/admin recommendations for
both weekend days, their saved policy provenance, the deployed source revision,
and the unchanged raw model package. A successful test alone does not establish
deployment or external delivery.
