-- Stage 5: additive prediction provenance for F6 monitoring.
-- Safe for existing rows: every new column is nullable and no row is rewritten.

alter table public.prediction_logs
  add column if not exists package_id text,
  add column if not exists model_package_schema_version integer,
  add column if not exists feature_set_id text,
  add column if not exists feature_order_sha256 text,
  add column if not exists recommendation_policy_id text,
  add column if not exists forecast_origin date,
  add column if not exists calendar_days_ahead integer,
  add column if not exists service_horizon integer;

comment on column public.prediction_logs.package_id is
  'Versioned model package used to create the prediction.';
comment on column public.prediction_logs.model_package_schema_version is
  'Model package compatibility schema version.';
comment on column public.prediction_logs.feature_set_id is
  'Feature-set identifier, such as F6_COMPACT_SELECTED.';
comment on column public.prediction_logs.feature_order_sha256 is
  'SHA-256 of the ordered production feature list.';
comment on column public.prediction_logs.recommendation_policy_id is
  'Meal recommendation policy used by the predictor.';
comment on column public.prediction_logs.forecast_origin is
  'Origin date from which target-valid features were constructed.';
comment on column public.prediction_logs.calendar_days_ahead is
  'Calendar-day difference between forecast origin and service date.';
comment on column public.prediction_logs.service_horizon is
  'Count of valid weekend services from origin through the target.';
