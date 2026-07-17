# Prediction Model Provenance Migration

Stage 5 adds nullable provenance fields to every newly generated prediction:
package ID, package schema, feature-set ID, feature-order hash, recommendation
policy, forecast origin, calendar days ahead, service horizon, and model segment.

Local SQLite stores are upgraded additively when opened. Existing rows remain in
place and receive `NULL` for fields that were not recorded historically.

For Supabase, review and apply
`supabase/migrations/20260716_add_prediction_model_provenance.sql` through the
normal database migration process. The migration only adds nullable columns; it
does not delete, update, or rewrite rows. This repository change does not execute
the migration automatically.

Until the Supabase migration is applied, the application detects PostgREST's
missing-column/schema-cache response and retries the prediction-log write with
the pre-Stage-5 columns. Predictions therefore remain writable, but provenance
will only appear in Supabase after the additive migration is active.
