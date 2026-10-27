# Deployment Notes

## Python Runtime

Use Python 3.12 for every deployed/runtime environment:

- Streamlit Cloud Admin app (`app.py`): Python 3.12
- Streamlit Cloud Staff app (`app_staff.py`): Python 3.12
- GitHub Actions nightly retraining: Python 3.12

Avoid Python 3.14 for this project. The trained `.joblib` model files are serialized with
scikit-learn/joblib versions from `requirements.txt`, and the model-loading runtime should match
the training runtime as closely as possible.

GitHub Actions is already configured to train with Python 3.12 in
`.github/workflows/nightly-retrain.yml`.

## Streamlit Cloud

When creating or redeploying each Streamlit Cloud app:

1. Select the correct app entrypoint:
   - Admin: `app.py`
   - Staff: `app_staff.py`
2. Open Advanced settings.
3. Select Python 3.12.
4. Add the required Supabase secrets.
5. Apply every SQL file in `supabase/migrations/`, including
   `20260724_queue_retraining_from_attendance.sql`. The attendance trigger is
   the safety net that queues retraining even if an application request is
   interrupted after saving attendance.

The Model Monitoring page and nightly workflow both print a 12-character
training-state store fingerprint. These fingerprints must match. A mismatch
means Streamlit and GitHub Actions are configured for different Supabase
projects.

The `runtime.txt` file documents the preferred runtime, but Streamlit Cloud Python version should
still be confirmed in the app's Advanced settings.

## Preparation policy releases

`config/meal_policies.json` is a versioned operating policy applied by
`VisitorPredictor` after the raw model recommendation is rounded. It is scoped to
an explicit package location and service date. Deploy it with the application
source to both apps on `main`; the API also includes its configuration fingerprint
in cache invalidation. Nightly training and raw F6/C0 package validation remain
independent of this operating adjustment.

Apply `supabase/migrations/20260716_add_prediction_model_provenance.sql` before
releasing a new preparation policy. It adds nullable provenance columns without
rewriting historical rows. Verify a normal future-service prediction saves its
effective `recommendation_policy_id`, package ID, and origin, in addition to the
adjusted meal count. Old rows with unknown provenance must remain unknown.

For the equal-weight release, verify both October 31 and November 1 predictions
match `max(0, ceil(predicted_quantile) - 7)`, have zero extra buffers, and record
`P1_EQUAL_WEIGHT_Q80_MINUS_7`. Check the actual staff/admin views and the API,
and confirm raw package-validation and weather-shadow paths still use C0.
Rollback the preparation adjustment by reverting the policy configuration and
redeploying; no attendance changes or model retraining are required. Recompute
future recommendations through the normal workflow after a policy change.
