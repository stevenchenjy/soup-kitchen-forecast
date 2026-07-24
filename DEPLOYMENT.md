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
