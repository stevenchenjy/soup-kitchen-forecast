from __future__ import annotations

from unittest.mock import patch

from src import model_training_runs


def test_store_fingerprint_is_stable_and_does_not_expose_url() -> None:
    url = "https://example-project.supabase.co"
    with patch.object(
        model_training_runs,
        "_supabase_config",
        return_value={"url": url, "key": "secret"},
    ):
        fingerprint = model_training_runs.model_training_run_store_fingerprint()

    assert len(fingerprint) == 12
    assert url not in fingerprint
    assert fingerprint == model_training_runs.hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:12]


def test_defer_location_clears_only_unchanged_dirty_marker() -> None:
    with patch.object(
        model_training_runs, "supabase_configured", return_value=True
    ), patch.object(
        model_training_runs, "now_iso", return_value="2026-07-24T12:00:00+00:00"
    ), patch.object(
        model_training_runs,
        "_supabase_request",
        return_value=[{"location_id": "ny_12550", "dirty": False}],
    ) as request:
        cleared = model_training_runs.defer_location_until_attendance_changes(
            "ny_12550",
            "2026-07-19T12:00:00+00:00",
        )

    assert cleared is True
    assert request.call_args.args[:2] == ("model_retrain_state", "PATCH")
    assert request.call_args.kwargs["params"] == {
        "location_id": "eq.ny_12550",
        "last_attendance_updated_at": "eq.2026-07-19T12:00:00+00:00",
    }
    assert request.call_args.kwargs["payload"] == {
        "dirty": False,
        "updated_at": "2026-07-24T12:00:00+00:00",
    }
