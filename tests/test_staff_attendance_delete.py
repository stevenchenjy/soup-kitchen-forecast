from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import data_admin, prediction_logs


class StaffAttendanceDeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.location_id = "test-location"
        self.other_location_id = "other-location"
        self.username = "staff-user"
        self.env = patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "",
                "SUPABASE_SERVICE_ROLE_KEY": "",
                "SUPABASE_ANON_KEY": "",
            },
            clear=False,
        )
        self.env.start()

        def db_for_location(location_id: str) -> Path:
            return Path(self.temp_dir.name) / f"{location_id}.db"

        self.path_patches = [
            patch.object(data_admin, "location_db_file", side_effect=db_for_location),
            patch.object(prediction_logs, "location_db_file", side_effect=db_for_location),
            patch.object(data_admin, "_mark_location_dirty"),
        ]
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self) -> None:
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.env.stop()
        self.temp_dir.cleanup()

    def _seed_prediction(self, service_date: str, actual: int) -> None:
        with prediction_logs._connect(self.location_id) as conn:
            conn.execute(
                "INSERT INTO prediction_logs "
                "(location_id, service_date, prediction_created_at, predicted_visitors, suggested_meals, "
                "actual_visitors, absolute_error, waste_avoided_meals, estimated_co2e_reduction_kg, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.location_id,
                    service_date,
                    "2026-06-01T12:00:00+00:00",
                    100,
                    110,
                    actual,
                    abs(actual - 100),
                    11.0,
                    18.81,
                    "2026-06-01T12:00:00+00:00",
                    "2026-06-01T12:00:00+00:00",
                ),
            )
            conn.commit()

    def _attendance_value(self, location_id: str, service_date: str) -> int | None:
        df = data_admin.load_clean_data(location_id)
        if df.empty:
            return None
        match = df[df["service_date"].dt.strftime("%Y-%m-%d") == service_date]
        return None if match.empty else int(match.iloc[0]["visitors"])

    def test_latest_add_becomes_eligible_and_update_does_not_replace_receipt(self) -> None:
        data_admin.upsert_record("2026-06-14", 120, self.location_id, changed_by=self.username)
        original = data_admin.latest_staff_created_attendance(self.location_id, self.username)

        data_admin.upsert_record("2026-06-14", 140, self.location_id, changed_by=self.username)
        updated = data_admin.latest_staff_created_attendance(self.location_id, self.username)

        self.assertEqual(original["receipt_id"], updated["receipt_id"])
        self.assertEqual(updated["service_date"], "2026-06-14")
        self.assertEqual(updated["visitors"], 140)

    def test_update_of_existing_unowned_date_does_not_create_receipt(self) -> None:
        data_admin.upsert_record("2026-06-14", 120, self.location_id)

        data_admin.upsert_record("2026-06-14", 140, self.location_id, changed_by=self.username)

        self.assertIsNone(data_admin.latest_staff_created_attendance(self.location_id, self.username))

    def test_delete_removes_attendance_and_clears_only_prediction_actual_fields(self) -> None:
        service_date = "2026-06-14"
        data_admin.upsert_record(service_date, 140, self.location_id, changed_by=self.username)
        self._seed_prediction(service_date, actual=140)

        deleted = data_admin.delete_latest_staff_created_attendance(self.location_id, self.username)

        self.assertEqual(deleted["service_date"], service_date)
        self.assertIsNone(self._attendance_value(self.location_id, service_date))
        log = prediction_logs.load_prediction_logs(self.location_id, limit=1)[0]
        self.assertIsNone(log["actual_visitors"])
        self.assertIsNone(log["absolute_error"])
        self.assertEqual(log["predicted_visitors"], 100.0)
        self.assertEqual(log["suggested_meals"], 110)
        self.assertEqual(log["waste_avoided_meals"], 11.0)
        self.assertEqual(log["estimated_co2e_reduction_kg"], 18.81)

    def test_receipts_are_scoped_by_user_and_location(self) -> None:
        data_admin.upsert_record("2026-06-14", 140, self.location_id, changed_by=self.username)
        data_admin.upsert_record("2026-06-21", 160, self.location_id, changed_by="another-user")
        data_admin.upsert_record("2026-06-28", 180, self.other_location_id, changed_by=self.username)

        entry = data_admin.latest_staff_created_attendance(self.location_id, self.username)
        deleted = data_admin.delete_latest_staff_created_attendance(self.location_id, self.username)

        self.assertEqual(entry["service_date"], "2026-06-14")
        self.assertEqual(deleted["service_date"], "2026-06-14")
        self.assertEqual(self._attendance_value(self.location_id, "2026-06-21"), 160)
        self.assertEqual(self._attendance_value(self.other_location_id, "2026-06-28"), 180)
        self.assertIsNotNone(data_admin.latest_staff_created_attendance(self.location_id, "another-user"))
        self.assertIsNotNone(data_admin.latest_staff_created_attendance(self.other_location_id, self.username))

    def test_receipt_cannot_be_reused_and_reentry_creates_new_receipt(self) -> None:
        service_date = "2026-06-14"
        data_admin.upsert_record(service_date, 140, self.location_id, changed_by=self.username)
        first = data_admin.latest_staff_created_attendance(self.location_id, self.username)

        data_admin.delete_latest_staff_created_attendance(self.location_id, self.username)

        self.assertIsNone(data_admin.latest_staff_created_attendance(self.location_id, self.username))
        self.assertIsNone(data_admin.delete_latest_staff_created_attendance(self.location_id, self.username))

        data_admin.upsert_record(service_date, 120, self.location_id, changed_by=self.username)
        second = data_admin.latest_staff_created_attendance(self.location_id, self.username)

        self.assertIsNotNone(second)
        self.assertNotEqual(first["receipt_id"], second["receipt_id"])
        self.assertEqual(second["visitors"], 120)

    def test_new_add_replaces_eligibility_without_creating_delete_chain(self) -> None:
        data_admin.upsert_record("2026-06-07", 120, self.location_id, changed_by=self.username)
        data_admin.upsert_record("2026-06-14", 140, self.location_id, changed_by=self.username)

        deleted = data_admin.delete_latest_staff_created_attendance(self.location_id, self.username)

        self.assertEqual(deleted["service_date"], "2026-06-14")
        self.assertEqual(self._attendance_value(self.location_id, "2026-06-07"), 120)
        self.assertIsNone(data_admin.latest_staff_created_attendance(self.location_id, self.username))


if __name__ == "__main__":
    unittest.main()
