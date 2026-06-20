from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import data_admin, prediction_logs


class AttendanceUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Path(self.temp_dir.name) / "attendance.db"
        self.location_id = "test-location"
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
        self.path_patches = [
            patch.object(data_admin, "location_db_file", return_value=self.db),
            patch.object(prediction_logs, "location_db_file", return_value=self.db),
            patch.object(data_admin, "_mark_location_dirty"),
        ]
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self) -> None:
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.env.stop()
        self.temp_dir.cleanup()

    def _seed_prediction(self, service_date: str, predicted: int, actual: int | None) -> None:
        absolute_error = abs(actual - predicted) if actual is not None else None
        with prediction_logs._connect(self.location_id) as conn:
            conn.execute(
                "INSERT INTO prediction_logs "
                "(location_id, service_date, prediction_created_at, predicted_visitors, suggested_meals, "
                "actual_visitors, absolute_error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.location_id,
                    service_date,
                    "2026-06-01T12:00:00+00:00",
                    predicted,
                    predicted,
                    actual,
                    absolute_error,
                    "2026-06-01T12:00:00+00:00",
                    "2026-06-01T12:00:00+00:00",
                ),
            )
            conn.commit()

    def _attendance_value(self, service_date: str) -> int | None:
        df = data_admin.load_clean_data(self.location_id)
        if df.empty:
            return None
        match = df[df["service_date"].dt.strftime("%Y-%m-%d") == service_date]
        return None if match.empty else int(match.iloc[0]["visitors"])

    def _prediction_actual(self) -> tuple[int | None, float | None]:
        row = prediction_logs.load_prediction_logs(self.location_id, limit=1)[0]
        return row["actual_visitors"], row["absolute_error"]

    def test_undo_add_removes_attendance_and_clears_prediction_actual(self) -> None:
        service_date = "2026-06-14"
        self._seed_prediction(service_date, predicted=100, actual=None)

        data_admin.upsert_record(service_date, 140, self.location_id, changed_by=self.username)
        prediction_logs.update_prediction_logs_with_actual(self.location_id, service_date, 140)

        change = data_admin.undo_last_attendance_input(self.location_id, self.username)

        self.assertEqual(change["operation"], "ADD")
        self.assertIsNone(self._attendance_value(service_date))
        self.assertEqual(self._prediction_actual(), (None, None))
        self.assertIsNone(data_admin.latest_attendance_change(self.location_id, self.username))

    def test_undo_update_restores_attendance_and_prediction_error(self) -> None:
        service_date = "2026-06-14"
        data_admin.upsert_record(service_date, 120, self.location_id)
        self._seed_prediction(service_date, predicted=100, actual=120)

        data_admin.upsert_record(service_date, 140, self.location_id, changed_by=self.username)
        prediction_logs.update_prediction_logs_with_actual(self.location_id, service_date, 140)

        change = data_admin.undo_last_attendance_input(self.location_id, self.username)

        self.assertEqual(change["operation"], "UPDATE")
        self.assertEqual(self._attendance_value(service_date), 120)
        self.assertEqual(self._prediction_actual(), (120, 20.0))

    def test_undo_delete_recreates_attendance_and_prediction_error(self) -> None:
        service_date = "2026-06-14"
        data_admin.upsert_record(service_date, 120, self.location_id)
        self._seed_prediction(service_date, predicted=100, actual=120)

        data_admin.delete_record(service_date, self.location_id, changed_by=self.username)
        prediction_logs.set_prediction_logs_actual(self.location_id, service_date, None)

        change = data_admin.undo_last_attendance_input(self.location_id, self.username)

        self.assertEqual(change["operation"], "DELETE")
        self.assertEqual(self._attendance_value(service_date), 120)
        self.assertEqual(self._prediction_actual(), (120, 20.0))

    def test_undo_is_user_scoped_and_only_one_level(self) -> None:
        data_admin.upsert_record("2026-06-07", 120, self.location_id, changed_by=self.username)
        data_admin.upsert_record("2026-06-14", 140, self.location_id, changed_by=self.username)
        data_admin.upsert_record("2026-06-21", 160, self.location_id, changed_by="another-user")

        change = data_admin.undo_last_attendance_input(self.location_id, self.username)

        self.assertEqual(change["service_date"], "2026-06-14")
        self.assertEqual(self._attendance_value("2026-06-07"), 120)
        self.assertIsNone(self._attendance_value("2026-06-14"))
        self.assertEqual(self._attendance_value("2026-06-21"), 160)
        self.assertIsNone(data_admin.undo_last_attendance_input(self.location_id, self.username))
        self.assertIsNotNone(data_admin.latest_attendance_change(self.location_id, "another-user"))


if __name__ == "__main__":
    unittest.main()
