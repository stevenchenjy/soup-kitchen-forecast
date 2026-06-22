from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import data_admin


class RecentAttendanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.location_id = "test-location"
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
        self.db_patch = patch.object(
            data_admin,
            "location_db_file",
            return_value=Path(self.temp_dir.name) / "attendance.db",
        )
        self.db_patch.start()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.env.stop()
        self.temp_dir.cleanup()

    def test_sqlite_recent_attendance_is_limited_and_descending(self) -> None:
        for day, visitors in [("2026-06-07", 100), ("2026-06-14", 110), ("2026-06-21", 120)]:
            data_admin.upsert_record(day, visitors, self.location_id)

        recent = data_admin.load_recent_attendance(self.location_id, limit=2)

        self.assertEqual(recent["service_date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-06-21", "2026-06-14"])
        self.assertEqual(recent["visitors"].tolist(), [120, 110])

    def test_supabase_recent_attendance_uses_limit_order_and_read_timeout(self) -> None:
        rows = [
            {"service_date": "2026-06-21", "visitors": 120},
            {"service_date": "2026-06-14", "visitors": 110},
        ]
        with (
            patch.object(data_admin, "_supabase_config", return_value={"url": "https://example.test", "key": "key", "table": "attendance"}),
            patch.object(data_admin, "_supabase_request", return_value=rows) as request,
        ):
            recent = data_admin.load_recent_attendance(self.location_id, limit=2)

        self.assertEqual(len(recent), 2)
        self.assertEqual(request.call_args.kwargs["params"]["location_id"], "eq.test-location")
        self.assertEqual(request.call_args.kwargs["params"]["order"], "service_date.desc")
        self.assertEqual(request.call_args.kwargs["params"]["limit"], "2")
        self.assertEqual(request.call_args.kwargs["timeout"], 20)

    def test_staff_supabase_upsert_can_skip_full_table_reload(self) -> None:
        with (
            patch.object(
                data_admin,
                "_supabase_config",
                return_value={"url": "https://example.test", "key": "key", "table": "attendance"},
            ),
            patch.object(data_admin, "_supabase_request"),
            patch.object(data_admin, "load_clean_data") as load_all,
            patch.object(data_admin, "_mark_location_dirty"),
        ):
            result = data_admin.upsert_record(
                "2026-06-21",
                120,
                self.location_id,
                changed_by=None,
                load_result=False,
            )

        load_all.assert_not_called()
        self.assertEqual(result["visitors"].tolist(), [120])


if __name__ == "__main__":
    unittest.main()
