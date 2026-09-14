from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

from src.location_config import Location
from src.weather_snapshots import (
    FORECAST_URL, GEOCODING_URL, HOURLY_UNITS, WEATHER_FEATURE_COLUMNS,
    WeatherSnapshotError, extract_weather_features, fetch_weather_snapshot, snapshot_sha256,
)


def hourly_payload(day: str = "2026-09-19") -> dict:
    return {
        "latitude": 41.5, "longitude": -74.0, "timezone": "America/New_York", "utc_offset_seconds": -14400,
        "hourly_units": dict(HOURLY_UNITS),
        "hourly": {
            "time": [f"{day}T{hour}:00" for hour in (10, 11, 12, 13, 14)],
            "apparent_temperature": [-90, -3, 2, 1, 90],
            "precipitation": [90, 80, 1.5, 2.5, 70],
            "wind_gusts_10m": [900, 800, 25, 40, 700],
            "snowfall": [90, 80, 0.5, 1.5, 70],
            "snow_depth": [9, 0.10, 0.12, 0.11, 8],
        },
    }


class WeatherFeatureTests(unittest.TestCase):
    def test_exact_service_interval_excludes_earlier_hour_and_later_values(self) -> None:
        features = extract_weather_features(hourly_payload(), "2026-09-19")
        self.assertEqual(tuple(features), WEATHER_FEATURE_COLUMNS)
        self.assertEqual(list(features.values()), [-3, 2, 4, 40, 2, 0.12])

    def test_every_required_hour_must_be_present(self) -> None:
        for missing_hour in (11, 12, 13):
            payload = hourly_payload()
            index = missing_hour - 10
            for values in payload["hourly"].values():
                del values[index]
            with self.subTest(hour=missing_hour), self.assertRaisesRegex(WeatherSnapshotError, "Missing required"):
                extract_weather_features(payload, "2026-09-19")

    def test_selected_values_cannot_be_missing_nonfinite_or_nonnumeric(self) -> None:
        for variable in HOURLY_UNITS:
            if variable == "time":
                continue
            for value in (None, float("nan"), float("inf"), "0", True):
                payload = hourly_payload()
                payload["hourly"][variable][2] = value
                with self.subTest(variable=variable, value=value), self.assertRaises(WeatherSnapshotError):
                    extract_weather_features(payload, "2026-09-19")

    def test_11am_precipitation_gusts_and_snowfall_are_not_required_service_intervals(self) -> None:
        payload = hourly_payload()
        for variable in ("precipitation", "wind_gusts_10m", "snowfall"):
            payload["hourly"][variable][1] = None
        self.assertEqual(extract_weather_features(payload, "2026-09-19")["wx_precip_mm"], 4)

    def test_negative_accumulations_or_gusts_are_rejected(self) -> None:
        for variable in ("precipitation", "wind_gusts_10m", "snowfall", "snow_depth"):
            payload = hourly_payload()
            payload["hourly"][variable][2] = -1
            with self.subTest(variable=variable), self.assertRaisesRegex(WeatherSnapshotError, "negative"):
                extract_weather_features(payload, "2026-09-19")

    def test_unit_mismatches_and_missing_units_are_rejected(self) -> None:
        for variable in HOURLY_UNITS:
            payload = hourly_payload()
            payload["hourly_units"][variable] = "wrong"
            with self.subTest(variable=variable), self.assertRaisesRegex(WeatherSnapshotError, "unit"):
                extract_weather_features(payload, "2026-09-19")

    def test_duplicate_hour_cannot_silently_double_count_or_choose_a_value(self) -> None:
        payload = hourly_payload()
        for values in payload["hourly"].values():
            values.append(values[2])
        with self.assertRaisesRegex(WeatherSnapshotError, "Duplicate"):
            extract_weather_features(payload, "2026-09-19")

    def test_misaligned_arrays_and_non_hour_timestamps_are_rejected(self) -> None:
        payload = hourly_payload()
        payload["hourly"]["snow_depth"].pop()
        with self.assertRaisesRegex(WeatherSnapshotError, "length"):
            extract_weather_features(payload, "2026-09-19")
        payload = hourly_payload()
        payload["hourly"]["time"][2] = "2026-09-19T12:30"
        with self.assertRaisesRegex(WeatherSnapshotError, "timestamp"):
            extract_weather_features(payload, "2026-09-19")

    def test_wrong_date_timezone_or_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(WeatherSnapshotError, "Missing required"):
            extract_weather_features(hourly_payload(), "2026-09-20")
        payload = hourly_payload()
        payload["timezone"] = "UTC"
        with self.assertRaisesRegex(WeatherSnapshotError, "timezone"):
            extract_weather_features(payload, "2026-09-19")
        payload = hourly_payload()
        payload["hourly"]["time"][2] += "+00:00"
        with self.assertRaisesRegex(WeatherSnapshotError, "timestamp"):
            extract_weather_features(payload, "2026-09-19")

    def test_dst_transition_days_use_local_service_hours(self) -> None:
        for day, offset in (("2026-03-08", "-04:00"), ("2026-11-01", "-05:00")):
            payload = hourly_payload(day)
            payload["hourly"]["time"] = [stamp + offset for stamp in payload["hourly"]["time"]]
            with self.subTest(day=day):
                self.assertEqual(extract_weather_features(payload, day)["wx_precip_mm"], 4)


class WeatherSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.location = {"id": "ny_12550", "timezone": "America/New_York", "latitude": 41.5034, "longitude": -74.0104}
        self.started = datetime(2026, 9, 19, 12, 58, tzinfo=timezone.utc)
        self.cutoff = datetime(2026, 9, 19, 13, 0, tzinfo=timezone.utc)
        self.response = Mock()
        self.response.json.return_value = hourly_payload()

    def _fetch(self, **kwargs) -> dict:
        defaults = {"service_date": "2026-09-19", "cutoff_at": self.cutoff, "location": self.location,
                    "now": Mock(side_effect=[self.started, self.started + timedelta(seconds=2)])}
        defaults.update(kwargs)
        return fetch_weather_snapshot(**defaults)

    @patch("src.weather_snapshots.requests.get")
    def test_snapshot_preserves_availability_provenance_and_explicit_units(self, get: Mock) -> None:
        get.return_value = self.response
        snapshot = self._fetch()
        self.assertEqual(snapshot["features"]["wx_snowfall_cm"], 2)
        self.assertEqual(snapshot["raw_response"], hourly_payload())
        self.assertEqual(snapshot["request_started_at"], self.started.isoformat())
        self.assertEqual(snapshot["retrieved_at"], (self.started + timedelta(seconds=2)).isoformat())
        self.assertIsNone(snapshot["provider_issued_at"])
        self.assertEqual(snapshot["location_id"], "ny_12550")
        self.assertEqual(snapshot["snapshot_sha256"], snapshot_sha256(snapshot))
        self.assertEqual(get.call_args.args, (FORECAST_URL,))
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["temperature_unit"], "celsius")
        self.assertEqual(params["wind_speed_unit"], "kmh")
        self.assertEqual(params["precipitation_unit"], "mm")
        self.assertEqual(params["start_hour"], "2026-09-19T11:00")
        self.assertEqual(params["end_hour"], "2026-09-19T13:00")
        altered = deepcopy(snapshot)
        altered["cutoff_at"] = "2026-09-19T14:00:00+00:00"
        self.assertNotEqual(snapshot["snapshot_sha256"], snapshot_sha256(altered))

    @patch("src.weather_snapshots.requests.get")
    def test_past_cutoff_cannot_backdate_current_forecast(self, get: Mock) -> None:
        with self.assertRaisesRegex(WeatherSnapshotError, "backdated"):
            self._fetch(now=lambda: self.cutoff + timedelta(seconds=1))
        get.assert_not_called()

    @patch("src.weather_snapshots.requests.get")
    def test_response_finishing_after_cutoff_is_ineligible(self, get: Mock) -> None:
        get.return_value = self.response
        with self.assertRaisesRegex(WeatherSnapshotError, "after the preparation cutoff"):
            self._fetch(now=Mock(side_effect=[self.started, self.cutoff + timedelta(microseconds=1)]))
        self.assertEqual(get.call_count, 1)

    @patch("src.weather_snapshots.requests.get")
    def test_naive_cutoff_clock_and_during_service_cutoffs_are_rejected(self, get: Mock) -> None:
        for kwargs in ({"cutoff_at": "2026-09-19T09:00"},
                       {"now": lambda: self.started.replace(tzinfo=None)},
                       {"cutoff_at": "2026-09-19T11:00:00-04:00"},
                       {"cutoff_at": "2026-09-19T12:00:00-04:00"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(WeatherSnapshotError):
                self._fetch(**kwargs)
        get.assert_not_called()

    @patch("src.weather_snapshots.requests.get")
    def test_service_date_beyond_live_horizon_is_rejected_before_network(self, get: Mock) -> None:
        with self.assertRaisesRegex(WeatherSnapshotError, "horizon"):
            self._fetch(service_date="2026-10-10", cutoff_at="2026-10-10T09:00:00-04:00")
        get.assert_not_called()

    @patch("src.weather_snapshots.requests.get")
    def test_clock_going_backward_is_rejected(self, get: Mock) -> None:
        get.return_value = self.response
        with self.assertRaisesRegex(WeatherSnapshotError, "clock is inconsistent"):
            self._fetch(now=Mock(side_effect=[self.started, self.started - timedelta(seconds=1)]))

    @patch("src.weather_snapshots.requests.get")
    def test_unresolved_other_location_never_falls_back_to_newburgh(self, get: Mock) -> None:
        get.return_value.json.return_value = {"results": []}
        with self.assertRaisesRegex(WeatherSnapshotError, "did not resolve"):
            self._fetch(location=Location("boston_02108", "Boston", "02108"))
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.args, (GEOCODING_URL,))

    @patch("src.weather_snapshots.requests.get")
    def test_geocoding_requires_matching_postcode_country_and_timezone(self, get: Mock) -> None:
        match = {"id": 5128654, "name": "Newburgh", "country_code": "US", "timezone": "America/New_York",
                 "postcodes": ["12550"], "latitude": 41.5034, "longitude": -74.0104}
        geocode_response = Mock()
        geocode_response.json.return_value = {"results": [match]}
        get.side_effect = [geocode_response, self.response]
        snapshot = self._fetch(location=Location("ny_12550", "Newburgh", "12550"))
        self.assertEqual(snapshot["geolocation"]["result"], match)
        self.assertEqual(snapshot["geolocation"]["request_params"]["countryCode"], "US")
        for field, wrong in (("country_code", "CA"), ("timezone", "UTC"), ("postcodes", ["10001"])):
            geocode_response.json.return_value = {"results": [{**match, field: wrong}]}
            get.side_effect = [geocode_response]
            with self.subTest(field=field), self.assertRaises(WeatherSnapshotError):
                self._fetch(location=Location("ny_12550", "Newburgh", "12550"))

    @patch("src.weather_snapshots.requests.get")
    def test_dst_cutoff_is_compared_with_service_start_in_utc(self, get: Mock) -> None:
        for day, cutoff, start in (("2026-03-08", "2026-03-08T10:59:59-04:00", "2026-03-08T14:55:00+00:00"),
                                   ("2026-11-01", "2026-11-01T10:59:59-05:00", "2026-11-01T15:55:00+00:00")):
            self.response.json.return_value = hourly_payload(day)
            get.return_value = self.response
            with self.subTest(day=day):
                result = self._fetch(service_date=day, cutoff_at=cutoff, now=lambda: datetime.fromisoformat(start))
                self.assertEqual(result["features"]["wx_precip_mm"], 4)


if __name__ == "__main__":
    unittest.main()
