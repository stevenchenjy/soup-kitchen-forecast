from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import LOCATIONS_FILE


@dataclass
class Location:
    id: str
    name: str
    zip_code: str
    country_code: str = "US"
    timezone: str = "America/New_York"


DEFAULT_LOCATIONS = [
    {
        "id": "ny_12550",
        "name": "Newburgh, NY 12550",
        "zip_code": "12550",
        "country_code": "US",
        "timezone": "America/New_York",
    },
    {
        "id": "nyc_10001",
        "name": "New York, NY 10001",
        "zip_code": "10001",
        "country_code": "US",
        "timezone": "America/New_York",
    },
    {
        "id": "boston_02108",
        "name": "Boston, MA 02108",
        "zip_code": "02108",
        "country_code": "US",
        "timezone": "America/New_York",
    },
]



def ensure_locations_file() -> None:
    LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCATIONS_FILE.exists():
        return
    payload = {"locations": DEFAULT_LOCATIONS}
    LOCATIONS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def list_locations() -> list[Location]:
    ensure_locations_file()
    payload = json.loads(Path(LOCATIONS_FILE).read_text(encoding="utf-8"))
    raw = payload.get("locations", [])
    out: list[Location] = []
    for item in raw:
        if not item.get("id"):
            continue
        out.append(
            Location(
                id=item["id"],
                name=item.get("name", item["id"]),
                zip_code=item.get("zip_code", "12550"),
                country_code=item.get("country_code", "US"),
                timezone=item.get("timezone", "America/New_York"),
            )
        )
    return out



def get_location(location_id: str) -> Location:
    for loc in list_locations():
        if loc.id == location_id:
            return loc
    raise ValueError(f"Unknown location: {location_id}")
