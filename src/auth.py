from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"


@dataclass
class User:
    username: str
    role: str
    authorized_locations: list[str]


DEFAULT_USERS = [
    {
        "username": "admin",
        "password": "VeryStrongAdmin2026",
        "role": "master",
        "authorized_locations": ["*"],
    },
    {
        "username": "staff",
        "password": "SoupKitchenStaff2026",
        "role": "staff",
        "authorized_locations": ["ny_12550"],
    },
]


def _normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    if role == "admin":
        return "master"
    if role not in {"master", "staff"}:
        return "staff"
    return role


def _normalize_authorized_locations(value: Any, role: str) -> list[str]:
    if role == "master":
        return ["*"]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(location_id).strip() for location_id in value if str(location_id).strip()]


def _normalize_user(row: dict[str, Any]) -> dict[str, Any] | None:
    username = str(row.get("username", "")).strip()
    if not username:
        return None
    role = _normalize_role(row.get("role", "staff"))
    return {
        "username": username,
        "password": str(row.get("password", "")),
        "role": role,
        "authorized_locations": _normalize_authorized_locations(row.get("authorized_locations", []), role),
    }


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": user["username"],
        "role": user["role"],
        "authorized_locations": list(user.get("authorized_locations", [])),
    }


def _user_value(user: dict[str, Any] | User, key: str, default: Any = None) -> Any:
    if isinstance(user, dict):
        return user.get(key, default)
    return getattr(user, key, default)


def ensure_users_file() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if USERS_FILE.exists():
        return
    save_users(deepcopy(DEFAULT_USERS))


def load_users() -> list[dict[str, Any]]:
    ensure_users_file()
    payload = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    raw_users = payload.get("users", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_users, list):
        return []
    users: list[dict[str, Any]] = []
    for row in raw_users:
        if not isinstance(row, dict):
            continue
        user = _normalize_user(row)
        if user is not None:
            users.append(user)
    return users


def save_users(users: list[dict[str, Any]]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized_users: list[dict[str, Any]] = []
    for row in users:
        user = _normalize_user(row)
        if user is not None:
            normalized_users.append(user)
    payload = {"users": normalized_users}
    USERS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_user(username: str) -> dict[str, Any] | None:
    username = username.strip()
    for user in load_users():
        if user["username"] == username:
            return _public_user(user)
    return None


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    username = username.strip()
    for user in load_users():
        if user["username"] == username and user["password"] == password:
            return _public_user(user)
    return None


def get_authorized_locations(user: dict[str, Any] | User, all_locations: list[Any]) -> list[Any]:
    role = _normalize_role(_user_value(user, "role", "staff"))
    authorized_locations = _normalize_authorized_locations(_user_value(user, "authorized_locations", []), role)
    if role == "master":
        return list(all_locations)
    authorized_ids = set(authorized_locations)
    return [location for location in all_locations if getattr(location, "id", None) in authorized_ids]


def require_role(user: dict[str, Any] | User, allowed_roles: set[str]) -> bool:
    role = _normalize_role(_user_value(user, "role", "staff"))
    allowed = {_normalize_role(allowed_role) for allowed_role in allowed_roles}
    return role in allowed


def authenticate(username: str, password: str) -> User | None:
    user = authenticate_user(username, password)
    if user is None:
        return None
    role = "admin" if user["role"] == "master" else user["role"]
    return User(
        username=user["username"],
        role=role,
        authorized_locations=user["authorized_locations"],
    )
