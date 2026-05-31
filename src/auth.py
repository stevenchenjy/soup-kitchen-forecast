from __future__ import annotations

import json
import hashlib
import hmac
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import secrets
from typing import Any


USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
PASSWORD_ITERATIONS = 200_000
PASSWORD_MIN_LENGTH = 8


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


def validate_password(password: str) -> str | None:
    if not password:
        return "Please enter a password."
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    return None


def hash_password(password: str, iterations: int = PASSWORD_ITERATIONS, validate: bool = True) -> dict[str, Any]:
    if validate:
        error = validate_password(password)
        if error:
            raise ValueError(error)
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        iterations,
    )
    return {
        "password_hash": digest.hex(),
        "salt": salt,
        "iterations": iterations,
    }


def _verify_password(password: str, user: dict[str, Any]) -> bool:
    password_hash = user.get("password_hash")
    salt = user.get("salt")
    iterations = user.get("iterations")
    if not password_hash or not salt or not iterations:
        return False
    try:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(str(salt)),
            int(iterations),
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(digest.hex(), str(password_hash))


def _set_password_hash(user: dict[str, Any], password: str) -> dict[str, Any]:
    out = dict(user)
    out.pop("password", None)
    out.update(hash_password(password, validate=False))
    return out


def _normalize_user(row: dict[str, Any]) -> dict[str, Any] | None:
    username = str(row.get("username", "")).strip()
    if not username:
        return None
    role = _normalize_role(row.get("role", "staff"))
    user = {
        "username": username,
        "role": role,
        "authorized_locations": _normalize_authorized_locations(row.get("authorized_locations", []), role),
    }
    if row.get("password_hash") and row.get("salt") and row.get("iterations"):
        user["password_hash"] = str(row["password_hash"])
        user["salt"] = str(row["salt"])
        user["iterations"] = int(row["iterations"])
    elif "password" in row:
        user["password"] = str(row.get("password", ""))
    return user


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


def _same_username(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def save_users(users: list[dict[str, Any]]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized_users: list[dict[str, Any]] = []
    for row in users:
        user = _normalize_user(row)
        if user is not None:
            if "password" in user:
                password = user.pop("password")
                if password:
                    user.update(hash_password(password, validate=False))
            normalized_users.append(user)
    payload = {"users": normalized_users}
    USERS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_user(username: str) -> dict[str, Any] | None:
    username = username.strip()
    for user in load_users():
        if _same_username(user["username"], username):
            return _public_user(user)
    return None


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    username = username.strip()
    users = load_users()
    for index, user in enumerate(users):
        if not _same_username(user["username"], username):
            continue
        if _verify_password(password, user):
            return _public_user(user)
        if user.get("password") == password:
            migrated_user = _set_password_hash(user, password)
            users[index] = migrated_user
            save_users(users)
            return _public_user(migrated_user)
    return None


def delete_user(username: str) -> bool:
    username = username.strip()
    users = load_users()
    kept_users = [user for user in users if not _same_username(user["username"], username)]
    if len(kept_users) == len(users):
        return False
    save_users(kept_users)
    return True


def validate_user_record(username: str) -> dict[str, Any]:
    username = username.strip()
    if not username:
        return {"passed": False, "reason": "Username is empty."}

    user = next((row for row in load_users() if _same_username(row["username"], username)), None)
    if user is None:
        return {"passed": False, "reason": "Username does not exist."}

    role = user.get("role")
    if role not in {"master", "staff"}:
        return {"passed": False, "reason": "Role must be master or staff."}

    authorized_locations = user.get("authorized_locations")
    if not isinstance(authorized_locations, list):
        return {"passed": False, "reason": "authorized_locations must be a list."}
    if role == "master" and authorized_locations != ["*"]:
        return {"passed": False, "reason": "Master accounts must use authorized_locations ['*']."}
    if role == "staff":
        if not authorized_locations:
            return {"passed": False, "reason": "Staff account has no authorized locations."}
        if "*" in authorized_locations:
            return {"passed": False, "reason": "Wildcard '*' is only valid for master accounts."}
    if any(not isinstance(location_id, str) or not location_id.strip() for location_id in authorized_locations):
        return {"passed": False, "reason": "authorized_locations contains an invalid location id."}

    for field in ("password_hash", "salt", "iterations"):
        if not user.get(field):
            return {"passed": False, "reason": f"Missing {field}."}

    return {"passed": True, "reason": "User record is valid."}


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
