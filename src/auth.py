from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    username: str
    role: str


USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "staff": {"password": "staff123", "role": "staff"},
}



def authenticate(username: str, password: str) -> User | None:
    row = USERS.get(username)
    if not row:
        return None
    if row["password"] != password:
        return None
    return User(username=username, role=row["role"])
