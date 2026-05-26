from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    username: str
    role: str


USERS = {
    "admin": {"password": "VeryStrongAdmin2026", "role": "admin"},
    "staff": {"password": "SoupKitchenStaff2026", "role": "staff"},
}



def authenticate(username: str, password: str) -> User | None:
    row = USERS.get(username)
    if not row:
        return None
    if row["password"] != password:
        return None
    return User(username=username, role=row["role"])
