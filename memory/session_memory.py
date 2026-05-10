import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from upstash_redis import Redis


# ---------------------------------------------------------------------------
# Session data model — captures all inputs needed to invoke the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    place: str = ""
    destination_query: str = ""
    checkin: str = ""
    checkout: str = ""
    adults: int = 1
    rooms: int = 1
    days: int | None = None
    mode_of_travel: str = "car"
    user_preference: str = "ideal"
    hotel_preferences: str = ""
    hotel_preference: str | None = None
    places_per_day: int = 3
    max_km_per_day: int | None = None
    place_preferences: dict = field(default_factory=lambda: {"visited": [], "optional": []})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Redis session handler
# ---------------------------------------------------------------------------

_KEY_PREFIX = "travelstack:session:"


class RedisSessionHandler:
    """
    Manages orchestrator session state in Redis.

    Key format : travelstack:session:{user_session_id}
    Value       : JSON-encoded dict of SessionData fields
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._client = Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL", ""),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN", ""),
        )
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, user_session_id: str) -> str:
        return f"{_KEY_PREFIX}{user_session_id}"

    def _load(self, user_session_id: str) -> dict:
        raw = self._client.get(self._key(user_session_id))
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _dump(self, user_session_id: str, data: dict) -> None:
        self._client.setex(self._key(user_session_id), self._ttl, json.dumps(data))


    def save(self, user_session_id: str, key: str, value: Any) -> None:
        """Set or update a single field in the session JSON."""
        data = self._load(user_session_id)
        data[key] = value
        self._dump(user_session_id, data)

    def save_session(self, user_session_id: str, session: SessionData) -> None:
        """Persist an entire SessionData object."""
        self._dump(user_session_id, session.to_dict())

    def get(self, user_session_id: str, key: str | None = None) -> Any:
        """
        Retrieve the whole session dict, or a specific field if key is given.
        Returns None if the session or field does not exist.
        """
        data = self._load(user_session_id)
        if not data:
            return None
        if key is None:
            return data
        return data.get(key)

    def get_session(self, user_session_id: str) -> SessionData | None:
        """Load session as a SessionData object, or None if not found."""
        data = self._load(user_session_id)
        if not data:
            return None
        return SessionData.from_dict(data)

    def update(self, user_session_id: str, updates: dict) -> None:
        """Merge multiple field updates into the session in one round-trip."""
        data = self._load(user_session_id)
        data.update(updates)
        self._dump(user_session_id, data)

    def delete(self, user_session_id: str) -> None:
        """Remove the session entirely."""
        self._client.delete(self._key(user_session_id))

    def exists(self, user_session_id: str) -> bool:
        """Return True if a session exists for this id."""
        return self._client.exists(self._key(user_session_id)) == 1

    def refresh_ttl(self, user_session_id: str) -> None:
        """Reset the TTL on an existing session."""
        self._client.expire(self._key(user_session_id), self._ttl)
