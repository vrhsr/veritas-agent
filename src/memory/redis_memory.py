"""
Redis Short-Term Memory
Stores conversation history (last 10 turns) with 24-hour TTL.
Key format: session:{session_id}:history
"""
import json
import os
from typing import List, Optional

import redis as redis_lib
from utils.logger import get_logger

logger = get_logger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HISTORY_TTL = 24 * 3600  # 24 hours
MAX_TURNS = 10


class RedisMemory:
    def __init__(self):
        try:
            self._client = redis_lib.from_url(REDIS_URL, decode_responses=True)
            self._client.ping()
            logger.info("redis_connected", url=REDIS_URL)
            self._available = True
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            self._available = False
            self._client = None
            self._local_store: dict[str, list] = {}  # In-memory fallback

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    def push_turn(self, session_id: str, turn: dict) -> None:
        """Add a conversation turn to the session history."""
        if not self._available:
            history = self._local_store.get(session_id, [])
            history.append(turn)
            self._local_store[session_id] = history[-MAX_TURNS:]
            return

        key = self._key(session_id)
        try:
            self._client.rpush(key, json.dumps(turn))
            # Keep only last MAX_TURNS
            self._client.ltrim(key, -MAX_TURNS, -1)
            # Reset TTL on each interaction
            self._client.expire(key, HISTORY_TTL)
        except Exception as e:
            logger.warning("redis_push_error", error=str(e))

    def get_history(self, session_id: str, max_turns: int = MAX_TURNS) -> List[dict]:
        """Retrieve conversation history for a session."""
        if not self._available:
            history = self._local_store.get(session_id, [])
            return history[-max_turns:]

        key = self._key(session_id)
        try:
            raw_turns = self._client.lrange(key, -max_turns, -1)
            return [json.loads(t) for t in raw_turns]
        except Exception as e:
            logger.warning("redis_get_error", error=str(e))
            return []

    def clear_session(self, session_id: str) -> None:
        """Delete all history for a session."""
        if not self._available:
            self._local_store.pop(session_id, None)
            return

        key = self._key(session_id)
        try:
            self._client.delete(key)
        except Exception as e:
            logger.warning("redis_delete_error", error=str(e))

    def set_preference(self, session_id: str, key: str, value: str) -> None:
        """Store a session-level user preference."""
        if not self._available:
            return

        pref_key = f"session:{session_id}:pref:{key}"
        try:
            self._client.setex(pref_key, HISTORY_TTL, value)
        except Exception as e:
            logger.warning("redis_pref_error", error=str(e))

    def get_preference(self, session_id: str, key: str) -> Optional[str]:
        """Retrieve a session-level user preference."""
        if not self._available:
            return None

        pref_key = f"session:{session_id}:pref:{key}"
        try:
            return self._client.get(pref_key)
        except Exception as e:
            logger.warning("redis_pref_get_error", error=str(e))
            return None

    # ── Intent Slot State ──────────────────────────────────────────────────────

    def get_intent(self, session_id: str) -> dict:
        """
        Retrieve the structured intent slot state for a session.
        Returns a dict with keys: task, entity, metric, timeframe.
        All values are None until filled by the query analyzer.
        """
        default = {"task": None, "entity": None, "metric": None, "timeframe": None}
        if not self._available:
            return self._local_store.get(f"{session_id}:intent", default)

        intent_key = f"session:{session_id}:intent"
        try:
            raw = self._client.get(intent_key)
            if raw:
                return json.loads(raw)
            return default
        except Exception as e:
            logger.warning("redis_intent_get_error", error=str(e))
            return default

    def set_intent(self, session_id: str, slots: dict) -> None:
        """
        Persist the intent slot state for a session.
        Only updates keys that are present (non-None values in slots override stored state).
        """
        current = self.get_intent(session_id)
        merged = {**current}
        for k, v in slots.items():
            if v is not None:  # Only overwrite with real values
                merged[k] = v

        if not self._available:
            self._local_store[f"{session_id}:intent"] = merged
            return

        intent_key = f"session:{session_id}:intent"
        try:
            self._client.setex(intent_key, HISTORY_TTL, json.dumps(merged))
        except Exception as e:
            logger.warning("redis_intent_set_error", error=str(e))

    def clear_intent(self, session_id: str) -> None:
        """Clear the intent slot state when a query is fully resolved and answered."""
        if not self._available:
            self._local_store.pop(f"{session_id}:intent", None)
            return

        intent_key = f"session:{session_id}:intent"
        try:
            self._client.delete(intent_key)
        except Exception as e:
            logger.warning("redis_intent_clear_error", error=str(e))

