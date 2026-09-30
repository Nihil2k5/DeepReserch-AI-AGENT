"""Redis-backed coordination: rate limiting and distributed locks.

Rate limits protect the paid upstream APIs (LLM provider, Exa) from a burst of
worker activity; locks keep two workers from mutating the same run.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from .errors import ConflictError
from .logging_config import get_logger

logger = get_logger(__name__)

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""


class RedisCoordinator:
    def __init__(self, url: str) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover
            raise ConflictError("redis package is required: pip install redis") from exc

        self._redis = redis
        self._client = redis.Redis.from_url(url, decode_responses=True)

    @property
    def client(self):
        return self._client

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    # -- rate limiting ---------------------------------------------------
    def allow(self, key: str, limit: int, window_s: int) -> bool:
        """Fixed-window counter. Returns False when the caller is over budget."""
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            values = pipe.execute()
            count = int(values[0])
            if count == 1:
                self._client.expire(key, window_s)
            return count <= limit
        except Exception as exc:  # pragma: no cover - fail open on Redis outage
            logger.warning("Rate limiter unavailable, allowing call: %s", exc)
            return True

    # -- locks -----------------------------------------------------------
    @contextmanager
    def lock(self, name: str, ttl_s: int = 900, wait: bool = False) -> Iterator[bool]:
        """Acquire a lock. Raises :class:`ConflictError` unless ``wait`` is set."""
        token = uuid.uuid4().hex
        acquired = False
        try:
            acquired = bool(self._client.set(name, token, nx=True, ex=ttl_s))
        except Exception as exc:  # pragma: no cover
            logger.warning("Lock backend unavailable (%s); proceeding unlocked", exc)
            yield False
            return

        if not acquired and not wait:
            raise ConflictError(f"Resource is locked by another worker: {name}")

        try:
            yield acquired
        finally:
            if acquired:
                try:
                    self._client.eval(_RELEASE_SCRIPT, 1, name, token)
                except Exception:  # pragma: no cover
                    pass

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass
