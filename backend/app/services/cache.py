"""Redis-backed JSON cache with an in-process TTL fallback so the API still works without Redis."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

log = logging.getLogger(__name__)


class Cache:
    def __init__(self, url: str, default_ttl: int = 300, namespace: str = "riq"):
        self.url = url
        self.ns = namespace
        self.default_ttl = default_ttl
        self._redis: aioredis.Redis | None = None
        self._local: dict[str, tuple[float, str]] = {}
        self.redis_ok = False

    async def open(self) -> None:
        try:
            self._redis = aioredis.from_url(self.url, socket_connect_timeout=1, socket_timeout=1)
            await self._redis.ping()
            self.redis_ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("redis unavailable, using in-process cache", extra={"error": str(exc)})
            self._redis, self.redis_ok = None, False

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def get(self, key: str) -> Any | None:
        key = f"{self.ns}:{key}"
        if self._redis:
            try:
                raw = await self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                self.redis_ok = False
        item = self._local.get(key)
        if item and item[0] > time.time():
            return json.loads(item[1])
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        key = f"{self.ns}:{key}"
        ttl = ttl or self.default_ttl
        raw = json.dumps(value, default=str)
        if self._redis:
            try:
                await self._redis.set(key, raw, ex=ttl)
                return
            except Exception:  # noqa: BLE001
                self.redis_ok = False
        if len(self._local) > 2000:
            self._local.clear()
        self._local[key] = (time.time() + ttl, raw)

    async def incr_window(self, key: str, window_s: int) -> int:
        """Fixed-window counter used by the rate limiter (shared across replicas when Redis is up)."""
        key = f"{self.ns}:{key}"
        if self._redis:
            try:
                n = await self._redis.incr(key)
                if n == 1:
                    await self._redis.expire(key, window_s)
                return int(n)
            except Exception:  # noqa: BLE001
                self.redis_ok = False
        now = time.time()
        exp, raw = self._local.get(key, (0, "0"))
        if exp < now:
            exp, raw = now + window_s, "0"
        n = int(raw) + 1
        self._local[key] = (exp, str(n))
        return n

    async def ping(self) -> bool:
        if not self._redis:
            return False
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001
            return False
