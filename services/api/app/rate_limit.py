from __future__ import annotations

import time
from functools import lru_cache

import redis
from fastapi import HTTPException, Request

from .config import get_settings


@lru_cache
def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().valkey_url, decode_responses=True, socket_timeout=1.0)


def _enforce(request: Request, *, bucket: str, limit: int, window_seconds: int) -> None:
    settings = get_settings()
    identity = request.client.host if request.client else "unknown"
    epoch_bucket = int(time.time()) // window_seconds
    key = f"ratelimit:{bucket}:{identity}:{epoch_bucket}"
    try:
        count = int(_client().incr(key))
        if count == 1:
            _client().expire(key, window_seconds + 2)
    except redis.RedisError as exc:
        # A local demo should remain usable if the cache is restarted. A shared deployment should
        # fail closed so an outage does not silently remove abuse protection.
        if settings.app_env.lower() in {"development", "test", "local"}:
            return
        raise HTTPException(status_code=503, detail="Rate limiter unavailable") from exc
    if count > limit:
        raise HTTPException(status_code=429, detail="Too many requests")


def optimizer_rate_limit(request: Request) -> None:
    _enforce(request, bucket="optimizer", limit=30, window_seconds=60)


def auth_rate_limit(request: Request) -> None:
    _enforce(request, bucket="auth", limit=20, window_seconds=60)
