"""API-key authentication and rate limiting (dependencies applied to /api/v1/*)."""
from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request

from app.observability.metrics import RATE_LIMITED
from app.services.container import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


async def require_api_key(request: Request, svc: Services = Depends(get_services)) -> str:
    keys = svc.settings.api_key_set
    if not keys:  # auth disabled (dev). A warning is logged at startup.
        return "anonymous"
    supplied = request.headers.get("x-api-key", "")
    if not any(hmac.compare_digest(supplied, k) for k in keys):
        raise HTTPException(status_code=401, detail="missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
    return supplied[:6] + "…"


async def rate_limit(request: Request, who: str = Depends(require_api_key), svc: Services = Depends(get_services)) -> None:
    limit = svc.settings.rate_limit_per_minute
    if limit <= 0:
        return
    ident = who if who != "anonymous" else (request.client.host if request.client else "unknown")
    n = await svc.cache.incr_window(f"rl:{ident}", 60)
    if n > limit:
        RATE_LIMITED.inc()
        raise HTTPException(status_code=429, detail=f"rate limit exceeded ({limit}/min)", headers={"Retry-After": "60"})
