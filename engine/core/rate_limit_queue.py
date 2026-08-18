"""
engine/core/rate_limit_queue.py — records X posts that hit a rate limit, so
a 429 is never silently dropped.

Deliberately NOT a working retry mechanism: nothing in this codebase
currently re-drives this queue after the recorded retry_at time passes —
that would need a scheduler/poller, which doesn't exist yet (same gap as
run_sweep()'s own trigger, never built). This only guarantees the failure
is recorded and visible; it does not resume the post automatically. See
distribution.py's confirm_publish() for where this gets written.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class RateLimitQueue(Protocol):
    async def record(self, *, job_id: str, platform: str, retry_at: datetime) -> None: ...


@dataclass
class _QueuedRetry:
    job_id: str
    platform: str
    retry_at: datetime


class InMemoryRateLimitQueue:
    """Test/dev store. Nothing reads this back yet except tests -- there is
    no poller. See module docstring."""

    def __init__(self) -> None:
        self.records: list[_QueuedRetry] = []

    async def record(self, *, job_id: str, platform: str, retry_at: datetime) -> None:
        self.records.append(_QueuedRetry(job_id=job_id, platform=platform, retry_at=retry_at))
