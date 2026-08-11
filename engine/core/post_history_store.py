"""
engine/core/post_history_store.py — queryable record of staged items, for
Timing's duplicate_check and narrative_gap_check.

Deliberately NOT EpisodicMemory: different semantics (exact source_url
match + time window, not top-k vector similarity), different write cadence
(once per staged item, not per arbitrary "content happened" event),
different consumer (Timing, not Strategy's narrative-conflict check).

Recorded at STAGE time (inside Orchestrator._process_item(), right after
distribution.stage_v2() succeeds) — not at publish time — so the same
source_url doesn't re-enter tomorrow's sweep while today's staged item is
still sitting unreviewed in the queue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, runtime_checkable

_TOPIC_SIMILARITY_THRESHOLD = 0.85   # cosine similarity above this = "same topic" (tunable — picked, not spec'd)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@runtime_checkable
class PostHistoryStore(Protocol):
    async def record(self, *, source_url: str, item_id: str, posted_at: datetime,
                     slot: str, post_angle_embedding: Optional[list[float]] = None) -> None: ...
    async def was_posted_recently(self, *, source_url: str, within_hours: int = 72) -> bool: ...
    async def topic_posted_recently(self, *, post_angle_embedding: Optional[list[float]],
                                    within_hours: int = 48) -> bool: ...


@dataclass
class _PostHistoryRecord:
    source_url: str
    item_id: str
    posted_at: datetime
    slot: str
    post_angle_embedding: Optional[list[float]] = None


class InMemoryPostHistoryStore:
    """Test/dev store. O(n) scan — fine at this scale (a Sql-backed variant
    can follow later, same deferral choice as narrative_conflict_sink.py)."""

    def __init__(self) -> None:
        self._records: list[_PostHistoryRecord] = []

    async def record(self, *, source_url: str, item_id: str, posted_at: datetime,
                     slot: str, post_angle_embedding: Optional[list[float]] = None) -> None:
        self._records.append(_PostHistoryRecord(
            source_url=source_url, item_id=item_id, posted_at=posted_at,
            slot=slot, post_angle_embedding=post_angle_embedding))

    async def was_posted_recently(self, *, source_url: str, within_hours: int = 72) -> bool:
        cutoff = _utcnow() - timedelta(hours=within_hours)
        return any(r.source_url == source_url and r.posted_at >= cutoff for r in self._records)

    async def topic_posted_recently(self, *, post_angle_embedding: Optional[list[float]],
                                    within_hours: int = 48) -> bool:
        if post_angle_embedding is None:
            return False   # nothing to compare — graceful degrade, not a match
        cutoff = _utcnow() - timedelta(hours=within_hours)
        for r in self._records:
            if r.posted_at < cutoff or r.post_angle_embedding is None:
                continue
            if _cosine(post_angle_embedding, r.post_angle_embedding) >= _TOPIC_SIMILARITY_THRESHOLD:
                return True
        return False
