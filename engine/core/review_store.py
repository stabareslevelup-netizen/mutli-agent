"""
engine/core/review_store.py — holds staged jobs for the review dashboard.

A staged job's screenshot-able ReviewItem (content + agent attribution) and its
distribution StagedBundle are kept here so the dashboard can list the queue,
render a job, and act (approve -> confirm publish; reject -> dead-letter).
InMemory for dev/tests; DI seam for a Sql-backed store later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from engine.core.models import ReviewItem


@dataclass
class StoredReview:
    item: ReviewItem
    bundle: Any = None   # distribution StagedBundle, for one-tap approve -> publish


@runtime_checkable
class ReviewStore(Protocol):
    async def add(self, *, item: ReviewItem, bundle: Any = None) -> None: ...
    async def get(self, job_id: str) -> Optional[StoredReview]: ...
    async def list_staged(self) -> list[ReviewItem]: ...
    async def set_status(self, job_id: str, status: str) -> None: ...


class InMemoryReviewStore:
    def __init__(self) -> None:
        self._items: dict[str, StoredReview] = {}

    async def add(self, *, item: ReviewItem, bundle: Any = None) -> None:
        self._items[item.job_id] = StoredReview(item=item, bundle=bundle)

    async def get(self, job_id: str) -> Optional[StoredReview]:
        return self._items.get(job_id)

    async def list_staged(self) -> list[ReviewItem]:
        return [s.item for s in self._items.values() if s.item.status == "staged_for_review"]

    async def set_status(self, job_id: str, status: str) -> None:
        s = self._items.get(job_id)
        if s:
            s.item.status = status
