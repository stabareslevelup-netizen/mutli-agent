"""
engine/core/review_store.py — holds staged jobs for the review dashboard.

A staged job's screenshot-able ReviewItem (content + agent attribution) and its
distribution StagedBundle are kept here so the dashboard can list the queue,
render a job, and act (approve -> confirm publish; reject -> dead-letter).
InMemory for dev/tests; SqlReviewStore for production (same async-SQLAlchemy
pattern as SqlJobStore/SqlDeadLetterSink in job_manager.py/dead_letter.py --
sessionmaker injected, ORM model imported lazily inside each method). This is
the piece that lets a separate process (e.g. a scheduled sweep script) stage
something the long-lived dashboard server can actually show a human --
without it, two separate Python processes each hold their own in-memory
queue and never see each other's work.
"""
from __future__ import annotations

import dataclasses
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
    async def list_all(self) -> list[ReviewItem]: ...
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

    async def list_all(self) -> list[ReviewItem]:
        # every item ever staged, regardless of status -- see
        # engine/core/pillar_report.py, the one caller that needs this.
        # NOTE: once PR #8 (SqlReviewStore) merges, it needs the same method
        # (an unfiltered `select(db.Review)`) -- it doesn't exist on main yet
        # so it can't be added here.
        return [s.item for s in self._items.values()]

    async def set_status(self, job_id: str, status: str) -> None:
        s = self._items.get(job_id)
        if s:
            s.item.status = status


def _serialize_bundle(bundle: Any) -> Optional[dict]:
    """bundle is a distribution StagedBundle -- a plain dataclass wrapping a
    DistributionPlan (Pydantic) and a dict of social_apis.StagedPost (plain
    dataclasses, not Pydantic). No single .model_dump() covers this shape,
    so it's assembled by hand: .model_dump() for the Pydantic piece,
    dataclasses.asdict() for the plain-dataclass pieces."""
    if bundle is None:
        return None
    return {
        "plan": bundle.plan.model_dump(mode="json"),
        "staged": {platform: dataclasses.asdict(sp) for platform, sp in bundle.staged.items()},
    }


def _deserialize_bundle(data: Optional[dict]) -> Any:
    if data is None:
        return None
    from engine.agents.distribution import StagedBundle
    from engine.core.models import DistributionPlan
    from engine.tools.social_apis import StagedPost
    plan = DistributionPlan.model_validate(data["plan"])
    staged = {platform: StagedPost(**sp) for platform, sp in data["staged"].items()}
    return StagedBundle(plan=plan, staged=staged)


class SqlReviewStore:
    def __init__(self, sessionmaker) -> None:
        self._sm = sessionmaker

    async def add(self, *, item: ReviewItem, bundle: Any = None) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            s.add(db.Review(job_id=item.job_id, brand_id=item.brand_id, status=item.status,
                            item=item.model_dump(mode="json"), bundle=_serialize_bundle(bundle)))
            await s.commit()

    async def get(self, job_id: str) -> Optional[StoredReview]:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.Review, job_id)
            if row is None:
                return None
            return StoredReview(item=ReviewItem.model_validate(row.item),
                                bundle=_deserialize_bundle(row.bundle))

    async def list_staged(self) -> list[ReviewItem]:
        from sqlalchemy import select
        from engine.core import database as db
        async with self._sm() as s:
            stmt = select(db.Review).where(db.Review.status == "staged_for_review")
            rows = (await s.execute(stmt)).scalars().all()
            return [ReviewItem.model_validate(r.item) for r in rows]

    async def set_status(self, job_id: str, status: str) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.Review, job_id)
            if row:
                row.status = status
                # reassign the whole dict (not row.item["status"] = status in
                # place) -- SQLAlchemy's change-tracking for JSON columns
                # doesn't reliably detect in-place mutation of a mutable
                # dict, only attribute reassignment. Keeps the JSON blob's
                # embedded status in sync with the indexed column, matching
                # InMemoryReviewStore's behavior (it mutates item.status
                # directly, so a later get() must see the update too).
                item_dict = dict(row.item)
                item_dict["status"] = status
                row.item = item_dict
                await s.commit()
