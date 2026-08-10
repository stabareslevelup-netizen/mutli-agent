"""
engine/core/job_manager.py — job state lifecycle.

Tracks a job through the tiers and persists state. The store is injectable
(InMemory for tests, Sql for the jobs table) — same DI pattern as the other
backends, so the orchestrator is testable without a live DB.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    brand_id: str
    status: str = "created"
    current_tier: int = 0
    trigger: str = "manual"
    payload: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@runtime_checkable
class JobStore(Protocol):
    async def create(self, *, brand_id: str, trigger: str, payload: dict) -> Job: ...
    async def update(self, job_id: str, **fields) -> None: ...
    async def get(self, job_id: str) -> Optional[Job]: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    async def create(self, *, brand_id: str, trigger: str, payload: dict) -> Job:
        job = Job(id=str(uuid.uuid4()), brand_id=brand_id, trigger=trigger, payload=payload)
        self._jobs[job.id] = job
        return job

    async def update(self, job_id: str, **fields) -> None:
        job = self._jobs[job_id]
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = _utcnow()

    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)


class SqlJobStore:
    def __init__(self, sessionmaker) -> None:
        self._sm = sessionmaker

    async def create(self, *, brand_id: str, trigger: str, payload: dict) -> Job:
        from engine.core import database as db
        jid = str(uuid.uuid4())
        async with self._sm() as s:
            s.add(db.Job(id=jid, brand_id=brand_id, status="created", current_tier=0,
                         trigger=trigger, payload=payload))
            await s.commit()
        return Job(id=jid, brand_id=brand_id, trigger=trigger, payload=payload)

    async def update(self, job_id: str, **fields) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.Job, job_id)
            if row:
                for k, v in fields.items():
                    setattr(row, k, v)
                await s.commit()

    async def get(self, job_id: str) -> Optional[Job]:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.Job, job_id)
            if not row:
                return None
            return Job(id=row.id, brand_id=row.brand_id, status=row.status,
                       current_tier=row.current_tier, trigger=row.trigger,
                       payload=row.payload, created_at=row.created_at, updated_at=row.updated_at)
