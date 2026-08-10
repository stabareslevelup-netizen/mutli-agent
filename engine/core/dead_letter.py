"""
engine/core/dead_letter.py — failure capture for recover/diagnose.

Every failed step writes here with enough context to resume or diagnose. The
sink is an injectable Protocol so the rails can be tested without a live DB
(InMemory) and wired to Postgres in production (Sql).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class DeadLetterSink(Protocol):
    async def record(self, *, job_id: Optional[str], step: str, error: str,
                     context: Optional[dict] = None) -> None: ...


class InMemoryDeadLetterSink:
    """Test/dev sink. Holds records in a list."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(self, *, job_id: Optional[str], step: str, error: str,
                     context: Optional[dict] = None) -> None:
        self.records.append({
            "job_id": job_id, "step": step, "error": error,
            "context": context or {}, "created_at": _utcnow(),
        })


class SqlDeadLetterSink:
    """Production sink. Writes a row to the dead_letter table."""

    def __init__(self, sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def record(self, *, job_id: Optional[str], step: str, error: str,
                     context: Optional[dict] = None) -> None:
        from engine.core.database import DeadLetter
        async with self._sessionmaker() as session:
            session.add(DeadLetter(job_id=job_id, step=step, error=error,
                                   context=context or {}))
            await session.commit()
