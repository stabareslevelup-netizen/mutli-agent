"""
engine/core/narrative_conflict_sink.py — narrative-conflict review queue.

Separate from dead_letter.py on purpose: a narrative conflict means the
source item is real and the pipeline worked correctly — it's flagging that
the chosen angle would contradict something the account has previously
stated. That's not a failure to discard, it's a human decision to surface
(same Protocol/InMemory shape as dead_letter.py; a Sql-backed variant can
follow once there's a real conflict-review table to write it to).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class NarrativeConflictSink(Protocol):
    async def record(self, *, job_id: str, item_id: str, chosen_angle: str,
                     conflict_note: str, source_url: str) -> None: ...


class InMemoryNarrativeConflictSink:
    """Test/dev sink. Holds records in a list for the review queue to read."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(self, *, job_id: str, item_id: str, chosen_angle: str,
                     conflict_note: str, source_url: str) -> None:
        self.records.append({
            "job_id": job_id, "item_id": item_id, "chosen_angle": chosen_angle,
            "conflict_note": conflict_note, "source_url": source_url,
            "created_at": _utcnow(),
        })
