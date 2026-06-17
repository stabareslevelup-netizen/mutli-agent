"""
engine/triggers/manual_ui.py — manual trigger.

The thinnest possible entry point: a human (or the review UI / an HTTP POST)
fires a job by topic. Wires straight to Orchestrator.run. Voice trigger is
deferred to v2 per spec; this and the calendar cron are the v1 triggers.
"""
from __future__ import annotations

from typing import Optional

from engine.agents.orchestrator import JobResult, Orchestrator


class ManualTrigger:
    name = "manual"

    def __init__(self, orchestrator: Orchestrator):
        self._orch = orchestrator

    async def fire(self, *, topic: str, entity: Optional[str] = None) -> JobResult:
        return await self._orch.run(topic=topic, entity=entity, trigger=self.name)
