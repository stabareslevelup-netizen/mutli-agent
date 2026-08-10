"""
engine/triggers/calendar_cron.py — daily calendar trigger.

Default cadence is ONE JOB PER DAY (not multi-job frequency): the cron fires at
most once per calendar day, at a target time.

Fire time is dynamic *where possible*: an injected `window_provider` can return
an optimal publish hour (UTC) — this is the seam for an audience-timing/Timing
recommendation. When none is available it falls back to a configurable daily
time (`default_time`, "HH:MM"). Note: v1 Timing emits topic intelligence, not a
time-of-day window, so the default time is used until an analytics-backed
window provider is wired (Phase 9+ engagement data). We don't fabricate a clock
time from velocity — that would be fake precision.

The topic for the daily job comes from an injected `topic_provider` (e.g. the
content_calendar's plan for the day, or a static default).

Everything is clock-injectable so the due logic is testable without waiting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Tuple

from engine.agents.orchestrator import JobResult, Orchestrator

# topic_provider() -> (topic, entity)
TopicProvider = Callable[[], Awaitable[Tuple[str, Optional[str]]]]
# window_provider() -> optimal UTC hour 0..23, or None to use the default time
WindowProvider = Callable[[], Awaitable[Optional[int]]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def static_topic_provider(topic: str, entity: Optional[str] = None) -> TopicProvider:
    async def _p() -> Tuple[str, Optional[str]]:
        return topic, entity
    return _p


@dataclass
class CalendarCron:
    name: str = "calendar"
    orchestrator: Optional[Orchestrator] = None
    topic_provider: Optional[TopicProvider] = None
    default_time: str = "09:00"                    # HH:MM, UTC
    window_provider: Optional[WindowProvider] = None
    clock: Callable[[], datetime] = _utcnow
    _last_fired_date: Optional[date] = field(default=None, init=False)

    async def _target_hm(self) -> Tuple[int, int]:
        if self.window_provider is not None:
            try:
                hour = await self.window_provider()
                if hour is not None and 0 <= int(hour) <= 23:
                    return int(hour), 0
            except Exception:
                pass  # provider failure -> fall back to the configured default
        hh, mm = self.default_time.split(":")
        return int(hh), int(mm)

    async def next_fire_at(self, now: Optional[datetime] = None) -> datetime:
        now = now or self.clock()
        h, m = await self._target_hm()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    async def is_due(self, now: Optional[datetime] = None) -> bool:
        now = now or self.clock()
        if self._last_fired_date == now.date():   # one-per-day guard
            return False
        h, m = await self._target_hm()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return now >= target

    async def run_due(self, now: Optional[datetime] = None) -> Optional[JobResult]:
        """Fire the daily job if due; return its JobResult, else None."""
        now = now or self.clock()
        if not await self.is_due(now):
            return None
        topic, entity = await self.topic_provider()
        result = await self.orchestrator.run(topic=topic, entity=entity, trigger=self.name)
        self._last_fired_date = now.date()   # enforce at-most-once-per-day
        return result
