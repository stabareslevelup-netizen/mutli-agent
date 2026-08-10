"""
Phase 7 trigger tests — FREE (fake orchestrator, injected clock, no LLM/HTTP spend).

Covers:
  - ManualTrigger -> Orchestrator.run(trigger="manual")
  - CalendarCron: next_fire_at, is_due, one-job-per-day guard, dynamic window
    override vs configurable default time, topic provider -> orchestrator
  - POST /jobs route wiring (manual trigger) via TestClient with a fake engine
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from engine.agents.orchestrator import JobResult
from engine.triggers.calendar_cron import CalendarCron, static_topic_provider
from engine.triggers.manual_ui import ManualTrigger

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
UTC = timezone.utc
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    async def run(self, *, topic, entity=None, trigger="manual", critical=False):
        self.calls.append({"topic": topic, "entity": entity, "trigger": trigger})
        return JobResult(job_id=f"job-{len(self.calls)}", status="staged_for_review",
                         artifacts={"packet": SimpleNamespace(chosen_angle="A")}, cost_usd=1.23)


async def test_manual_trigger():
    orch = FakeOrchestrator()
    res = await ManualTrigger(orch).fire(topic="robots", entity="Figure")
    check("manual trigger calls Orchestrator.run", len(orch.calls) == 1)
    check("manual trigger passes trigger='manual'", orch.calls[0]["trigger"] == "manual")
    check("manual trigger forwards topic/entity",
          orch.calls[0]["topic"] == "robots" and orch.calls[0]["entity"] == "Figure")
    check("manual trigger returns JobResult", res.status == "staged_for_review")


async def test_calendar_default_time():
    orch = FakeOrchestrator()
    cron = CalendarCron(orchestrator=orch, topic_provider=static_topic_provider("daily topic", "Figure"),
                        default_time="09:00")
    before = datetime(2026, 6, 16, 8, 0, tzinfo=UTC)
    after = datetime(2026, 6, 16, 10, 0, tzinfo=UTC)

    nf = await cron.next_fire_at(before)
    check("next_fire_at = today 09:00 when before target", nf == datetime(2026, 6, 16, 9, 0, tzinfo=UTC), str(nf))
    check("not due before target time", await cron.is_due(before) is False)
    check("due at/after target time", await cron.is_due(after) is True)


async def test_one_job_per_day():
    orch = FakeOrchestrator()
    cron = CalendarCron(orchestrator=orch, topic_provider=static_topic_provider("t", None),
                        default_time="09:00")
    d1_am = datetime(2026, 6, 16, 10, 0, tzinfo=UTC)
    d1_pm = datetime(2026, 6, 16, 20, 0, tzinfo=UTC)
    d2 = datetime(2026, 6, 17, 10, 0, tzinfo=UTC)

    r1 = await cron.run_due(d1_am)
    check("fires when due", r1 is not None and len(orch.calls) == 1)
    check("calendar trigger tag", orch.calls[0]["trigger"] == "calendar")
    r2 = await cron.run_due(d1_pm)
    check("does NOT fire twice same day (one-per-day)", r2 is None and len(orch.calls) == 1)
    next_at = await cron.next_fire_at(d1_pm)
    check("after firing, next fire rolls to tomorrow", next_at.date() == datetime(2026, 6, 17, tzinfo=UTC).date())
    r3 = await cron.run_due(d2)
    check("fires again next day", r3 is not None and len(orch.calls) == 2)


async def test_dynamic_window_override():
    orch = FakeOrchestrator()

    async def window():
        return 14   # optimal hour from a provider (e.g. future analytics/Timing)

    cron = CalendarCron(orchestrator=orch, topic_provider=static_topic_provider("t", None),
                        default_time="09:00", window_provider=window)
    nf = await cron.next_fire_at(datetime(2026, 6, 16, 8, 0, tzinfo=UTC))
    check("dynamic window sets fire hour (14:00, not default 09:00)", nf.hour == 14, str(nf))
    check("not due before dynamic window", await cron.is_due(datetime(2026, 6, 16, 13, 0, tzinfo=UTC)) is False)
    check("due after dynamic window", await cron.is_due(datetime(2026, 6, 16, 15, 0, tzinfo=UTC)) is True)

    # provider failure falls back to default time
    async def bad_window():
        raise RuntimeError("provider down")
    cron2 = CalendarCron(orchestrator=orch, topic_provider=static_topic_provider("t", None),
                         default_time="09:00", window_provider=bad_window)
    nf2 = await cron2.next_fire_at(datetime(2026, 6, 16, 8, 0, tzinfo=UTC))
    check("window provider failure -> falls back to default time", nf2.hour == 9, str(nf2))


def test_post_jobs_route():
    from fastapi.testclient import TestClient
    import main

    fake = FakeOrchestrator()

    def fake_build(brand, **kw):
        # main's lifespan also builds Review + Feedback services from these fields
        return SimpleNamespace(orchestrator=fake, review_store=None, distribution=None,
                               procedural=None, dead_letter=None, memory=(None, None, None))

    with patch("engine.core.assembly.build_orchestrator", fake_build):
        with TestClient(main.app) as c:
            r = c.post("/jobs", json={"topic": "humanoid robots", "entity": "Figure"})
    check("POST /jobs returns 200", r.status_code == 200, str(r.status_code))
    body = r.json() if r.status_code == 200 else {}
    check("POST /jobs fired manual trigger", fake.calls and fake.calls[0]["trigger"] == "manual")
    check("POST /jobs returns job summary", body.get("status") == "staged_for_review")


async def main_async() -> int:
    await test_manual_trigger()
    await test_calendar_default_time()
    await test_one_job_per_day()
    await test_dynamic_window_override()
    test_post_jobs_route()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
