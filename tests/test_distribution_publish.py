"""
Phase 4b posting error-handling tests — FREE (fake transports, no network).

Covers engine/tools/x_client.py's error classification and engine/agents/
distribution.py's confirm_publish() branching: each of the three specific
X error types gets its own proven behavior (not just "doesn't crash"), plus
the 72h content dedup pre-check and post-success recording, plus proof that
unclassified errors still get the normal retry treatment unchanged.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from engine.agents.distribution import DistributionAgent, StagedBundle
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.models import DistributionPlan, PostingMode
from engine.core.post_history_store import InMemoryPostHistoryStore
from engine.core.rate_limit_queue import InMemoryRateLimitQueue
from engine.core.validation_gate import ValidationGate
from engine.tools.social_apis import StagedPost
from engine.tools.x_client import (
    XAuthError, XDuplicateContentError, XPostError, XRateLimitError, _classify_x_error,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


def _utcnow():
    return datetime.now(timezone.utc)


# ===========================================================================
# _classify_x_error — direct unit tests
# ===========================================================================
def test_classify_x_error():
    check("429 -> XRateLimitError", _classify_x_error(429, {}) is XRateLimitError)
    check("401 -> XAuthError", _classify_x_error(401, {"detail": "invalid token"}) is XAuthError)
    check("403 with 'duplicate' in body -> XDuplicateContentError",
          _classify_x_error(403, {"detail": "You are not allowed to create a Tweet with duplicate content"})
          is XDuplicateContentError)
    check("403 WITHOUT 'duplicate' in body -> XAuthError (ambiguous case defaults to auth)",
          _classify_x_error(403, {"detail": "Forbidden"}) is XAuthError)
    check("case-insensitive duplicate match",
          _classify_x_error(403, {"detail": "DUPLICATE content detected"}) is XDuplicateContentError)
    check("500 -> generic XPostError fallback", _classify_x_error(500, {}) is XPostError)


# ===========================================================================
# confirm_publish() branching — fake adapter + fake http, no real network
# ===========================================================================
class FakeXAdapter:
    """Minimal PlatformAdapter stand-in. `outcome` controls what publish()
    does: an exception instance to raise, or a dict to return as success."""
    name = "x"
    live = True
    requires_setup = None

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def publish(self, staged, *, http, confirmed):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _bundle(text="Epirus just won a $66M Army contract."):
    staged = StagedPost(platform="x", payload={"posts": [{"text": text}], "link_mode": "reply"},
                        contains_url=False, estimated_cost_usd=0.0)
    plan = DistributionPlan(posting_mode=PostingMode.confirm, platforms=["x"], staged=True, published=False)
    return StagedBundle(plan=plan, staged={"x": staged})


def _agent(adapter, *, post_history=None, rate_limit_queue=None, dl=None):
    return DistributionAgent([adapter], CostGuard(daily_budget_usd=25, cost_sink=InMemoryCostSink()),
                             ValidationGate(InMemoryDeadLetterSink()), "madre_de_maquinas",
                             dead_letter_sink=dl, post_history_store=post_history,
                             rate_limit_queue=rate_limit_queue)


async def test_auth_error_halts_no_retry():
    adapter = FakeXAdapter(XAuthError(401, {"detail": "invalid token"}))
    dl = InMemoryDeadLetterSink()
    agent = _agent(adapter, dl=dl)
    r = await agent.confirm_publish(bundle=_bundle(), platform="x", http=object(),
                                    confirmed=True, job_id="j1")
    check("auth error -> published=False", r.published is False)
    check("auth error -> exactly 1 attempt, no retry", adapter.calls == 1, str(adapter.calls))
    check("auth error -> dead-lettered", len(dl.records) == 1)
    check("auth error -> detail mentions halted/no retry", "no retry" in r.detail.lower())


async def test_duplicate_content_api_rejection_skips_no_retry():
    adapter = FakeXAdapter(XDuplicateContentError(403, {"detail": "duplicate content"}))
    dl = InMemoryDeadLetterSink()
    agent = _agent(adapter, dl=dl)
    r = await agent.confirm_publish(bundle=_bundle(), platform="x", http=object(),
                                    confirmed=True, job_id="j2")
    check("duplicate content (API) -> published=False", r.published is False)
    check("duplicate content (API) -> exactly 1 attempt, no retry", adapter.calls == 1)
    check("duplicate content (API) -> dead-lettered", len(dl.records) == 1)
    check("duplicate content (API) -> detail mentions skipped/no retry", "no retry" in r.detail.lower())


async def test_rate_limit_queued_not_blocked_not_resumed():
    adapter = FakeXAdapter(XRateLimitError(429, {}))
    dl = InMemoryDeadLetterSink()
    rlq = InMemoryRateLimitQueue()
    agent = _agent(adapter, dl=dl, rate_limit_queue=rlq)
    before = _utcnow()
    r = await agent.confirm_publish(bundle=_bundle(), platform="x", http=object(),
                                    confirmed=True, job_id="j3")
    check("rate limit -> published=False", r.published is False)
    check("rate limit -> exactly 1 attempt, no inline retry (returns immediately)", adapter.calls == 1)
    check("rate limit -> recorded in the rate-limit queue", len(rlq.records) == 1)
    rec = rlq.records[0]
    check("queued record carries job_id/platform", rec.job_id == "j3" and rec.platform == "x")
    delta = rec.retry_at - before
    check("retry_at is ~15 minutes out", timedelta(minutes=14) < delta < timedelta(minutes=16), str(delta))
    check("rate limit -> dead-lettered with the 'not auto-resumed' note",
          len(dl.records) == 1 and "not auto-resumed" in dl.records[0]["error"].lower())
    check("PublishResult detail also says not auto-resumed", "not auto-resumed" in r.detail.lower())


async def test_72h_precheck_skips_before_even_attempting():
    adapter = FakeXAdapter({"ids": ["should-never-be-called"]})
    ph = InMemoryPostHistoryStore()
    text = "Epirus just won a $66M Army contract."
    await ph.record_published(content=text, published_at=_utcnow())
    dl = InMemoryDeadLetterSink()
    agent = _agent(adapter, post_history=ph, dl=dl)
    r = await agent.confirm_publish(bundle=_bundle(text=text), platform="x", http=object(),
                                    confirmed=True, job_id="j4")
    check("72h pre-check duplicate -> published=False", r.published is False)
    check("72h pre-check duplicate -> adapter.publish() NEVER CALLED (pre-check, not post-hoc)",
          adapter.calls == 0, str(adapter.calls))
    check("72h pre-check duplicate -> dead-lettered", len(dl.records) == 1)


async def test_72h_precheck_allows_different_content():
    adapter = FakeXAdapter({"ids": ["tid1", "tid2"]})
    ph = InMemoryPostHistoryStore()
    await ph.record_published(content="a totally different post", published_at=_utcnow())
    agent = _agent(adapter, post_history=ph)
    r = await agent.confirm_publish(bundle=_bundle(text="a brand new post"), platform="x",
                                    http=object(), confirmed=True, job_id="j5")
    check("different content -> not blocked by the 72h check", r.published is True)
    check("different content -> adapter.publish() was called", adapter.calls == 1)


async def test_successful_publish_recorded_for_future_dedup():
    adapter = FakeXAdapter({"ids": ["tid1"]})
    ph = InMemoryPostHistoryStore()
    text = "Epirus just won a $66M Army contract."
    agent = _agent(adapter, post_history=ph)
    r = await agent.confirm_publish(bundle=_bundle(text=text), platform="x", http=object(),
                                    confirmed=True, job_id="j6")
    check("publish succeeded", r.published is True)
    check("successful publish recorded for future 72h dedup checks",
          await ph.was_content_posted_recently(content=text, within_hours=72))


async def test_unclassified_error_still_gets_normal_retry():
    # a plain, non-X exception -- must NOT be swallowed by the stop_on carve-out
    class TransientNetworkError(Exception):
        pass

    adapter = FakeXAdapter(TransientNetworkError("connection reset"))
    dl = InMemoryDeadLetterSink()
    agent = _agent(adapter, dl=dl)
    r = await agent.confirm_publish(bundle=_bundle(), platform="x", http=object(),
                                    confirmed=True, job_id="j7")
    check("unclassified error -> published=False (eventually gives up)", r.published is False)
    check("unclassified error -> retried the normal 3 times, not short-circuited to 1",
          adapter.calls == 3, str(adapter.calls))
    check("unclassified error -> still dead-lettered exactly once (not lost, not doubled)",
          len(dl.records) == 1, str(dl.records))


async def main() -> int:
    test_classify_x_error()
    await test_auth_error_halts_no_retry()
    await test_duplicate_content_api_rejection_skips_no_retry()
    await test_rate_limit_queued_not_blocked_not_resumed()
    await test_72h_precheck_skips_before_even_attempting()
    await test_72h_precheck_allows_different_content()
    await test_successful_publish_recorded_for_future_dedup()
    await test_unclassified_error_still_gets_normal_retry()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_distribution_publish():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_distribution_publish`)
    is unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "distribution-publish suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
