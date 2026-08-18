"""
engine/agents/distribution.py — Distribution agent [PROVEN engine / RISK-MANAGED action].

v1 is posting_mode=confirm ONLY: it stages every platform path and NOTHING
publishes without an explicit human confirm. Auto/scheduled modes are built but
dormant and refuse to run unless deliberately enabled (off by default,
unreachable in tests).

- staging logs per-post X cost via cost_guard.record_external
- publishing (after human confirm) goes through resilience retry + circuit
  breaker; failures land in the dead-letter, never an infinite loop
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from engine.core.cost_guard import CostGuard
from engine.core.models import DistributionPlan, PostingMode
from engine.core.models import StagedPost as StagedPostRecord   # avoid collision with
                                                                  # social_apis.StagedPost below
from engine.core.resilience import CircuitBreaker, RetryConfig, resilient_call
from engine.core.validation_gate import ValidationGate
from engine.tools.social_apis import PlatformAdapter, PublishBlocked, StagedPost
from engine.tools.x_client import XAuthError, XDuplicateContentError, XRateLimitError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _extract_main_post_text(staged: StagedPost) -> Optional[str]:
    """Best-effort extraction of the primary post text from a staged
    platform-adapter payload, for the 72h content dedup check (Phase 4b).
    Only XAdapter's payload shape ({"posts": [...]}) is recognized -- other
    platforms' payloads don't carry a comparable field, so this returns
    None rather than assume a shape that isn't there (and the dedup check
    is skipped gracefully for those platforms, see confirm_publish())."""
    posts = staged.payload.get("posts") if isinstance(staged.payload, dict) else None
    if not posts:
        return None
    return posts[0].get("text")


class PostingModeDisabled(Exception):
    """auto/scheduled requested but not explicitly enabled in v1."""


@dataclass
class StagedBundle:
    plan: DistributionPlan
    staged: dict[str, StagedPost]   # platform -> staged post


@dataclass
class PublishResult:
    platform: str
    published: bool
    detail: Any = None


class DistributionAgent:
    name = "distribution"

    def __init__(self, adapters: list[PlatformAdapter], cost_guard: CostGuard,
                 gate: ValidationGate, brand_id: str, dead_letter_sink=None,
                 enable_auto: bool = False, post_history_store=None, rate_limit_queue=None):
        self._adapters = adapters
        self._cost = cost_guard
        self._gate = gate
        self._brand_id = brand_id
        self._dl = dead_letter_sink
        self._enable_auto = enable_auto   # dormant; must be flipped on deliberately
        self._post_history = post_history_store   # Phase 4b: 72h content dedup at publish time
        self._rate_limit_queue = rate_limit_queue  # Phase 4b: records XRateLimitError, doesn't resume it

    async def stage(self, *, content: dict, posting_mode: PostingMode, job_id: str) -> StagedBundle:
        # auto/scheduled are dormant in v1 unless explicitly enabled
        if posting_mode in (PostingMode.auto, PostingMode.scheduled) and not self._enable_auto:
            raise PostingModeDisabled(f"{posting_mode.value} mode is off by default in v1")

        staged: dict[str, StagedPost] = {}
        for adapter in self._adapters:
            try:
                sp = adapter.stage(content)
            except PublishBlocked as exc:
                # e.g. Instagram missing media URL — record, don't crash the bundle
                if self._dl is not None:
                    await self._dl.record(job_id=job_id, step=f"stage.{adapter.name}",
                                          error=str(exc), context={"platform": adapter.name})
                continue
            staged[adapter.name] = sp
            if sp.estimated_cost_usd:
                await self._cost.record_external(
                    job_id=job_id, brand_id=self._brand_id,
                    agent=f"{self.name}.{adapter.name}", usd=sp.estimated_cost_usd,
                    label=f"{adapter.name}-api")

        plan = DistributionPlan(posting_mode=posting_mode,
                                platforms=list(staged.keys()),
                                staged=True, published=False)  # confirm => never published here
        validated = await self._gate.validate(DistributionPlan, plan.model_dump(),
                                               job_id=job_id, step="distribution.stage")
        return StagedBundle(plan=validated, staged=staged)

    async def stage_v2(self, *, staged: StagedPostRecord, job_id: str) -> StagedBundle:
        """Additive Phase 2 method. Reuses XAdapter unchanged, which already
        chains every post sequentially by real tweet ID (verified in
        x_client.py's post_thread()) -- gets the spec's "two-call posting
        sequence" (or full N-call chain for a thread) for free.

        NEVER publishes. requires_human_approval on `staged` is Literal[True]
        by construction, not something this method has to enforce -- a
        StagedPostRecord with it False is unconstructable. The actual post
        only happens later, via confirm_publish() below (unchanged),
        triggered by a human clicking Approve on the review dashboard.
        """
        content = {
            "x_thread": staged.thread_tweets or [staged.main_post],
            "link": staged.reply_link,
        }
        x_adapter = next((a for a in self._adapters if a.name == "x"), None)
        if x_adapter is None:
            raise PublishBlocked("no X adapter configured")
        sp = x_adapter.stage(content)
        if sp.estimated_cost_usd:
            await self._cost.record_external(
                job_id=job_id, brand_id=self._brand_id, agent=f"{self.name}.x",
                usd=sp.estimated_cost_usd, label="x-api")
        plan = DistributionPlan(posting_mode=PostingMode.confirm, platforms=["x"],
                                staged=True, published=False)
        validated = await self._gate.validate(DistributionPlan, plan.model_dump(),
                                              job_id=job_id, step="distribution.stage_v2")
        return StagedBundle(plan=validated, staged={"x": sp})

    async def confirm_publish(self, *, bundle: StagedBundle, platform: str, http: Any,
                              confirmed: bool, job_id: str,
                              breaker: Optional[CircuitBreaker] = None) -> PublishResult:
        """The ONLY publish path. Requires confirmed=True (human) and a live adapter.

        Phase 4b error handling: XAuthError halts immediately (no retry,
        logged). XDuplicateContentError skips immediately (no retry,
        logged). XRateLimitError is recorded as queued-for-retry-at-T+15min
        and returns immediately WITHOUT blocking this call -- nothing
        currently re-drives that queue automatically (no scheduler exists
        in this codebase, same gap as run_sweep()'s own trigger). This is
        explicitly a recording mechanism, not a resumption one -- logged
        clearly as such, not silently dropped. Everything else still goes
        through the existing generic retry/circuit-breaker path unchanged
        (stop_on excludes only the three specific types above).

        A 72-hour exact-content dedup check runs BEFORE attempting to post
        at all (PostHistoryStore.was_content_posted_recently) -- distinct
        from, and in addition to, XDuplicateContentError handling: the
        pre-check avoids even trying when we already know; the error
        handler is defense-in-depth for whatever the pre-check's exact
        string match might miss (e.g. two staged items racing each other).
        """
        if not confirmed:
            raise PublishBlocked("nothing publishes without an explicit human confirm")
        adapter = next((a for a in self._adapters if a.name == platform), None)
        if adapter is None or platform not in bundle.staged:
            raise PublishBlocked(f"no staged post for platform '{platform}'")
        if not adapter.live:
            raise PublishBlocked(adapter.requires_setup or f"{platform} not live")

        staged = bundle.staged[platform]
        content = _extract_main_post_text(staged)   # None for non-X-shaped payloads

        if self._post_history is not None and content is not None:
            if await self._post_history.was_content_posted_recently(content=content, within_hours=72):
                if self._dl is not None:
                    await self._dl.record(job_id=job_id, step=f"publish.{platform}",
                                          error="duplicate content within 72h (pre-check, not attempted)",
                                          context={"platform": platform})
                return PublishResult(platform=platform, published=False,
                                     detail="duplicate content within 72h, skipped (pre-check)")

        brk = breaker or CircuitBreaker(name=f"publish.{platform}")
        retry = RetryConfig(retries=3, base_delay=1.0, max_delay=8.0,
                            stop_on=(XAuthError, XDuplicateContentError, XRateLimitError))

        async def _do():
            return await adapter.publish(staged, http=http, confirmed=True)

        try:
            # dead_letter_sink=None here deliberately: resilient_call would
            # otherwise write its own generic dead-letter record for ANY
            # propagated exception (see resilience.py), duplicating the
            # more specific ones each except clause below writes for the
            # three classified types. confirm_publish() owns dead-lettering
            # for this call, in exactly one place, below.
            detail = await resilient_call(_do, breaker=brk, retry=retry, job_id=job_id,
                                          step=f"publish.{platform}", dead_letter_sink=None)
        except XAuthError as exc:
            if self._dl is not None:
                await self._dl.record(job_id=job_id, step=f"publish.{platform}",
                                      error=f"auth failure -- halting, no retry: {exc}",
                                      context={"platform": platform})
            return PublishResult(platform=platform, published=False,
                                 detail=f"auth failure, halted, no retry: {exc}")
        except XDuplicateContentError as exc:
            if self._dl is not None:
                await self._dl.record(job_id=job_id, step=f"publish.{platform}",
                                      error=f"duplicate content (API-rejected), no retry: {exc}",
                                      context={"platform": platform})
            return PublishResult(platform=platform, published=False,
                                 detail=f"duplicate content, skipped, no retry: {exc}")
        except XRateLimitError as exc:
            retry_at = _utcnow() + timedelta(minutes=15)
            if self._rate_limit_queue is not None:
                await self._rate_limit_queue.record(job_id=job_id, platform=platform, retry_at=retry_at)
            if self._dl is not None:
                await self._dl.record(
                    job_id=job_id, step=f"publish.{platform}",
                    error=(f"rate limited -- queued for retry at {retry_at.isoformat()}, "
                          f"NOT auto-resumed (no scheduler exists yet): {exc}"),
                    context={"platform": platform, "retry_at": retry_at.isoformat()})
            return PublishResult(
                platform=platform, published=False,
                detail=f"rate limited, queued for retry at {retry_at.isoformat()} (not auto-resumed)")
        except Exception as exc:
            # unclassified failure (already retried the normal number of times
            # by resilient_call above) -- restore the dead-letter write that
            # resilient_call would have done itself, had dead_letter_sink not
            # been suppressed above to avoid double-writing the classified cases
            if self._dl is not None:
                await self._dl.record(job_id=job_id, step=f"publish.{platform}",
                                      error=f"{type(exc).__name__}: {exc}",
                                      context={"platform": platform, "breaker_state": brk.state})
            return PublishResult(platform=platform, published=False, detail=str(exc))

        if self._post_history is not None and content is not None:
            await self._post_history.record_published(content=content, published_at=_utcnow())
        return PublishResult(platform=platform, published=True, detail=detail)
