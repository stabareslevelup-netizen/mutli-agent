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
from typing import Any, Optional

from engine.core.cost_guard import CostGuard
from engine.core.models import DistributionPlan, PostingMode
from engine.core.resilience import CircuitBreaker, RetryConfig, resilient_call
from engine.core.validation_gate import ValidationGate
from engine.tools.social_apis import PlatformAdapter, PublishBlocked, StagedPost


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
                 enable_auto: bool = False):
        self._adapters = adapters
        self._cost = cost_guard
        self._gate = gate
        self._brand_id = brand_id
        self._dl = dead_letter_sink
        self._enable_auto = enable_auto   # dormant; must be flipped on deliberately

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

    async def confirm_publish(self, *, bundle: StagedBundle, platform: str, http: Any,
                              confirmed: bool, job_id: str,
                              breaker: Optional[CircuitBreaker] = None) -> PublishResult:
        """The ONLY publish path. Requires confirmed=True (human) and a live adapter."""
        if not confirmed:
            raise PublishBlocked("nothing publishes without an explicit human confirm")
        adapter = next((a for a in self._adapters if a.name == platform), None)
        if adapter is None or platform not in bundle.staged:
            raise PublishBlocked(f"no staged post for platform '{platform}'")
        if not adapter.live:
            raise PublishBlocked(adapter.requires_setup or f"{platform} not live")

        staged = bundle.staged[platform]
        brk = breaker or CircuitBreaker(name=f"publish.{platform}")
        retry = RetryConfig(retries=3, base_delay=1.0, max_delay=8.0)

        async def _do():
            return await adapter.publish(staged, http=http, confirmed=True)

        try:
            detail = await resilient_call(_do, breaker=brk, retry=retry, job_id=job_id,
                                          step=f"publish.{platform}", dead_letter_sink=self._dl)
        except Exception as exc:
            return PublishResult(platform=platform, published=False, detail=str(exc))
        return PublishResult(platform=platform, published=True, detail=detail)
