"""
engine/agents/orchestrator.py — Orchestrator [PROVEN].

Activates tiers, manages job state, routes, logs. NEVER generates content.

  Tier 1 (parallel):  Research || Memory(query) || Timing
  Tier 2:             fusion -> Strategy -> StrategyPacket  (narrative = hard constraint)
  Tier 3 (parallel):  Copy || Prompt Engineer
  Tier 4:             Production (render) -> Quality (gate + route)
  Tier 5:             Distribution (stage; confirm-mode -> nothing publishes)

Validation gates live inside each agent (malformed handoff -> HandoffHalted ->
job fails, dead-letter already written). Budget is checked up front: when over
the daily cap, non-critical generation is halted before any spend. Per-job cost
is the spent_today delta across the run.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.agents.strategy import StrategyBlocked
from engine.core.brand_loader import BrandConfig
from engine.core.cost_guard import CostGuard
from engine.core.fusion import fuse
from engine.core.job_manager import JobStore
from engine.core.models import PostingMode, QualityRoute
from engine.core.validation_gate import HandoffHalted


@dataclass
class OrchestratorAgents:
    research: Any
    memory: Any
    timing: Any
    strategy: Any
    copy: Any
    prompt_engineer: Any
    production: Any
    quality: Any
    distribution: Any


@dataclass
class JobResult:
    job_id: str
    status: str
    reason: Optional[str] = None
    artifacts: dict = field(default_factory=dict)
    cost_usd: float = 0.0


class Orchestrator:
    name = "orchestrator"

    def __init__(self, *, agents: OrchestratorAgents, job_store: JobStore,
                 cost_guard: CostGuard, brand: BrandConfig):
        self._a = agents
        self._jobs = job_store
        self._cost = cost_guard
        self._brand = brand

    async def run(self, *, topic: str, entity: Optional[str] = None,
                  trigger: str = "manual", critical: bool = False) -> JobResult:
        brand = self._brand
        job = await self._jobs.create(brand_id=brand.brand_id, trigger=trigger,
                                      payload={"topic": topic, "entity": entity})
        jid = job.id
        spent_before = self._cost.spent_today()

        def result(status, reason=None, **artifacts):
            return JobResult(job_id=jid, status=status, reason=reason, artifacts=artifacts,
                             cost_usd=round(self._cost.spent_today() - spent_before, 6))

        # budget gate — halt non-critical generation before any spend
        if not self._cost.can_spend(0.0, critical=critical):
            await self._jobs.update(jid, status="halted_budget")
            return result("halted_budget", reason="daily budget cap reached")

        try:
            # --- Tier 1: parallel intelligence ------------------------------
            await self._jobs.update(jid, status="running", current_tier=1)
            research, memory, timing = await asyncio.gather(
                self._a.research.run(topic=topic, job_id=jid),
                self._a.memory.query(brand_id=brand.brand_id, text=topic, k=5),
                self._a.timing.run(topic=topic, job_id=jid,
                                   pillars=[p.model_dump() for p in brand.pillars],
                                   brand_aliases=[brand.display_name, brand.brand_id,
                                                  brand.character.name],
                                   entity=entity),
            )

            # --- Tier 2: fusion + Strategy (hard narrative constraint) ------
            await self._jobs.update(jid, current_tier=2)
            fused = fuse(research=research, memory=memory, timing=timing,
                         weights=brand.fusion_weights)
            packet = await self._a.strategy.decide(
                fused=fused, constraints=memory.narrative, brand=brand, job_id=jid)

            # --- Tier 3: parallel production inputs --------------------------
            await self._jobs.update(jid, current_tier=3)
            copy, prompt = await asyncio.gather(
                self._a.copy.run(packet=packet, brand=brand, job_id=jid),
                self._a.prompt_engineer.run(packet=packet, brand=brand, job_id=jid),
            )

            # --- Tier 4: Production then Quality gate -------------------------
            await self._jobs.update(jid, current_tier=4)
            asset = await self._a.production.run(
                prompt=prompt, character_element_id=brand.character.higgsfield_element_id,
                job_id=jid)
            has_video = asset.status == "ready"
            quality, auto_eligible = await self._a.quality.evaluate(
                content=copy, brand=brand, job_id=jid, has_video=has_video)

            base = dict(research=research, memory=memory, timing=timing, packet=packet,
                        copy=copy, prompt=prompt, asset=asset, quality=quality,
                        auto_eligible=auto_eligible)

            if quality.route == QualityRoute.reject:
                await self._jobs.update(jid, status="quality_rejected")
                return result("quality_rejected", reason="quality below bar", **base)
            if quality.route == QualityRoute.revise:
                await self._jobs.update(jid, status="needs_revision")
                return result("needs_revision", reason="quality flagged for revision", **base)

            # --- Tier 5: Distribution (confirm-mode: stage only) -------------
            await self._jobs.update(jid, current_tier=5)
            content = {"x_thread": copy.x_thread, "ig_caption": copy.ig_caption,
                       "youtube_script": copy.youtube_script, "asset_url": asset.asset_url,
                       "media_url": asset.asset_url}
            bundle = await self._a.distribution.stage(
                content=content, posting_mode=brand.posting_mode, job_id=jid)

            await self._jobs.update(jid, status="staged_for_review")
            return result("staged_for_review", **base, bundle=bundle)

        except HandoffHalted as exc:
            await self._jobs.update(jid, status="failed")
            return result("failed", reason=f"handoff halted at {exc.step}: {exc.reason}")
        except StrategyBlocked as exc:
            await self._jobs.update(jid, status="halted_narrative_conflict")
            return result("halted_narrative_conflict", reason=str(exc))
