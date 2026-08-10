"""
engine/agents/orchestrator.py — Orchestrator [PROVEN].

Activates tiers, manages job state, routes, logs. NEVER generates content.

  Tier 1 (parallel):  Research || Memory(query) || Timing
  Tier 2:             fusion -> Strategy -> Skeptic (adversarial review, informs)
  Tier 3:             Copy (timed)
  Tier 4:             Quality (gate + route)
  Tier 5:             Distribution (stage; confirm-mode -> nothing publishes)

(9 agents total: Orchestrator, Research, Memory, Timing, Strategy, Skeptic,
Copy, Quality, Distribution. Text-only brand — no Prompt Engineer / Production
/ Higgsfield: those were removed when video generation was dropped from scope.)

Validation gates live inside each agent (malformed handoff -> HandoffHalted ->
job fails, dead-letter already written). Budget is checked up front: when over
the daily cap, non-critical generation is halted before any spend. Per-job cost
is the spent_today delta across the run.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from engine.agents.strategy import StrategyBlocked
from engine.core.brand_loader import BrandConfig
from engine.core.cost_guard import CostGuard
from engine.core.fusion import fuse
from engine.core.job_manager import JobStore
from engine.core.models import AgentAttribution, PostingMode, QualityRoute, ReviewItem
from engine.core.validation_gate import HandoffHalted


@dataclass
class OrchestratorAgents:
    research: Any
    memory: Any
    timing: Any
    strategy: Any
    skeptic: Any
    copy: Any
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
                 cost_guard: CostGuard, brand: BrandConfig, dead_letter_sink=None,
                 review_store=None, notifier=None):
        self._a = agents
        self._jobs = job_store
        self._cost = cost_guard
        self._brand = brand
        self._dl = dead_letter_sink
        self._reviews = review_store
        self._notifier = notifier   # optional "something's waiting" push alert

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
                                                  brand.character_name],
                                   entity=entity),
            )

            # --- Tier 2: fusion + Strategy + Skeptic (adversarial review) -----
            await self._jobs.update(jid, current_tier=2)
            fused = fuse(research=research, memory=memory, timing=timing,
                         weights=brand.fusion_weights)
            packet = await self._a.strategy.decide(
                fused=fused, constraints=memory.narrative, brand=brand, job_id=jid,
                timing=timing)
            # Skeptic reviews the chosen angle and informs Copy (never blocks)
            skeptic = await self._a.skeptic.review(packet=packet, job_id=jid)
            packet = packet.model_copy(update={
                "skeptic_summary": skeptic.skeptic_summary,
                "confidence_adjustment": skeptic.confidence_adjustment})

            # --- Tier 3: Copy (timed) -----------------------------------------
            await self._jobs.update(jid, current_tier=3)
            t0 = perf_counter()
            copy = await self._a.copy.run(packet=packet, brand=brand, job_id=jid)
            copy_secs = round(perf_counter() - t0, 2)

            # --- Tier 4: Quality gate -------------------------------------------
            await self._jobs.update(jid, current_tier=4)
            quality, auto_eligible = await self._a.quality.evaluate(
                content=copy, brand=brand, job_id=jid)

            base = dict(research=research, memory=memory, timing=timing, packet=packet,
                        copy=copy, quality=quality, auto_eligible=auto_eligible)

            if quality.route == QualityRoute.reject:
                await self._jobs.update(jid, status="quality_rejected")
                return result("quality_rejected", reason="quality below bar", **base)
            if quality.route == QualityRoute.revise:
                await self._jobs.update(jid, status="needs_revision")
                return result("needs_revision", reason="quality flagged for revision", **base)

            # --- Tier 5: Distribution (confirm-mode: stage only) -------------
            await self._jobs.update(jid, current_tier=5)
            content = {"x_thread": copy.x_thread, "ig_caption": copy.ig_caption,
                       "youtube_script": copy.youtube_script}
            bundle = await self._a.distribution.stage(
                content=content, posting_mode=brand.posting_mode, job_id=jid)

            await self._jobs.update(jid, status="staged_for_review")

            # build the screenshot-able review item (content + agent attribution)
            item = self._build_review_item(jid, fused, research, memory, timing,
                                           packet, copy, quality, auto_eligible,
                                           bundle, copy_secs)
            if self._reviews is not None:
                await self._reviews.add(item=item, bundle=bundle)
            if self._notifier is not None:
                # best-effort: wrapped independently of the outer try/except
                # so a misbehaving notifier can never overwrite an already-
                # successful job's status to "failed" (see notify.py — the
                # real SlackNotifier also never raises, but this guarantee
                # must not depend on that implementation detail).
                try:
                    await self._notifier.notify_staged(
                        job_id=jid, chosen_angle=packet.chosen_angle,
                        quality_overall=quality.overall,
                        requires_hedging=packet.requires_hedging, trigger=trigger)
                except Exception:
                    pass

            return result("staged_for_review", **base, bundle=bundle, review_item=item)

        except HandoffHalted as exc:
            await self._jobs.update(jid, status="failed")
            return result("failed", reason=f"handoff halted at {exc.step}: {exc.reason}")
        except StrategyBlocked as exc:
            await self._jobs.update(jid, status="halted_narrative_conflict")
            return result("halted_narrative_conflict", reason=str(exc))
        except Exception as exc:  # defensive: a tool/agent crash must not escape the job
            await self._jobs.update(jid, status="failed")
            if self._dl is not None:
                await self._dl.record(job_id=jid, step="orchestrator",
                                      error=f"{type(exc).__name__}: {exc}", context={})
            return result("failed", reason=f"unexpected error: {type(exc).__name__}: {exc}")

    def _build_review_item(self, jid, fused, research, memory, timing, packet,
                           copy, quality, auto_eligible, bundle, copy_secs) -> ReviewItem:
        alternatives = [fa.angle for fa in fused if fa.angle != packet.chosen_angle][:3]
        mem_ctx = [m.content for m in (*memory.episodic, *memory.semantic)][:4]
        attribution = AgentAttribution(
            research_chosen=packet.chosen_angle,
            research_alternatives=alternatives,
            timing_velocity=timing.velocity.verdict.value,
            timing_velocity_confidence=timing.velocity.confidence,
            timing_gaps=[g.angle for g in timing.gaps.gaps if g.is_open],
            timing_citation=timing.citation.presence.value,
            memory_context=mem_ctx,
            copy_output_seconds=copy_secs,
            quality={"voice": quality.voice, "narrative": quality.narrative,
                     "format": quality.format, "hook": quality.hook,
                     "coherence": quality.coherence, "overall": quality.overall},
            skeptic_summary=packet.skeptic_summary,
        )
        return ReviewItem(
            job_id=jid, brand_id=self._brand.brand_id, status="staged_for_review",
            pillar_id=packet.pillar_id,
            requires_hedging=packet.requires_hedging, chosen_angle=packet.chosen_angle,
            x_thread=copy.x_thread, ig_caption=copy.ig_caption,
            youtube_script=copy.youtube_script,
            quality_overall=quality.overall, quality_route=quality.route.value,
            auto_eligible=auto_eligible, platforms=bundle.plan.platforms,
            attribution=attribution,
        )
