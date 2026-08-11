"""
engine/agents/orchestrator.py — Orchestrator [PROVEN].

Activates tiers, manages job state, routes, logs. NEVER generates content.

Phase 2 (X-agent migration) — Research is now a multi-source SWEEP, not a
single-topic lookup, so the pipeline has two levels:

  run_sweep():    Tier 1 batch level — Research (sweep all sources) then
                  Timing (assigns each item a slot, or rejects it, across
                  the WHOLE batch at once — "one item per slot per day" is
                  a cross-item constraint, so this cannot run in parallel
                  with Research the way the old three-way Tier 1 did).

  _process_item(): Tier 2 onward, PER accepted item, each gets its own
                  job_id — this reuses the existing JobStore/review-store
                  machinery unchanged:
                    Memory(query on item.post_angle) -> Strategy -> Skeptic
                    (max 1 'revise' retry; 'reject' hard-halts, no retry)
                    -> Copy -> Quality (max 1 Copy-only retry on 'revise';
                    'reject' sends back to a FRESH Strategy attempt, max 1)
                    -> Distribution (stage_v2; always requires_human_approval)

Two safety mechanisms are preserved as hard invariants (not dropped in the
migration): TimingDecision.citation_hedge_required (Copy must hedge claims)
and StrategyOutput.narrative_conflict_flag (hard-halts immediately, logged
to a SEPARATE review queue from dead_letter — a conflict means the source
is real, not that the pipeline failed; it's a human decision, not a defect).

Validation gates live inside each agent (malformed handoff -> HandoffHalted
-> job fails, dead-letter already written). Budget is checked up front: when
over the daily cap, non-critical generation is halted before any spend.
Per-job cost is the spent_today delta across that item's run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from engine.core.brand_loader import BrandConfig
from engine.core.cost_guard import CostGuard
from engine.core.embeddings import safe_embed
from engine.core.job_manager import JobStore
from engine.core.models import (
    PostingSlot,
    QualityVerdict,
    SkepticVerdict,
    AgentAttribution,
    ReviewItem,
)
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
                 narrative_conflict_sink=None, post_history_store=None,
                 embeddings=None, review_store=None, notifier=None):
        self._a = agents
        self._jobs = job_store
        self._cost = cost_guard
        self._brand = brand
        self._dl = dead_letter_sink
        self._nc = narrative_conflict_sink   # separate from dead_letter — see module docstring
        self._post_history = post_history_store   # duplicate_check / narrative_gap_check for Timing
        self._embeddings = embeddings             # computes post_angle_embedding at stage time;
                                                   # None -> safe_embed degrades gracefully (see below)
        self._reviews = review_store
        self._notifier = notifier            # optional "something's waiting" push alert

    # -----------------------------------------------------------------
    # Tier 1 (batch level): Research sweep -> Timing slot assignment
    # -----------------------------------------------------------------
    async def run_sweep(self) -> list[JobResult]:
        """Research scans all configured sources (SAM.gov, Federal Register,
        Congress, arXiv, DARPA-fallback) and returns candidate items; Timing
        assigns each a posting slot (or rejects it) across the whole batch at
        once. Every accepted item then runs its own full job via
        _process_item() — see module docstring for why this can't be a
        single flat pipeline the way the old topic-in/content-out flow was."""
        items = await self._a.research.sweep(job_id=None)
        decisions = await self._a.timing.assign(items=items, job_id=None)
        by_id = {d.item_id: d for d in decisions}

        accepted = [(item, by_id[item.item_id]) for item in items
                    if by_id.get(item.item_id) is not None
                    and by_id[item.item_id].recommended_slot != PostingSlot.reject]

        results = []
        for item, decision in accepted:
            results.append(await self._process_item(item, decision))
        return results

    # -----------------------------------------------------------------
    # Tier 2 onward, per item — reuses the existing per-job machinery
    # -----------------------------------------------------------------
    async def _process_item(self, item, timing) -> JobResult:
        brand = self._brand
        job = await self._jobs.create(brand_id=brand.brand_id, trigger="sweep",
                                      payload={"item_id": item.item_id,
                                               "source_url": str(item.source_url)})
        jid = job.id
        spent_before = self._cost.spent_today()

        def result(status, reason=None, **artifacts):
            return JobResult(job_id=jid, status=status, reason=reason, artifacts=artifacts,
                             cost_usd=round(self._cost.spent_today() - spent_before, 6))

        # budget gate — halt before any spend
        if not self._cost.can_spend(0.0, critical=False):
            await self._jobs.update(jid, status="halted_budget")
            return result("halted_budget", reason="daily budget cap reached")

        try:
            await self._jobs.update(jid, status="running", current_tier=2)
            memory = await self._a.memory.query(brand_id=brand.brand_id, text=item.post_angle, k=5)

            quality_restart_budget = 1   # Quality 'reject' -> fresh Strategy attempt, bounded
            copy_secs = 0.0
            while True:
                # --- Strategy ------------------------------------------------
                strategy_out = await self._a.strategy.decide_v2(
                    item=item, timing=timing, memory=memory, job_id=jid)

                if strategy_out.narrative_conflict_flag:
                    return await self._halt_narrative_conflict(jid, item, strategy_out, result)

                # --- Skeptic (max 1 'revise' retry; 'reject' halts, no retry) --
                skeptic_out = await self._a.skeptic.review_v2(strategy=strategy_out, job_id=jid)
                if skeptic_out.verdict == SkepticVerdict.revise:
                    strategy_out = await self._a.strategy.decide_v2(
                        item=item, timing=timing, memory=memory, job_id=jid,
                        skeptic_critique=skeptic_out.critique)
                    if strategy_out.narrative_conflict_flag:   # re-check after the revised pass too
                        return await self._halt_narrative_conflict(jid, item, strategy_out, result)
                    skeptic_out = await self._a.skeptic.review_v2(strategy=strategy_out, job_id=jid)

                if skeptic_out.verdict != SkepticVerdict.approved:
                    await self._jobs.update(jid, status="skeptic_rejected")
                    if self._dl is not None:
                        await self._dl.record(
                            job_id=jid, step="skeptic",
                            error=f"verdict={skeptic_out.verdict.value}: {skeptic_out.critique}",
                            context={"item_id": item.item_id})
                    return result("skeptic_rejected", reason=skeptic_out.critique,
                                 item=item, strategy=strategy_out, skeptic=skeptic_out)

                # --- Copy + Quality (max 1 Copy-only retry on 'revise') -------
                await self._jobs.update(jid, current_tier=3)
                t0 = perf_counter()
                copy_out = await self._a.copy.write_v2(strategy=strategy_out, timing=timing, job_id=jid)
                copy_secs = round(perf_counter() - t0, 2)

                await self._jobs.update(jid, current_tier=4)
                copy_retry_left = 1
                while True:
                    quality_out = await self._a.quality.evaluate_v2(copy=copy_out, job_id=jid)
                    if quality_out.verdict == QualityVerdict.pass_:
                        break
                    if quality_out.verdict == QualityVerdict.revise and copy_retry_left > 0:
                        copy_retry_left -= 1
                        t0 = perf_counter()
                        copy_out = await self._a.copy.write_v2(
                            strategy=strategy_out, timing=timing, job_id=jid,
                            quality_notes=quality_out.notes)
                        copy_secs = round(perf_counter() - t0, 2)
                        continue
                    break   # 'reject', or the one revise-retry is exhausted

                if quality_out.verdict == QualityVerdict.pass_:
                    break   # good post -> fall through to Distribution below

                if quality_out.verdict == QualityVerdict.revise:
                    await self._jobs.update(jid, status="quality_revise_exhausted")
                    return result("quality_revise_exhausted", reason="; ".join(quality_out.notes),
                                 item=item, strategy=strategy_out, copy=copy_out, quality=quality_out)

                # verdict == 'reject' -> fresh Strategy attempt, bounded at 1
                if quality_restart_budget <= 0:
                    await self._jobs.update(jid, status="quality_rejected")
                    return result("quality_rejected", reason="; ".join(quality_out.notes),
                                 item=item, strategy=strategy_out, copy=copy_out, quality=quality_out)
                quality_restart_budget -= 1
                # loop back to the top: Strategy runs again from scratch

            # --- Distribution: always stages, always requires human approval ---
            await self._jobs.update(jid, current_tier=5)
            staged = await self._build_staged_post(item, timing, copy_out, quality_out)
            bundle = await self._a.distribution.stage_v2(staged=staged, job_id=jid)

            # Recorded at STAGE time, not publish time — so this source_url
            # can't re-enter tomorrow's sweep while today's staged item is
            # still sitting unreviewed in the queue (see post_history_store.py).
            if self._post_history is not None:
                from datetime import datetime, timezone
                embedding = None
                if self._embeddings is not None:
                    embedding = (await safe_embed(self._embeddings, [item.post_angle]))[0]
                await self._post_history.record(
                    source_url=str(item.source_url), item_id=item.item_id,
                    posted_at=datetime.now(timezone.utc), slot=timing.recommended_slot.value,
                    post_angle_embedding=embedding)

            await self._jobs.update(jid, status="staged_for_review")
            review_item = self._build_review_item(
                jid, item, memory, strategy_out, skeptic_out, timing, copy_out,
                quality_out, bundle, copy_secs)
            if self._reviews is not None:
                await self._reviews.add(item=review_item, bundle=bundle)
            if self._notifier is not None:
                try:
                    await self._notifier.notify_staged(
                        job_id=jid, chosen_angle=strategy_out.chosen_angle,
                        quality_overall=review_item.quality_overall,
                        requires_hedging=timing.citation_hedge_required, trigger="sweep")
                except Exception:
                    pass   # best-effort — must never affect an already-successful job

            return result("staged_for_review", item=item, strategy=strategy_out, copy=copy_out,
                         quality=quality_out, staged=staged, bundle=bundle, review_item=review_item)

        except HandoffHalted as exc:
            await self._jobs.update(jid, status="failed")
            return result("failed", reason=f"handoff halted at {exc.step}: {exc.reason}")
        except Exception as exc:   # defensive: a tool/agent crash must not escape the job
            await self._jobs.update(jid, status="failed")
            if self._dl is not None:
                await self._dl.record(job_id=jid, step="orchestrator",
                                      error=f"{type(exc).__name__}: {exc}", context={})
            return result("failed", reason=f"unexpected error: {type(exc).__name__}: {exc}")

    async def _halt_narrative_conflict(self, jid, item, strategy_out, result) -> JobResult:
        await self._jobs.update(jid, status="narrative_conflict")
        if self._nc is not None:
            await self._nc.record(
                job_id=jid, item_id=item.item_id, chosen_angle=strategy_out.chosen_angle,
                conflict_note=strategy_out.narrative_conflict_note, source_url=str(item.source_url))
        return result("narrative_conflict", reason=strategy_out.narrative_conflict_note,
                     item=item, strategy=strategy_out)

    async def _build_staged_post(self, item, timing, copy_out, quality_out):
        from engine.core.models import StagedPost
        return StagedPost(
            main_post=copy_out.main_post, reply_link=copy_out.reply_link,
            thread_tweets=copy_out.thread_tweets, format_used=copy_out.format_used,
            scheduled_slot=timing.recommended_slot, source_url=item.source_url,
            novelty_score=item.novelty_score, quality_scores=quality_out.scores,
            approval_note=quality_out.approval_note)

    # -----------------------------------------------------------------
    # Review-dashboard adapter (Phase 2 stopgap)
    #
    # ReviewItem/AgentAttribution are NOT rebuilt in this phase — the spec's
    # own tracker lists the review dashboard as Phase 4 ("4a. Human approval
    # gate in Distribution"). This maps the new fields into the existing
    # open-ended dict/list fields so the dashboard keeps working without a
    # schema change here; several old fields have no real Phase-2 equivalent
    # (pillar_id, auto_eligible, research_alternatives, timing velocity/gaps)
    # and are left blank/False rather than mapped to something misleading.
    # -----------------------------------------------------------------
    def _build_review_item(self, jid, item, memory, strategy_out, skeptic_out, timing,
                           copy_out, quality_out, bundle, copy_secs) -> ReviewItem:
        mem_ctx = [m.content for m in (*memory.episodic, *memory.semantic)][:4]
        scores = quality_out.scores.model_dump()
        attribution = AgentAttribution(
            research_chosen=item.headline,
            research_alternatives=[],           # no per-item "alternatives" in a sweep — see note above
            timing_velocity=timing.recommended_slot.value,   # closest analog; not the old velocity verdict
            timing_velocity_confidence=0.0,
            timing_gaps=[],
            timing_citation="hedge required" if timing.citation_hedge_required else "not required",
            memory_context=mem_ctx,
            copy_output_seconds=copy_secs,
            quality={**{k: float(v) for k, v in scores.items()}, "total": float(quality_out.total)},
            skeptic_summary=skeptic_out.critique,
        )
        return ReviewItem(
            job_id=jid, brand_id=self._brand.brand_id, status="staged_for_review",
            pillar_id="",                        # no pillar concept in the Phase 2 schema
            requires_hedging=timing.citation_hedge_required,
            chosen_angle=strategy_out.chosen_angle,
            x_thread=copy_out.thread_tweets or [copy_out.main_post],   # stopgap: no ig/yt anymore
            ig_caption="", youtube_script="",
            quality_overall=round(quality_out.total / 50.0, 4),
            quality_route=quality_out.verdict.value,
            auto_eligible=False,                 # always requires human approval now — see StagedPost
            platforms=bundle.plan.platforms,
            attribution=attribution,
        )
