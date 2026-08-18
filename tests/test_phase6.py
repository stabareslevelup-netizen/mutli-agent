"""
Phase 6 integration tests — FREE (fake agents, no LLM/network calls except
where a specific test needs the REAL StrategyAgent to prove sink-write
routing — see test_narrative_conflict_hard_halt_and_sink_routing).

Covers the Orchestrator end-to-end: run_sweep() -> _process_item() per
accepted item, the bounded retry loops (each capped at 1 retry, verified
against orchestrator.py directly, not assumed), the narrative-conflict
hard-halt and its sink-write routing (relocated into strategy.py per the
fix in the previous step), StagedPost construction, and PostHistoryStore
recording at stage time.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date

from engine.agents.base import AgentContext
from engine.agents.distribution import StagedBundle
from engine.agents.orchestrator import Orchestrator, OrchestratorAgents
from engine.agents.strategy import StrategyAgent
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.job_manager import InMemoryJobStore
from engine.core.llm import Usage
from engine.core.models import (
    CopyOutput, DistributionPlan, MemoryQueryResult, PostFormat, PostingMode,
    PostingSlot, QualityDimensions, QualityOutput, ResearchItem, SkepticOutput,
    SkepticVerdict, SourceKind, StagedPost, StrategyOutput, TimingDecision,
)
from engine.core.narrative_conflict_sink import InMemoryNarrativeConflictSink
from engine.core.validation_gate import ValidationGate

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


# ===========================================================================
# Fixtures
# ===========================================================================
def _item(item_id, url, novelty=8):
    return ResearchItem(item_id=item_id, source=SourceKind.sam_gov, date_found=date.today(),
                        headline="Epirus wins contract", raw_detail="detail", novelty_score=novelty,
                        post_angle="angle", source_url=url)


def _decision(item_id, slot=PostingSlot.slot_3pm, hedge=False, reason=None):
    return TimingDecision(item_id=item_id, recommended_slot=slot, rejection_reason=reason,
                          urgency_note="note", citation_hedge_required=hedge)


def _strategy(angle="an angle", conflict=False, note=None):
    return StrategyOutput(chosen_angle=angle, format=PostFormat.pov_post, must_include=["fact"],
                          narrative_conflict_flag=conflict, narrative_conflict_note=note)


def _skeptic(verdict=SkepticVerdict.approved, critique="fine",
            revised_angle=None, revised_must=None):
    return SkepticOutput(verdict=verdict, critique=critique,
                         revised_angle=revised_angle, revised_must_include=revised_must)


def _copy(main_post="post text", hedge=False):
    return CopyOutput(main_post=main_post, reply_link="Source: x",
                      format_used=PostFormat.pov_post, citation_hedged=hedge)


def _quality(lowest=8):
    scores = QualityDimensions(source_specificity=8, differentiation=8, hook_strength=lowest,
                               format_compliance=8, thesis_alignment=8)
    return QualityOutput(scores=scores, approval_note="ok")


def _cost_guard():
    return CostGuard(daily_budget_usd=25.0, cost_sink=InMemoryCostSink())


# ===========================================================================
# Fake agents — track call counts, return canned outputs in order
# ===========================================================================
class FakeResearch:
    def __init__(self, items):
        self._items = items
        self.calls = 0

    async def sweep(self, *, job_id):
        self.calls += 1
        return self._items


class FakeTiming:
    def __init__(self, decisions):
        self._decisions = decisions
        self.calls = 0

    async def assign(self, *, items, job_id):
        self.calls += 1
        return self._decisions


class FakeMemory:
    async def query(self, *, brand_id, text, k=5):
        return MemoryQueryResult(episodic=[], semantic=[], narrative=[])


class FakeStrategy:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    async def decide_v2(self, *, item, timing, memory, job_id, skeptic_critique=None):
        self.calls += 1
        return self._outputs.pop(0)


class FakeSkeptic:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    async def review_v2(self, *, item, strategy, job_id):
        self.calls += 1
        return self._outputs.pop(0)


class FakeCopy:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    async def write_v2(self, *, strategy, timing, job_id, quality_notes=None):
        self.calls += 1
        return self._outputs.pop(0)


class FakeQuality:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    async def evaluate_v2(self, *, copy, job_id):
        self.calls += 1
        return self._outputs.pop(0)


class FakeDistribution:
    def __init__(self):
        self.calls = 0
        self.staged_records = []

    async def stage_v2(self, *, staged, job_id):
        self.calls += 1
        self.staged_records.append(staged)
        plan = DistributionPlan(posting_mode=PostingMode.confirm, platforms=["x"],
                                staged=True, published=False)
        return StagedBundle(plan=plan, staged={})


class FakePostHistory:
    def __init__(self):
        self.records = []

    async def was_posted_recently(self, *, source_url, within_hours=72):
        return False

    async def topic_posted_recently(self, *, post_angle_embedding, within_hours=48):
        return False

    async def record(self, **kwargs):
        self.records.append(kwargs)


def _agents(research, timing, strategy, skeptic, copy, quality, distribution):
    return OrchestratorAgents(research=research, memory=FakeMemory(), timing=timing,
                              strategy=strategy, skeptic=skeptic, copy=copy,
                              quality=quality, distribution=distribution)


# ===========================================================================
# 1. run_sweep() -> _process_item() end-to-end
# ===========================================================================
async def test_sweep_to_process_item_end_to_end():
    accepted_item = _item("acc1", "https://sam.gov/1")
    rejected_item = _item("rej1", "https://sam.gov/2")
    decisions = [_decision("acc1"), _decision("rej1", slot=PostingSlot.reject, reason="dup")]

    strat, skep, cop, qual, dist = (FakeStrategy([_strategy()]), FakeSkeptic([_skeptic()]),
                                     FakeCopy([_copy()]), FakeQuality([_quality()]), FakeDistribution())
    agents = _agents(FakeResearch([accepted_item, rejected_item]), FakeTiming(decisions),
                     strat, skep, cop, qual, dist)
    job_store = InMemoryJobStore()
    orch = Orchestrator(agents=agents, job_store=job_store, cost_guard=_cost_guard(), brand=load_brand())

    out = await orch.run_sweep()
    check("only the accepted item gets processed (rejected one skipped entirely)",
          len(out) == 1, str(out))
    check("accepted item flows all the way through to staged_for_review",
          out[0].status == "staged_for_review", out[0].status)
    check("each accepted item gets its own job_id", out[0].job_id)
    job = await job_store.get(out[0].job_id)
    check("job actually persisted in the job store", job is not None and job.status == "staged_for_review")
    check("full chain called exactly once each on the happy path",
          strat.calls == 1 and skep.calls == 1 and cop.calls == 1 and qual.calls == 1 and dist.calls == 1)


# ===========================================================================
# 2a. Skeptic revise -> Strategy exactly 1 more pass, then final
# ===========================================================================
async def test_skeptic_revise_one_retry_then_final():
    item = _item("i1", "https://sam.gov/1")
    decision = _decision("i1")
    strat = FakeStrategy([_strategy(angle="first attempt"), _strategy(angle="revised attempt")])
    skep = FakeSkeptic([_skeptic(verdict=SkepticVerdict.revise, critique="fix this",
                                 revised_angle="x", revised_must=["y"]),
                        _skeptic(verdict=SkepticVerdict.approved)])
    cop, qual, dist = FakeCopy([_copy()]), FakeQuality([_quality()]), FakeDistribution()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(), brand=load_brand())

    out = await orch.run_sweep()
    check("skeptic revise -> exactly 2 Strategy calls (initial + 1 retry, no more)", strat.calls == 2)
    check("skeptic called exactly twice (initial + re-review after retry)", skep.calls == 2)
    check("job proceeds to staged_for_review after the approved re-review", out[0].status == "staged_for_review")


# ===========================================================================
# 2b. Skeptic reject -> hard-halt immediately, no retry
# ===========================================================================
async def test_skeptic_reject_hard_halts_no_retry():
    item = _item("i1", "https://sam.gov/1")
    decision = _decision("i1")
    strat = FakeStrategy([_strategy()])
    # only ONE skeptic output queued -- if the orchestrator tried a retry it
    # would call review_v2 a second time and .pop(0) on an empty list would
    # raise IndexError, surfacing as an unexpected "failed" status below
    skep = FakeSkeptic([_skeptic(verdict=SkepticVerdict.reject, critique="fabricated claim")])
    cop, qual, dist = FakeCopy([]), FakeQuality([]), FakeDistribution()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    dl = InMemoryDeadLetterSink()
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(),
                        brand=load_brand(), dead_letter_sink=dl)

    out = await orch.run_sweep()
    check("skeptic reject -> status skeptic_rejected (not failed from a phantom retry)",
          out[0].status == "skeptic_rejected", out[0].status)
    check("strategy called exactly once -- no retry on reject", strat.calls == 1)
    check("skeptic called exactly once -- no retry on reject", skep.calls == 1)
    check("copy/quality never reached", cop.calls == 0 and qual.calls == 0)
    check("skeptic rejection dead-lettered", len(dl.records) == 1)


# ===========================================================================
# 2c. Quality revise -> Copy retries alone (not Strategy), exactly once
# ===========================================================================
async def test_quality_revise_copy_only_retry():
    item = _item("i1", "https://sam.gov/1")
    decision = _decision("i1")
    strat = FakeStrategy([_strategy()])
    skep = FakeSkeptic([_skeptic()])
    cop = FakeCopy([_copy(main_post="first draft"), _copy(main_post="revised draft")])
    qual = FakeQuality([_quality(lowest=6), _quality(lowest=8)])   # 6 -> revise, 8 -> pass
    dist = FakeDistribution()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(), brand=load_brand())

    out = await orch.run_sweep()
    check("quality revise -> Strategy NOT re-called (stays at 1)", strat.calls == 1)
    check("quality revise -> Skeptic NOT re-called (stays at 1)", skep.calls == 1)
    check("quality revise -> Copy retried exactly once (2 total calls)", cop.calls == 2)
    check("quality re-evaluated exactly once after the retry (2 total calls)", qual.calls == 2)
    check("job proceeds to staged_for_review with the revised draft",
          out[0].status == "staged_for_review" and dist.staged_records[0].main_post == "revised draft")


# ===========================================================================
# 2d. Quality reject -> fresh Strategy attempt (confirmed against actual wiring)
# ===========================================================================
async def test_quality_reject_triggers_fresh_strategy_attempt():
    item = _item("i1", "https://sam.gov/1")
    decision = _decision("i1")
    strat = FakeStrategy([_strategy(angle="first strategy"), _strategy(angle="second strategy")])
    skep = FakeSkeptic([_skeptic(), _skeptic()])
    cop = FakeCopy([_copy(), _copy()])
    qual = FakeQuality([_quality(lowest=2), _quality(lowest=8)])   # 2 -> reject, 8 -> pass
    dist = FakeDistribution()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(), brand=load_brand())

    out = await orch.run_sweep()
    check("quality reject -> Strategy re-runs from scratch (2 total calls)", strat.calls == 2)
    check("quality reject -> Skeptic re-runs too (fresh pass through the top, 2 total calls)", skep.calls == 2)
    check("quality reject -> Copy re-runs too (fresh pass, 2 total calls)", cop.calls == 2)
    check("job eventually proceeds to staged_for_review after the fresh attempt passes",
          out[0].status == "staged_for_review")


async def test_quality_reject_retry_exhausted_bounded_no_infinite_loop():
    item = _item("i1", "https://sam.gov/1")
    decision = _decision("i1")
    strat = FakeStrategy([_strategy(), _strategy()])   # exactly 2 -- proves it's bounded, not infinite
    skep = FakeSkeptic([_skeptic(), _skeptic()])
    cop = FakeCopy([_copy(), _copy()])
    qual = FakeQuality([_quality(lowest=2), _quality(lowest=2)])   # reject both times
    dist = FakeDistribution()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(), brand=load_brand())

    out = await orch.run_sweep()
    check("quality reject exhausted -> terminal quality_rejected status",
          out[0].status == "quality_rejected", out[0].status)
    check("bounded at exactly 1 restart -- Strategy called exactly twice, never a 3rd time", strat.calls == 2)
    check("distribution never reached", dist.calls == 0)


# ===========================================================================
# 3. Narrative-conflict hard-halt + sink-write routing (real StrategyAgent)
# ===========================================================================
class CapturingLLM:
    def __init__(self, resp):
        self.resp = resp

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, **kw):
        return self.resp, Usage(input_tokens=10, output_tokens=10)


async def test_narrative_conflict_hard_halt_and_sink_routing():
    class AlwaysViolates:
        def violates(self, angle_text, constraint):
            return True

    from engine.core.models import NarrativeConstraint
    ctx = AgentContext(llm=CapturingLLM('{"chosen_angle":"any angle","format":"pov_post","must_include":["fact"]}'),
                       cost_guard=_cost_guard(), gate=ValidationGate(InMemoryDeadLetterSink()),
                       brand_id="madre_de_maquinas")
    nc_sink = InMemoryNarrativeConflictSink()
    real_strategy = StrategyAgent(ctx, checker=AlwaysViolates(), narrative_conflict_sink=nc_sink)

    class FakeMemoryWithConstraint:
        async def query(self, *, brand_id, text, k=5):
            return MemoryQueryResult(narrative=[NarrativeConstraint(position="p", stance="s")])

    item = _item("nc1", "https://sam.gov/nc1")
    decision = _decision("nc1")
    skep, cop, qual, dist, ph = FakeSkeptic([]), FakeCopy([]), FakeQuality([]), FakeDistribution(), FakePostHistory()
    agents = OrchestratorAgents(research=FakeResearch([item]), memory=FakeMemoryWithConstraint(),
                                timing=FakeTiming([decision]), strategy=real_strategy, skeptic=skep,
                                copy=cop, quality=qual, distribution=dist)
    dl = InMemoryDeadLetterSink()
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(),
                        brand=load_brand(), dead_letter_sink=dl, post_history_store=ph)

    out = await orch.run_sweep()
    check("narrative conflict -> job halts with status narrative_conflict",
          out[0].status == "narrative_conflict", out[0].status)
    check("skeptic/copy/quality/distribution never reached (hard-halt, no retry)",
          skep.calls == 0 and cop.calls == 0 and qual.calls == 0 and dist.calls == 0)
    check("sink write happened INSIDE strategy.py's decide_v2(), not orchestrator.py",
          len(nc_sink.records) == 1, str(nc_sink.records))
    check("conflict routed to the SEPARATE conflict queue, not the normal dead-letter path",
          len(dl.records) == 0, str(dl.records))
    check("PostHistoryStore never recorded -- item never reached staging", len(ph.records) == 0)


# ===========================================================================
# 4. StagedPost construction: requires_human_approval can't even be False
# ===========================================================================
async def test_staged_post_construction_and_human_approval_invariant():
    item = _item("sp1", "https://sam.gov/sp1")
    decision = _decision("sp1")
    copy_out = _copy(main_post="the actual post text")
    quality_out = _quality()

    orch = Orchestrator(agents=_agents(FakeResearch([]), FakeTiming([]), FakeStrategy([]),
                                       FakeSkeptic([]), FakeCopy([]), FakeQuality([]), FakeDistribution()),
                        job_store=InMemoryJobStore(), cost_guard=_cost_guard(), brand=load_brand())
    staged = await orch._build_staged_post(item, decision, copy_out, quality_out)
    check("_build_staged_post assembles a real StagedPost", isinstance(staged, StagedPost))
    check("main_post carried through from copy_out", staged.main_post == "the actual post text")
    check("source_url carried through from item", str(staged.source_url) == str(item.source_url))
    check("requires_human_approval is True by construction", staged.requires_human_approval is True)

    # can't even construct one with it False -- Literal[True], not just a default
    rejected = False
    try:
        StagedPost(main_post="x", reply_link="Source: y", format_used=PostFormat.pov_post,
                  scheduled_slot=PostingSlot.slot_3pm, source_url="https://sam.gov/1",
                  novelty_score=8, quality_scores=quality_out.scores, approval_note="a",
                  requires_human_approval=False)
    except Exception:
        rejected = True
    check("StagedPost(requires_human_approval=False) is structurally unconstructable", rejected)


# ===========================================================================
# 5. PostHistoryStore recorded at STAGE time, not publish time
# ===========================================================================
async def test_post_history_recorded_at_stage_time():
    item = _item("ph1", "https://sam.gov/ph1")
    decision = _decision("ph1")
    strat, skep, cop, qual, dist = (FakeStrategy([_strategy()]), FakeSkeptic([_skeptic()]),
                                     FakeCopy([_copy()]), FakeQuality([_quality()]), FakeDistribution())
    ph = FakePostHistory()
    agents = _agents(FakeResearch([item]), FakeTiming([decision]), strat, skep, cop, qual, dist)
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=_cost_guard(),
                        brand=load_brand(), post_history_store=ph)

    out = await orch.run_sweep()
    check("job reached staged_for_review", out[0].status == "staged_for_review")
    check("exactly one PostHistoryStore.record() call happened (at stage time)",
          len(ph.records) == 1, str(ph.records))
    check("recorded source_url matches the item that got staged",
          ph.records[0]["source_url"] == str(item.source_url))
    check("recorded item_id matches", ph.records[0]["item_id"] == "ph1")


async def main() -> int:
    await test_sweep_to_process_item_end_to_end()
    await test_skeptic_revise_one_retry_then_final()
    await test_skeptic_reject_hard_halts_no_retry()
    await test_quality_revise_copy_only_retry()
    await test_quality_reject_triggers_fresh_strategy_attempt()
    await test_quality_reject_retry_exhausted_bounded_no_infinite_loop()
    await test_narrative_conflict_hard_halt_and_sink_routing()
    await test_staged_post_construction_and_human_approval_invariant()
    await test_post_history_recorded_at_stage_time()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_phase6():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_phase6`) is
    unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "phase 6 suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
