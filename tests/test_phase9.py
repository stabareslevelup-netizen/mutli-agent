"""
Phase 9 tests — FREE (fakes, no spend). Rebuilt for Phase 2 (X-agent
migration): Strategy (decide_v2()) + Skeptic (review_v2()), plus the
LexicalConstraintChecker unit tests moved here from test_phase4.py, plus
the feedback loop (unaffected this migration -- kept verbatim, only the
broken top-level imports it never actually needed were removed).

The old Fix-1 (Copy verification-gate injection) and Fix-2 (Skeptic
confidence_adjustment clamping) sections are gone -- both are superseded,
not silently dropped:
  - Fix 1's core guarantee (claims get hedged when required) is now fully
    covered elsewhere: TimingDecision.citation_hedge_required (tested in
    test_phase4.py's batch-assignment test) and CopyOutput.citation_hedged
    being code-forced from it (tested in test_phase5.py). Nothing left to
    re-test here.
  - Fix 2's confidence_adjustment field doesn't exist on SkepticOutput at
    all anymore -- Skeptic can now genuinely reject/request revision
    (verdict-based), a strictly stronger enforcement mechanism than a
    numeric nudge that never blocked anything.
"""
from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

from engine.agents.base import AgentContext
from engine.agents.skeptic import SkepticAgent
from engine.agents.strategy import (
    LexicalConstraintChecker, StrategyAgent,
)
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_SONNET, model_for
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.embeddings import NullEmbeddingProvider
from engine.core.feedback_service import FeedbackService, InMemoryFeedbackStore, engagement_score
from engine.core.llm import Usage
from engine.core.models import (
    AgentAttribution, MemoryQueryResult, NarrativeConstraint, PostFormat,
    PostingSlot, ReviewItem, ResearchItem, SkepticVerdict, SourceKind,
    StrategyOutput, TimingDecision,
)
from engine.core.narrative_conflict_sink import InMemoryNarrativeConflictSink
from engine.core.validation_gate import HandoffHalted, ValidationGate
from engine.memory.backend import InMemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.procedural import ProceduralMemory
from engine.memory.semantic import SemanticMemory

import os
os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


class CapturingLLM:
    def __init__(self, resp):
        self.resp = resp
        self.last_user = self.last_system = None
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, output_schema=None, **kw):
        self.last_system, self.last_user = system, user
        self.calls.append({"model": model, "user": user})
        return self.resp, Usage(input_tokens=10, output_tokens=10)


def _ctx(llm):
    return AgentContext(llm=llm, cost_guard=CostGuard(daily_budget_usd=25, cost_sink=InMemoryCostSink()),
                        gate=ValidationGate(InMemoryDeadLetterSink()), brand_id="madre_de_maquinas")


def _item():
    return ResearchItem(item_id="i1", source=SourceKind.sam_gov, date_found=date.today(),
                        headline="Epirus wins $66M Army contract", raw_detail="180 employees, directed energy",
                        novelty_score=9, post_angle="post-Ukraine doctrine in procurement form",
                        source_url="https://sam.gov/opp/1")


def _timing():
    return TimingDecision(item_id="i1", recommended_slot=PostingSlot.slot_3pm,
                          urgency_note="note", citation_hedge_required=False)


def _strategy_out():
    return StrategyOutput(chosen_angle="Epirus wins $66M Army contract",
                          format=PostFormat.pov_post, must_include=["$66M", "Epirus"])


# ===========================================================================
# Strategy agent: decide_v2()
# ===========================================================================
async def test_strategy_happy_path():
    resp = '{"chosen_angle":"a safe angle","format":"pov_post","must_include":["fact1"]}'
    llm = CapturingLLM(resp)
    memory = MemoryQueryResult(narrative=[])
    strat = StrategyAgent(_ctx(llm))
    out = await strat.decide_v2(item=_item(), timing=_timing(), memory=memory, job_id="j")
    check("decide_v2 returns a StrategyOutput", isinstance(out, StrategyOutput))
    check("no constraints -> no narrative conflict", out.narrative_conflict_flag is False)


async def test_strategy_narrative_conflict_code_computed():
    class FakeChecker:
        def violates(self, angle_text, constraint):
            return "BANNED" in angle_text

    resp = '{"chosen_angle":"a BANNED angle that contradicts staked view","format":"pov_post","must_include":["fact1"]}'
    llm = CapturingLLM(resp)
    memory = MemoryQueryResult(narrative=[NarrativeConstraint(position="p", stance="s")])
    nc_sink = InMemoryNarrativeConflictSink()
    strat = StrategyAgent(_ctx(llm), checker=FakeChecker(), narrative_conflict_sink=nc_sink)
    out = await strat.decide_v2(item=_item(), timing=_timing(), memory=memory, job_id="j")
    check("checker violation -> narrative_conflict_flag forced True (code-computed)",
          out.narrative_conflict_flag is True)
    check("conflict note populated", bool(out.narrative_conflict_note))
    check("sink write relocated into strategy.py -- record() was actually called",
          len(nc_sink.records) == 1, str(nc_sink.records))
    check("sink record carries the job_id/item_id/chosen_angle",
          nc_sink.records[0]["item_id"] == "i1" and nc_sink.records[0]["chosen_angle"] == out.chosen_angle)


async def test_strategy_llm_self_reported_conflict_not_reset_by_code():
    # ADVERSARIAL: the LLM self-reports narrative_conflict_flag=True with its
    # own made-up note, but the checker finds NO real violation. The design
    # intent ("code-computed, never LLM self-report") implies this should be
    # forced back to False -- confirming whether the current code actually
    # does that, or only overrides False->True and never resets True->False.
    resp = ('{"chosen_angle":"a perfectly fine angle","format":"pov_post","must_include":["fact1"],'
            '"narrative_conflict_flag":true,"narrative_conflict_note":"LLM made this up"}')
    llm = CapturingLLM(resp)
    memory = MemoryQueryResult(narrative=[NarrativeConstraint(position="p", stance="s")])

    class NeverViolates:
        def violates(self, angle_text, constraint):
            return False

    strat = StrategyAgent(_ctx(llm), checker=NeverViolates())
    out = await strat.decide_v2(item=_item(), timing=_timing(), memory=memory, job_id="j")
    check("checker finds no violation -> code should force flag back to False "
          "(self-report must not leak through unchecked)",
          out.narrative_conflict_flag is False, f"got {out.narrative_conflict_flag}")


async def test_strategy_skeptic_critique_threaded_into_retry():
    resp = '{"chosen_angle":"revised angle","format":"pov_post","must_include":["fact1"]}'
    llm = CapturingLLM(resp)
    memory = MemoryQueryResult(narrative=[])
    strat = StrategyAgent(_ctx(llm))
    await strat.decide_v2(item=_item(), timing=_timing(), memory=memory, job_id="j",
                          skeptic_critique="your angle overclaims the contract scope")
    check("skeptic_critique text reaches the actual prompt sent to the LLM",
          "your angle overclaims the contract scope" in llm.last_user)

    llm2 = CapturingLLM(resp)
    strat2 = StrategyAgent(_ctx(llm2))
    await strat2.decide_v2(item=_item(), timing=_timing(), memory=memory, job_id="j")
    check("no critique on first attempt -> prompt says so, not a stale critique",
          "first attempt" in llm2.last_user.lower())


def test_lexical_constraint_checker():
    c = LexicalConstraintChecker()
    cons = NarrativeConstraint(position="hype cycle exaggerated",
                               stance="we remain skeptical of the hype and overpromising")
    aligned = "hype cycle exaggerated: we remain skeptical of the hype overpromising"
    violating = "hype cycle exaggerated is wrong; the hype is fully justified and underhyped"
    check("lexical: aligned angle does not violate", c.violates(aligned, cons) is False)
    check("lexical: off-stance angle on same position violates", c.violates(violating, cons) is True)


# ===========================================================================
# Skeptic agent: review_v2()
# ===========================================================================
async def test_skeptic_approved():
    resp = '{"verdict":"approved","critique":"solid, well-sourced angle"}'
    llm = CapturingLLM(resp)
    rev = await SkepticAgent(_ctx(llm)).review_v2(item=_item(), strategy=_strategy_out(), job_id="j")
    check("skeptic runs on Sonnet tier", model_for("skeptic") == MODEL_SONNET)
    check("approved verdict returned", rev.verdict == SkepticVerdict.approved)
    check("critique populated", bool(rev.critique))


async def test_skeptic_revise_requires_revision_fields():
    good = ('{"verdict":"revise","critique":"buries the dollar amount",'
            '"revised_angle":"lead with the $66M figure","revised_must_include":["$66M"]}')
    rev = await SkepticAgent(_ctx(CapturingLLM(good))).review_v2(
        item=_item(), strategy=_strategy_out(), job_id="j")
    check("well-formed revise verdict passes", rev.verdict == SkepticVerdict.revise)
    check("revised_angle carried", rev.revised_angle == "lead with the $66M figure")

    malformed = '{"verdict":"revise","critique":"buries the dollar amount"}'   # missing revised_* fields
    halted = False
    try:
        await SkepticAgent(_ctx(CapturingLLM(malformed))).review_v2(
            item=_item(), strategy=_strategy_out(), job_id="j")
    except HandoffHalted:
        halted = True
    check("revise verdict WITHOUT revised_angle/revised_must_include -> HandoffHalted", halted)


async def test_skeptic_source_cross_check_reaches_prompt():
    llm = CapturingLLM('{"verdict":"approved","critique":"fine"}')
    item = _item()
    await SkepticAgent(_ctx(llm)).review_v2(item=item, strategy=_strategy_out(), job_id="j")
    check("source raw_detail reaches the actual prompt (needed for the overclaim check)",
          item.raw_detail in llm.last_user)
    check("the explicit line-by-line cross-check instruction is present",
          "line by line" in llm.last_user.lower())


# ===========================================================================
# Feedback loop (unaffected this migration -- kept verbatim)
# ===========================================================================
def _review_item(job_id, pillar, velocity, angle):
    return ReviewItem(job_id=job_id, brand_id="madre_de_maquinas", status="approved",
                      pillar_id=pillar, chosen_angle=angle,
                      attribution=AgentAttribution(timing_velocity=velocity))


class StubReviewStore:
    def __init__(self):
        self.items = {}

    def put(self, item):
        self.items[item.job_id] = SimpleNamespace(item=item, bundle=None)

    async def get(self, job_id):
        return self.items.get(job_id)


async def test_feedback_loop():
    brand = load_brand()
    be = InMemoryBackend()
    epi, sem = EpisodicMemory(be, NullEmbeddingProvider()), SemanticMemory(be, NullEmbeddingProvider())
    proc = ProceduralMemory(be, reference_set=[brand.voice], voice_threshold=0.2)
    rs = StubReviewStore()
    fb = FeedbackService(review_store=rs, episodic=epi, semantic=sem, procedural=proc,
                         brand_id=brand.brand_id, feedback_store=InMemoryFeedbackStore(), min_jobs=5)

    check("engagement scoring weights shares/saves over views",
          engagement_score({"x": {"shares": 1}}) > engagement_score({"x": {"views": 1}}))

    rs.put(_review_item("J0", "incident_file", "surging", "an incident"))
    r0 = await fb.ingest(job_id="J0", engagement={"x": {"views": 1000, "likes": 50, "shares": 10}})
    check("ingest returns engagement score", r0["engagement_score"] > 0)
    check("episodic auto-updated", r0["episodic_updated"] and len(await epi.query(brand_id=brand.brand_id, text="incident", k=9)) >= 1)
    check("semantic auto-updated", r0["semantic_updated"] and len(await sem.query(brand_id=brand.brand_id, text="performance", k=9)) >= 1)
    check("single ingest does not surface a proposal yet", r0["proposal_surfaced"] is None)

    proposal = None
    for i in range(1, 5):
        rs.put(_review_item(f"A{i}", "incident_file", "surging", "incident angle"))
        r = await fb.ingest(job_id=f"A{i}", engagement={"x": {"views": 5000, "likes": 400, "shares": 80, "saves": 60}})
        proposal = proposal or r["proposal_surfaced"]
    for i in range(1, 4):
        rs.put(_review_item(f"B{i}", "company_intel", "steady", "company angle"))
        r = await fb.ingest(job_id=f"B{i}", engagement={"x": {"views": 400, "likes": 5}})
        proposal = proposal or r["proposal_surfaced"]

    check("pattern across 5+ jobs surfaces a procedural proposal", proposal is not None)
    pending = await proc.pending_proposals()
    check("proposal is pending (NOT auto-applied)", len(pending) == 1 and pending[0].status == "pending")
    check("proposal targets strategy with performance data",
          pending[0].agent == "strategy" and "best" in pending[0].performance_data)
    check("proposal carries a voice-similarity check", pending[0].voice_similarity is not None)
    check("nothing auto-applied to procedural memory",
          await proc.active_prompt(brand_id=brand.brand_id, agent="strategy") is None)


async def main_async() -> int:
    await test_strategy_happy_path()
    await test_strategy_narrative_conflict_code_computed()
    await test_strategy_llm_self_reported_conflict_not_reset_by_code()
    await test_strategy_skeptic_critique_threaded_into_retry()
    test_lexical_constraint_checker()
    await test_skeptic_approved()
    await test_skeptic_revise_requires_revision_fields()
    await test_skeptic_source_cross_check_reaches_prompt()
    await test_feedback_loop()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_phase9():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_phase9`) is
    unaffected below."""
    import asyncio
    assert asyncio.run(main_async()) == 0, "phase 9 suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
