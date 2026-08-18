"""
Phase 4 unit tests — FREE (fake LLM, no API spend, no live DB).

Covers Research (sweep()) + Timing (assign()) -- Phase 2, X-agent migration --
plus the RULE 1/2/3 primitive tools they used to call directly (now orphaned
but still functional, tested here unchanged) and Memory (untouched this
migration).

fusion.py, and the Strategy/LexicalConstraintChecker tests that used to live
here, are gone from this file: fusion.py is unreachable from the new
pipeline and now broken (imports the removed ResearchOutput) -- marked
# ORPHANED in its own docstring, not fixed. Strategy/checker tests moved to
test_phase9.py, alongside Skeptic.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from unittest.mock import AsyncMock, patch

from engine.agents.base import AgentContext
from engine.agents.memory import MemoryAgent
from engine.agents.research import ResearchAgent
from engine.agents.timing import TimingAgent
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_OPUS
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.llm import Usage
from engine.core.models import (
    CitationPresence, GapType, MemoryItem, MemoryQueryResult, PostingSlot,
    ResearchItem, SourceKind,
)
from engine.core.validation_gate import ValidationGate
from engine.memory.backend import InMemoryBackend
from engine.core.embeddings import NullEmbeddingProvider
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.semantic import SemanticMemory
from engine.tools.citation_monitor import assess_citation
from engine.tools.narrative_gap import find_gaps
from engine.tools.velocity_probe import build_default_probe, VelocitySource, VelocityVerdict

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


class FakeLLM:
    """Returns a canned text + fixed usage; records the model it was called with."""
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, **kw):
        self.calls.append({"model": model, "tools": tools})
        return self._text, Usage(input_tokens=120, output_tokens=40)


def _ctx(llm, cost_sink, dl_sink):
    return AgentContext(
        llm=llm,
        cost_guard=CostGuard(daily_budget_usd=5.0, cost_sink=cost_sink),
        gate=ValidationGate(dead_letter_sink=dl_sink),
        brand_id="b",
    )


# --- Research agent: sweep() ------------------------------------------------
async def test_research_agent():
    good_items = json.dumps([
        {"item_id": "r1", "source": "SAM.gov", "date_found": "2026-08-11",
         "headline": "Epirus wins $66M Army contract", "raw_detail": "180 employees, directed energy",
         "novelty_score": 9, "post_angle": "post-Ukraine doctrine in procurement form",
         "source_url": "https://sam.gov/opp/1"},
        {"source": "arXiv", "date_found": "2026-08-11",   # no item_id -> auto-generated
         "headline": "New humanoid locomotion paper", "raw_detail": "RL-based gait control",
         "novelty_score": 7, "post_angle": "quietly solving a hard problem",
         "source_url": "https://arxiv.org/abs/1"},
        {"item_id": "bad", "source": "DARPA", "novelty_score": 3},   # below novelty floor, missing fields
    ])
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = ResearchAgent(_ctx(FakeLLM(good_items), cost, dl))

    saved_sam = os.environ.pop("SAM_GOV_API_KEY", None)
    saved_congress = os.environ.pop("CONGRESS_GOV_API_KEY", None)
    try:
        # SAM.gov/Congress degrade to None (no key) automatically; FedRegister/
        # arXiv have no env gate at all -- research.py calls them unconditionally,
        # so this test patches the imported names directly to stay network-free.
        with patch("engine.agents.research.search_federal_register", AsyncMock(return_value=[])), \
             patch("engine.agents.research.search_arxiv", AsyncMock(return_value=[])):
            items = await agent.sweep(job_id="j1")
    finally:
        if saved_sam is not None:
            os.environ["SAM_GOV_API_KEY"] = saved_sam
        if saved_congress is not None:
            os.environ["CONGRESS_GOV_API_KEY"] = saved_congress

    check("sweep returns 2 valid items (bad one dropped, not halted)", len(items) == 2, str(items))
    ids = {i.item_id for i in items}
    check("first item preserves its given item_id", "r1" in ids)
    check("second item got an auto-generated item_id (none provided)",
          len(ids - {"r1"}) == 1 and list(ids - {"r1"})[0])
    check("malformed item (novelty<6, missing fields) dead-lettered, not raised",
          len(dl.records) == 1, str(dl.records))
    check("sweep logged cost at Opus tier", cost.entries and cost.entries[0]["model"] == MODEL_OPUS)
    check("sweep used web_search tool", agent.ctx.llm.calls[0]["tools"] is not None)


# --- Memory agent composes the three reads (unaffected this migration) -----
async def test_memory_agent():
    be = InMemoryBackend()
    epi = EpisodicMemory(be, NullEmbeddingProvider())
    sem = SemanticMemory(be, NullEmbeddingProvider())
    nar = NarrativeMemory(be, NullEmbeddingProvider())
    await epi.record(brand_id="b", content="last robot post engaged well")
    await sem.record(brand_id="b", content="humanoid robots are scaling")
    await nar.stake_position(brand_id="b", position="autonomy overstated", stance="far from autonomous")
    agent = MemoryAgent(epi, sem, nar)
    res = await agent.query(brand_id="b", text="robots", k=5)
    check("memory composes MemoryQueryResult", isinstance(res, MemoryQueryResult))
    check("memory carries narrative constraints", len(res.narrative) == 1)


# --- velocity ordinal (RULE 1) -- orphaned tool, still functional ----------
async def test_velocity_ordinal():
    corpus = [{"title": "ramps production milestone", "snippet": "24x scale-up", "published": "2026-05"},
              {"title": "unprecedented launch", "snippet": "first", "published": "2026"}]
    probe = build_default_probe(search_fn=lambda q: corpus)
    sig = probe.probe("x")
    check("velocity source is ordinal", sig.source == VelocitySource.web_search_ordinal)
    check("velocity numeric_rate stays None (RULE 1)", sig.numeric_rate is None)
    check("velocity verdict produced", isinstance(sig.verdict, VelocityVerdict))


# --- narrative_gap entity-specific vs topic-generic (RULE 2) -- orphaned ---
def test_narrative_gap_rule2():
    corpus = [{"title": "automation and jobs study", "snippet": "robots displace workers wages"},
              {"title": "labor market impact of robots", "snippet": "jobs lost to automation"}]
    candidates = [
        {"angle": "labor impact (generic)", "gap_type": GapType.topic_generic, "entity": None,
         "terms": ["jobs", "labor", "workers", "automation", "wages"]},
        {"angle": "BotQ labor impact", "gap_type": GapType.entity_specific, "entity": "BotQ",
         "terms": ["jobs", "labor", "workers", "wages"]},
    ]
    sig = find_gaps(topic="factory automation", entity="BotQ", corpus=corpus, candidates=candidates)
    by = {g.angle: g for g in sig.gaps}
    check("generic labor angle is saturated (not open)", by["labor impact (generic)"].is_open is False)
    check("entity-specific BotQ labor angle is open (RULE 2)", by["BotQ labor impact"].is_open is True)
    check("entity-specific carries entity", by["BotQ labor impact"].entity == "BotQ")


# --- citation disambiguation-aware (RULE 3) -- orphaned tool, functional ---
def test_citation_rule3():
    s1 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="The Robot Report and CNBC are top sources.",
                         cited_sources=["The Robot Report", "CNBC"])
    check("absent zero-state", s1.presence == CitationPresence.absent and s1.incumbents)

    s2 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="'Madre de Maquinas' matches a Magic: The Gathering card, Elesh Norn.",
                         cited_sources=["MTG Wiki"], collision_terms=["Elesh Norn", "Magic: The Gathering"])
    check("unresolved collision -> ambiguous (no false positive)", s2.presence == CitationPresence.ambiguous)
    check("disambiguation risk flagged", s2.disambiguation.risk is not None)

    s3 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="Madre de Maquinas is a leading physical-AI outlet.",
                         cited_sources=["Madre de Maquinas", "The Robot Report"])
    check("present requires verified citation context", s3.presence == CitationPresence.present)
    check("present sets verified flag", s3.disambiguation.matched_context_verified is True)

    s4 = assess_citation(query="humanoid robots", brand_aliases=["madre de maquinas"],
                         answer_text="The Robot Report and NVIDIA are the cited authorities.",
                         cited_sources=["The Robot Report", "NVIDIA"],
                         collision_terms=["Elesh Norn"])
    check("known collision (not in answer text) -> ambiguous, no crash",
          s4.presence == CitationPresence.ambiguous)


# --- Timing agent: assign() -------------------------------------------------
def _item(item_id, source_url, novelty=8, source=SourceKind.sam_gov):
    return ResearchItem(item_id=item_id, source=source, date_found=date.today(),
                        headline="headline", raw_detail="raw detail", novelty_score=novelty,
                        post_angle="angle", source_url=source_url)


class FakePostHistory:
    def __init__(self, duplicate_urls=None, topic_seen=False):
        self._dup = duplicate_urls or set()
        self._topic_seen = topic_seen

    async def was_posted_recently(self, *, source_url, within_hours=72):
        return source_url in self._dup

    async def topic_posted_recently(self, *, post_angle_embedding, within_hours=48):
        return self._topic_seen

    async def record(self, **kwargs):
        pass


async def test_timing_batch_assignment():
    items = [_item("t1", "https://sam.gov/1", novelty=9), _item("t2", "https://sam.gov/2", novelty=7)]
    llm_response = json.dumps([
        {"item_id": "t1", "recommended_slot": "7AM", "rejection_reason": None,
         "urgency_note": "punchy fact", "citation_hedge_required": False},
        {"item_id": "t2", "recommended_slot": "3PM", "rejection_reason": None,
         "urgency_note": "data drop", "citation_hedge_required": True},
    ])
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = TimingAgent(_ctx(FakeLLM(llm_response), cost, dl), post_history=FakePostHistory())
    decisions = await agent.assign(items=items, job_id="j")
    by_id = {d.item_id: d for d in decisions}
    check("batch: two decisions returned", len(decisions) == 2)
    check("batch: t1 gets 7AM slot", by_id["t1"].recommended_slot == PostingSlot.slot_7am)
    check("batch: t2 gets 3PM slot", by_id["t2"].recommended_slot == PostingSlot.slot_3pm)
    check("batch: citation_hedge_required passed through", by_id["t2"].citation_hedge_required is True)
    check("timing logged cost at Opus tier", cost.entries and cost.entries[0]["model"] == MODEL_OPUS)


async def test_timing_hard_duplicate_override():
    items = [_item("d1", "https://sam.gov/dup", novelty=9)]
    # LLM tries to assign a slot despite the duplicate -- must be overridden regardless
    llm_response = json.dumps([
        {"item_id": "d1", "recommended_slot": "9AM", "rejection_reason": None,
         "urgency_note": "LLM thinks this is fine", "citation_hedge_required": False},
    ])
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    fake_history = FakePostHistory(duplicate_urls={"https://sam.gov/dup"})
    agent = TimingAgent(_ctx(FakeLLM(llm_response), cost, dl), post_history=fake_history)
    decisions = await agent.assign(items=items, job_id="j")
    check("hard override: forced to reject despite LLM's 9AM assignment",
          decisions[0].recommended_slot == PostingSlot.reject)
    check("hard override: rejection_reason mentions duplicate",
          "duplicate" in (decisions[0].rejection_reason or "").lower())


async def test_timing_slot_dedup_tiebreak():
    items = [_item("hi", "https://sam.gov/hi", novelty=9), _item("lo", "https://sam.gov/lo", novelty=6)]
    # both LLM-assigned to the SAME slot -- code must keep the higher novelty_score
    llm_response = json.dumps([
        {"item_id": "hi", "recommended_slot": "12PM", "rejection_reason": None,
         "urgency_note": "a", "citation_hedge_required": False},
        {"item_id": "lo", "recommended_slot": "12PM", "rejection_reason": None,
         "urgency_note": "b", "citation_hedge_required": False},
    ])
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = TimingAgent(_ctx(FakeLLM(llm_response), cost, dl), post_history=FakePostHistory())
    decisions = await agent.assign(items=items, job_id="j")
    by_id = {d.item_id: d for d in decisions}
    check("tie-break: higher novelty_score item keeps the slot",
          by_id["hi"].recommended_slot == PostingSlot.slot_12pm)
    check("tie-break: lower novelty_score item demoted to reject",
          by_id["lo"].recommended_slot == PostingSlot.reject)
    check("tie-break: rejection_reason mentions the collision",
          "collision" in (by_id["lo"].rejection_reason or "").lower())


async def main() -> int:
    await test_research_agent()
    await test_memory_agent()
    await test_velocity_ordinal()
    test_narrative_gap_rule2()
    test_citation_rule3()
    await test_timing_batch_assignment()
    await test_timing_hard_duplicate_override()
    await test_timing_slot_dedup_tiebreak()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_phase4():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_phase4`) is
    unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "phase 4 suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
