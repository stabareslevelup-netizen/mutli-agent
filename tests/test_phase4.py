"""
Phase 4 unit tests — FREE (fake LLM, no API spend, no live DB).

Proves the logic + the three carried rules + the routing requirements:
  - every agent call routes through cost_guard (correct tier) + validation_gate
  - velocity stays ordinal (RULE 1); narrative_gap entity vs generic (RULE 2);
    citation disambiguation-aware (RULE 3)
  - fusion weighting/ordering; Strategy hard narrative constraint
The single real-API pass lives in scripts/phase4_live_pass.py.
"""
from __future__ import annotations

import asyncio

from engine.agents.base import AgentContext
from engine.agents.memory import MemoryAgent
from engine.agents.research import ResearchAgent
from engine.agents.strategy import (
    LexicalConstraintChecker, StrategyAgent, StrategyBlocked,
)
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_OPUS
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.fusion import FusedAngle, fuse
from engine.core.llm import Usage, extract_json
from engine.core.models import (
    CitationPresence, GapType, MemoryItem, MemoryQueryResult, NarrativeConstraint,
    ResearchAngle, ResearchOutput, TimingSignal, VelocitySource, VelocityVerdict,
)
from engine.core.validation_gate import HandoffHalted, ValidationGate
from engine.memory.backend import InMemoryBackend
from engine.core.embeddings import NullEmbeddingProvider
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.semantic import SemanticMemory
from engine.tools.citation_monitor import assess_citation
from engine.tools.narrative_gap import find_gaps
from engine.tools.velocity_probe import build_default_probe

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


# --- Research agent: routes through cost_guard (Opus) + validates -----------
async def test_research_agent():
    good = ('{"angles":['
            '{"angle":"A","rationale":"r","sources":[{"url":"http://x","title":"t"}],"confidence":0.8},'
            '{"angle":"B","rationale":"r","sources":[{"url":"http://y","title":"t"}],"confidence":0.6},'
            '{"angle":"C","rationale":"r","sources":[{"url":"http://z","title":"t"}],"confidence":0.4}]}')
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = ResearchAgent(_ctx(FakeLLM(good), cost, dl))
    out = await agent.run(topic="physical AI", job_id="j1")
    check("research returns ResearchOutput", isinstance(out, ResearchOutput) and len(out.angles) == 3)
    check("research logged cost at Opus tier", cost.entries and cost.entries[0]["model"] == MODEL_OPUS)
    check("research used web_search tool", agent.ctx.llm.calls[0]["tools"] is not None)

    # malformed LLM output -> gate halts + dead-letters (nothing flows downstream)
    cost2, dl2 = InMemoryCostSink(), InMemoryDeadLetterSink()
    bad = ResearchAgent(_ctx(FakeLLM("sorry, I couldn't do that"), cost2, dl2))
    halted = False
    try:
        await bad.run(topic="x", job_id="j2")
    except HandoffHalted:
        halted = True
    check("malformed research output halts", halted)
    check("malformed output dead-lettered", len(dl2.records) == 1)
    check("cost still logged on malformed (call happened)", len(cost2.entries) == 1)


# --- Memory agent composes the three reads ----------------------------------
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


# --- velocity ordinal (RULE 1) ----------------------------------------------
async def test_velocity_ordinal():
    corpus = [{"title": "ramps production milestone", "snippet": "24x scale-up", "published": "2026-05"},
              {"title": "unprecedented launch", "snippet": "first", "published": "2026"}]
    probe = build_default_probe(search_fn=lambda q: corpus)
    sig = probe.probe("x")
    check("velocity source is ordinal", sig.source == VelocitySource.web_search_ordinal)
    check("velocity numeric_rate stays None (RULE 1)", sig.numeric_rate is None)
    check("velocity verdict produced", isinstance(sig.verdict, VelocityVerdict))


# --- narrative_gap entity-specific vs topic-generic (RULE 2) ----------------
def test_narrative_gap_rule2():
    # topic 'jobs' saturated generically, but never about the entity 'BotQ'
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


# --- citation disambiguation-aware (RULE 3) ---------------------------------
def test_citation_rule3():
    # zero-state: brand absent, no collision
    s1 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="The Robot Report and CNBC are top sources.",
                         cited_sources=["The Robot Report", "CNBC"])
    check("absent zero-state", s1.presence == CitationPresence.absent and s1.incumbents)

    # collision present, brand NOT in citations -> ambiguous, NOT present (no false positive)
    s2 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="'Madre de Maquinas' matches a Magic: The Gathering card, Elesh Norn.",
                         cited_sources=["MTG Wiki"], collision_terms=["Elesh Norn", "Magic: The Gathering"])
    check("unresolved collision -> ambiguous (no false positive)", s2.presence == CitationPresence.ambiguous)
    check("disambiguation risk flagged", s2.disambiguation.risk is not None)

    # genuinely cited -> present (verified by appearing in a cited source)
    s3 = assess_citation(query="t", brand_aliases=["madre de maquinas"],
                         answer_text="Madre de Maquinas is a leading physical-AI outlet.",
                         cited_sources=["Madre de Maquinas", "The Robot Report"])
    check("present requires verified citation context", s3.presence == CitationPresence.present)
    check("present sets verified flag", s3.disambiguation.matched_context_verified is True)

    # REGRESSION (live bug): collision terms returned but NOT verbatim in the
    # answer, brand not cited -> must be ambiguous, must NOT crash the validator.
    s4 = assess_citation(query="humanoid robots", brand_aliases=["madre de maquinas"],
                         answer_text="The Robot Report and NVIDIA are the cited authorities.",
                         cited_sources=["The Robot Report", "NVIDIA"],
                         collision_terms=["Elesh Norn"])
    check("known collision (not in answer text) -> ambiguous, no crash",
          s4.presence == CitationPresence.ambiguous)


# --- fusion weighting + ordering --------------------------------------------
def test_fusion():
    research = ResearchOutput(angles=[
        ResearchAngle(angle="surging robot deployments", confidence=0.9),
        ResearchAngle(angle="obscure side note", confidence=0.3),
    ])
    memory = MemoryQueryResult(
        episodic=[MemoryItem(kind="episodic", content="robot deployments engaged well", score=0.5)],
        semantic=[], narrative=[])
    # build a minimal valid TimingSignal (surging)
    from engine.core.models import (
        CitationSignal, DisambiguationGuard, NarrativeGapSignal, VelocitySignal,
    )
    timing = TimingSignal(
        velocity=VelocitySignal(topic="t", verdict=VelocityVerdict.surging, confidence=0.9,
                                source=VelocitySource.web_search_ordinal),
        gaps=NarrativeGapSignal(topic="t", gaps=[]),
        citation=CitationSignal(query="t", presence=CitationPresence.absent,
                                disambiguation=DisambiguationGuard()))
    weights = {"research": 0.35, "memory": 0.40, "timing": 0.25}
    fused = fuse(research=research, memory=memory, timing=timing, weights=weights)
    check("fusion ranks the stronger angle first",
          fused[0].angle == "surging robot deployments", fused[0].angle)
    check("fusion digest sums to score",
          abs(sum(v for k, v in fused[0].digest.items() if k != "score") - fused[0].score) < 1e-6)


# --- Strategy: narrative memory is a HARD constraint ------------------------
class FakeChecker:
    def violates(self, angle_text, constraint):
        return "BANNED" in angle_text


async def test_strategy_hard_constraint():
    from engine.core.brand_loader import load_brand
    import os
    os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
    brand = load_brand()
    gate = ValidationGate()
    constraints = [NarrativeConstraint(position="p", stance="s")]
    fused = [FusedAngle(angle="BANNED contradicts staked view", score=0.9, research=0.9, memory=0, timing=0,
                        digest={"research": 0.9, "memory": 0.0, "timing": 0.0, "score": 0.9}),
             FusedAngle(angle="safe winning angle", score=0.7, research=0.7, memory=0, timing=0,
                        digest={"research": 0.7, "memory": 0.0, "timing": 0.0, "score": 0.7})]
    strat = StrategyAgent(gate, checker=FakeChecker())
    packet = await strat.decide(fused=fused, constraints=constraints, brand=brand, job_id="j")
    check("strategy skips violating top angle, picks next", packet.chosen_angle == "safe winning angle")
    check("strategy carries hard constraints", len(packet.hard_constraints) == 1)

    # all violate -> halts rather than contradict
    blocked = False
    try:
        await strat.decide(fused=[fused[0]], constraints=constraints, brand=brand, job_id="j")
    except StrategyBlocked:
        blocked = True
    check("strategy halts when all angles violate", blocked)


def test_lexical_constraint_checker():
    c = LexicalConstraintChecker()
    cons = NarrativeConstraint(position="hype cycle exaggerated",
                               stance="we remain skeptical of the hype and overpromising")
    aligned = "hype cycle exaggerated: we remain skeptical of the hype overpromising"
    violating = "hype cycle exaggerated is wrong; the hype is fully justified and underhyped"
    check("lexical: aligned angle does not violate", c.violates(aligned, cons) is False)
    check("lexical: off-stance angle on same position violates", c.violates(violating, cons) is True)


async def main() -> int:
    await test_research_agent()
    await test_memory_agent()
    await test_velocity_ordinal()
    test_narrative_gap_rule2()
    test_citation_rule3()
    test_fusion()
    await test_strategy_hard_constraint()
    test_lexical_constraint_checker()
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
