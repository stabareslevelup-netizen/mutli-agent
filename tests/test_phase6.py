"""
Phase 6 orchestrator tests — FREE (smart fake LLM, no spend). Text-only
pipeline: no Production/Prompt Engineer tiers.

Covers tier activation, job-state progression, per-job cost, and halt paths:
  - happy path -> staged_for_review (confirm-mode, nothing published)
  - malformed mid-pipeline handoff -> job failed + dead-letter
  - narrative conflict -> halted_narrative_conflict
  - quality reject -> quality_rejected (no distribution)
  - over budget -> halted_budget BEFORE any Tier-1 spend
"""
from __future__ import annotations

import asyncio
import os

from engine.agents.base import AgentContext
from engine.agents.copy_agent import CopyAgent
from engine.agents.distribution import DistributionAgent
from engine.agents.memory import MemoryAgent
from engine.agents.orchestrator import Orchestrator, OrchestratorAgents
from engine.agents.quality import QualityAgent
from engine.agents.research import ResearchAgent
from engine.agents.skeptic import SkepticAgent
from engine.agents.strategy import StrategyAgent
from engine.agents.timing import TimingAgent
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.embeddings import NullEmbeddingProvider
from engine.core.job_manager import InMemoryJobStore
from engine.core.llm import Usage
from engine.core.models import PostingMode
from engine.core.validation_gate import ValidationGate
from engine.memory.backend import InMemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.semantic import SemanticMemory
from engine.tools.social_apis import XAdapter

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


_RESEARCH = ('{"angles":['
             '{"angle":"surging robot deployments on factory floors","rationale":"r","sources":[{"url":"http://a","title":"t"}],"confidence":0.9},'
             '{"angle":"deployment reality check","rationale":"r","sources":[{"url":"http://b","title":"t"}],"confidence":0.7},'
             '{"angle":"incident learnings","rationale":"r","sources":[{"url":"http://c","title":"t"}],"confidence":0.6}]}')
_SEARCH = '{"results":[{"title":"ramps production milestone","snippet":"unprecedented 24x scale-up first","published":"2026-05","url":"http://x"},{"title":"factory launch","snippet":"record","published":"2026","url":"http://y"}]}'
_CITE = '{"answer_text":"The Robot Report and CNBC are the cited authorities.","cited_sources":["The Robot Report","CNBC"],"collision_terms":[]}'
_COPY = '{"x_thread":["robots are here","part two"],"ig_caption":"caption","youtube_script":"script body"}'
_QUALITY_HI = '{"voice":0.9,"narrative":0.9,"format":0.9,"hook":0.9,"coherence":0.9,"reasons":["on voice"]}'
_QUALITY_LO = '{"voice":0.2,"narrative":0.2,"format":0.2,"hook":0.2,"coherence":0.2,"reasons":["off voice"]}'
_SKEPTIC = ('{"disputes":"d","hidden_assumptions":"h","alternative_explanations":"a",'
            '"overstatement":"o","skeptic_summary":"rests on a single announcement","confidence_adjustment":-0.05}')


class SmartFakeLLM:
    """Routes a canned response by a marker in the system prompt. Per-marker
    overrides allow injecting a malformed handoff."""
    def __init__(self, overrides: dict | None = None, quality=_QUALITY_HI):
        self.calls: list[str] = []
        self._ov = overrides or {}
        self._quality = quality

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048,
                       output_schema=None, **kw):
        s = system.lower()
        if "research analyst" in s:
            key, text = "research", _RESEARCH
        elif "web research tool" in s:
            key, text = "search", _SEARCH
        elif "answer-engine" in s:
            key, text = "citation", _CITE
        elif "adversarial" in s:
            key, text = "skeptic", _SKEPTIC
        elif "copywriter" in s:
            key, text = "copy", _COPY
        elif "quality reviewer" in s:
            key, text = "quality", self._quality
        else:
            key, text = "other", "{}"
        self.calls.append(key)
        if key in self._ov:
            text = self._ov[key]
        return text, Usage(input_tokens=300, output_tokens=100)


def _build(llm, *, daily_budget=25.0, strategy_checker=None, notifier=None):
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    cg = CostGuard(daily_budget_usd=daily_budget, cost_sink=cost)
    gate = ValidationGate(dl)
    ctx = AgentContext(llm=llm, cost_guard=cg, gate=gate, brand_id=brand.brand_id)
    be = InMemoryBackend()
    epi, sem, nar = EpisodicMemory(be, NullEmbeddingProvider()), SemanticMemory(be, NullEmbeddingProvider()), NarrativeMemory(be, NullEmbeddingProvider())
    agents = OrchestratorAgents(
        research=ResearchAgent(ctx), memory=MemoryAgent(epi, sem, nar), timing=TimingAgent(ctx),
        strategy=StrategyAgent(gate, checker=strategy_checker), skeptic=SkepticAgent(ctx),
        copy=CopyAgent(ctx), quality=QualityAgent(ctx),
        distribution=DistributionAgent([XAdapter()], cg, gate, brand.brand_id, dead_letter_sink=dl))
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=cg, brand=brand,
                        notifier=notifier)
    return orch, cost, dl, nar


async def test_happy_path():
    orch, cost, dl, _ = _build(SmartFakeLLM())
    res = await orch.run(topic="humanoid robots", entity="Figure")
    check("happy path -> staged_for_review", res.status == "staged_for_review", res.status)
    job = await orch._jobs.get(res.job_id)
    check("job reached tier 5", job.current_tier == 5)
    check("job state persisted", job.status == "staged_for_review")
    check("per-job cost recorded > 0", res.cost_usd > 0, str(res.cost_usd))
    bundle = res.artifacts.get("bundle")
    check("distribution staged confirm-mode, nothing published",
          bundle and bundle.plan.posting_mode == PostingMode.confirm and not bundle.plan.published)
    check("skeptic summary reaches the review item", res.artifacts["packet"].skeptic_summary != "")


async def test_malformed_handoff_fails():
    orch, cost, dl, _ = _build(SmartFakeLLM(overrides={"copy": "sorry no json"}))
    res = await orch.run(topic="x")
    check("malformed copy -> job failed", res.status == "failed", res.status)
    check("failure dead-lettered", len(dl.records) >= 1)


async def test_narrative_conflict():
    class FlagAll:
        def violates(self, angle_text, c):
            return True
    orch, cost, dl, nar = _build(SmartFakeLLM(), strategy_checker=FlagAll())
    await nar.stake_position(brand_id="madre_de_maquinas", position="p", stance="s")
    res = await orch.run(topic="x")
    check("all angles conflict -> halted_narrative_conflict",
          res.status == "halted_narrative_conflict", res.status)


async def test_quality_reject():
    orch, cost, dl, _ = _build(SmartFakeLLM(quality=_QUALITY_LO))
    res = await orch.run(topic="x")
    check("low quality -> quality_rejected", res.status == "quality_rejected", res.status)
    check("rejected job has no distribution bundle", "bundle" not in res.artifacts)


async def test_budget_halt():
    orch, cost, dl, _ = _build(SmartFakeLLM(), daily_budget=0.50)
    # pre-spend over the cap
    await orch._cost.record_external(job_id="seed", brand_id="b", agent="seed", usd=1.0, label="seed")
    llm = orch._a.research.ctx.llm
    before_calls = len(llm.calls)
    res = await orch.run(topic="x")
    check("over budget -> halted_budget", res.status == "halted_budget", res.status)
    check("no Tier-1 spend when budget-halted (no agent LLM calls)",
          len(llm.calls) == before_calls, f"calls={len(llm.calls) - before_calls}")


async def test_raising_notifier_never_fails_the_job():
    class RaisingNotifier:
        async def notify_staged(self, **kwargs):
            raise ConnectionError("slack is down, and this notifier forgot to catch it")

    orch, cost, dl, _ = _build(SmartFakeLLM(), notifier=RaisingNotifier())
    res = await orch.run(topic="humanoid robots", entity="Figure")
    check("notifier raising -> job still staged_for_review, not failed",
          res.status == "staged_for_review", res.status)
    job = await orch._jobs.get(res.job_id)
    check("job state persisted as staged_for_review despite notifier exception",
          job.status == "staged_for_review", job.status)


async def main() -> int:
    await test_happy_path()
    await test_malformed_handoff_fails()
    await test_narrative_conflict()
    await test_quality_reject()
    await test_budget_halt()
    await test_raising_notifier_never_fails_the_job()
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
