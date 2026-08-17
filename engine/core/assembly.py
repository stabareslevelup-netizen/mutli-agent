"""
engine/core/assembly.py — wire a ready-to-run Orchestrator from a brand config.

Centralizes the agent wiring the phase scripts duplicated. Everything is
injectable: pass Sql stores in production, or take the defaults (in-memory
stores) for dev and tests. The LLM client defaults to the real Anthropic
client but can be swapped for a fake. Text-only pipeline — no render backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
from engine.core.brand_loader import BrandConfig
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.embeddings import build_embedding_provider
from engine.core.job_manager import InMemoryJobStore, JobStore
from engine.core.llm import LLMClient
from engine.core.narrative_conflict_sink import InMemoryNarrativeConflictSink
from engine.core.post_history_store import InMemoryPostHistoryStore
from engine.core.validation_gate import ValidationGate
from engine.memory.backend import InMemoryBackend, MemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.procedural import ProceduralMemory
from engine.memory.semantic import SemanticMemory
from engine.core.review_store import InMemoryReviewStore
from engine.tools.social_apis import InstagramAdapter, XAdapter, YouTubeAdapter


@dataclass
class AssembledEngine:
    orchestrator: Orchestrator
    cost_guard: CostGuard
    cost_sink: Any
    dead_letter: Any
    memory: tuple  # (episodic, semantic, narrative)
    review_store: Any = None
    procedural: Any = None
    distribution: Any = None


def build_orchestrator(brand: BrandConfig, *, llm: Optional[Any] = None,
                       memory_backend: Optional[MemoryBackend] = None,
                       job_store: Optional[JobStore] = None,
                       cost_sink: Optional[Any] = None,
                       dead_letter_sink: Optional[Any] = None) -> AssembledEngine:
    cost_sink = cost_sink or InMemoryCostSink()
    dl = dead_letter_sink or InMemoryDeadLetterSink()
    cg = CostGuard(daily_budget_usd=brand.budget.daily_usd, cost_sink=cost_sink)
    gate = ValidationGate(dl)
    ctx = AgentContext(llm=llm or LLMClient(max_retries=3), cost_guard=cg,
                       gate=gate, brand_id=brand.brand_id)

    be = memory_backend or InMemoryBackend()
    emb = build_embedding_provider()
    epi = EpisodicMemory(be, emb)
    sem = SemanticMemory(be, emb)
    nar = NarrativeMemory(be, emb)
    proc = ProceduralMemory(be, reference_set=[brand.voice], voice_threshold=0.2)
    review_store = InMemoryReviewStore()

    # Phase 2: shared between TimingAgent (reads: duplicate_check /
    # narrative_gap_check) and Orchestrator (writes: records at stage time) —
    # must be the SAME instance/provider on both sides, or Timing would never
    # see what Orchestrator recorded.
    post_history = InMemoryPostHistoryStore()
    narrative_conflicts = InMemoryNarrativeConflictSink()

    dist = DistributionAgent(
        [XAdapter(link_mode="reply"), InstagramAdapter(), YouTubeAdapter()],
        cg, gate, brand.brand_id, dead_letter_sink=dl)

    agents = OrchestratorAgents(
        research=ResearchAgent(ctx),
        memory=MemoryAgent(epi, sem, nar),
        timing=TimingAgent(ctx, post_history=post_history, embeddings=emb),
        strategy=StrategyAgent(ctx, narrative_conflict_sink=narrative_conflicts),
        skeptic=SkepticAgent(ctx),
        copy=CopyAgent(ctx),
        quality=QualityAgent(ctx),
        distribution=dist,
    )
    from engine.tools.notify import build_notifier_from_env
    orch = Orchestrator(agents=agents, job_store=job_store or InMemoryJobStore(),
                        cost_guard=cg, brand=brand, dead_letter_sink=dl,
                        post_history_store=post_history, embeddings=emb,
                        review_store=review_store, notifier=build_notifier_from_env())
    return AssembledEngine(orchestrator=orch, cost_guard=cg, cost_sink=cost_sink,
                           dead_letter=dl, memory=(epi, sem, nar),
                           review_store=review_store, procedural=proc, distribution=dist)
