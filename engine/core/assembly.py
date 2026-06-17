"""
engine/core/assembly.py — wire a ready-to-run Orchestrator from a brand config.

Centralizes the agent wiring the phase scripts duplicated. Everything is
injectable: pass a real Higgsfield backend / Sql stores in production, or take
the defaults (mock render, in-memory stores) for dev and tests. The LLM client
defaults to the real Anthropic client but can be swapped for a fake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from engine.agents.base import AgentContext
from engine.agents.copy_agent import CopyAgent
from engine.agents.distribution import DistributionAgent
from engine.agents.memory import MemoryAgent
from engine.agents.orchestrator import Orchestrator, OrchestratorAgents
from engine.agents.production import ProductionAgent
from engine.agents.prompt_engineer import PromptEngineerAgent
from engine.agents.quality import QualityAgent
from engine.agents.research import ResearchAgent
from engine.agents.strategy import StrategyAgent
from engine.agents.timing import TimingAgent
from engine.core.brand_loader import BrandConfig
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.embeddings import build_embedding_provider
from engine.core.job_manager import InMemoryJobStore, JobStore
from engine.core.llm import LLMClient
from engine.core.validation_gate import ValidationGate
from engine.memory.backend import InMemoryBackend, MemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.semantic import SemanticMemory
from engine.tools.higgsfield_mcp import MockProductionBackend, ProductionBackend
from engine.tools.social_apis import InstagramAdapter, XAdapter, YouTubeAdapter


@dataclass
class AssembledEngine:
    orchestrator: Orchestrator
    cost_guard: CostGuard
    cost_sink: Any
    dead_letter: Any
    memory: tuple  # (episodic, semantic, narrative)


def build_orchestrator(brand: BrandConfig, *, llm: Optional[Any] = None,
                       production_backend: Optional[ProductionBackend] = None,
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

    prod = production_backend or MockProductionBackend()  # real render needs MCP/egress

    agents = OrchestratorAgents(
        research=ResearchAgent(ctx),
        memory=MemoryAgent(epi, sem, nar),
        timing=TimingAgent(ctx),
        strategy=StrategyAgent(gate),
        copy=CopyAgent(ctx),
        prompt_engineer=PromptEngineerAgent(ctx),
        production=ProductionAgent(prod, cg, gate, brand.brand_id),
        quality=QualityAgent(ctx),
        distribution=DistributionAgent(
            [XAdapter(link_mode="reply"), InstagramAdapter(), YouTubeAdapter()],
            cg, gate, brand.brand_id, dead_letter_sink=dl),
    )
    orch = Orchestrator(agents=agents, job_store=job_store or InMemoryJobStore(),
                        cost_guard=cg, brand=brand, dead_letter_sink=dl)
    return AssembledEngine(orchestrator=orch, cost_guard=cg, cost_sink=cost_sink,
                           dead_letter=dl, memory=(epi, sem, nar))
