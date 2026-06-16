"""
ONE real end-to-end orchestrated job. Real Tier-1 (Research/Memory/Timing) +
Strategy + Tier-3 (Copy/Prompt) + Quality; render is MOCKED (avoids the ~$3.21
real render). Proves tier activation, parallel fan-out, job state, confirm-mode
staging, and per-job cost.

    python -m scripts.phase6_full_run "topic" [entity]
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv("/home/user/mutli-agent/.env")

from engine.agents.base import AgentContext               # noqa: E402
from engine.agents.copy_agent import CopyAgent            # noqa: E402
from engine.agents.distribution import DistributionAgent  # noqa: E402
from engine.agents.memory import MemoryAgent              # noqa: E402
from engine.agents.orchestrator import Orchestrator, OrchestratorAgents  # noqa: E402
from engine.agents.production import ProductionAgent      # noqa: E402
from engine.agents.prompt_engineer import PromptEngineerAgent  # noqa: E402
from engine.agents.quality import QualityAgent            # noqa: E402
from engine.agents.research import ResearchAgent          # noqa: E402
from engine.agents.strategy import StrategyAgent          # noqa: E402
from engine.agents.timing import TimingAgent              # noqa: E402
from engine.core.brand_loader import load_brand           # noqa: E402
from engine.core.cost_guard import CostGuard, InMemoryCostSink  # noqa: E402
from engine.core.dead_letter import InMemoryDeadLetterSink  # noqa: E402
from engine.core.embeddings import build_embedding_provider, NullEmbeddingProvider  # noqa: E402
from engine.core.job_manager import InMemoryJobStore      # noqa: E402
from engine.core.llm import LLMClient                     # noqa: E402
from engine.core.validation_gate import ValidationGate    # noqa: E402
from engine.memory.backend import InMemoryBackend         # noqa: E402
from engine.memory.episodic import EpisodicMemory         # noqa: E402
from engine.memory.narrative import NarrativeMemory       # noqa: E402
from engine.memory.semantic import SemanticMemory         # noqa: E402
from engine.tools.higgsfield_mcp import MockProductionBackend  # noqa: E402
from engine.tools.social_apis import InstagramAdapter, XAdapter, YouTubeAdapter  # noqa: E402


async def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "humanoid robots on real factory floors"
    entity = sys.argv[2] if len(sys.argv) > 2 else "Figure"
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    cg = CostGuard(daily_budget_usd=brand.budget.daily_usd, cost_sink=cost)
    gate = ValidationGate(dl)
    ctx = AgentContext(llm=LLMClient(max_retries=3), cost_guard=cg, gate=gate, brand_id=brand.brand_id)

    be = InMemoryBackend()
    emb = build_embedding_provider()
    epi, sem, nar = EpisodicMemory(be, emb), SemanticMemory(be, emb), NarrativeMemory(be, NullEmbeddingProvider())
    await nar.stake_position(brand_id=brand.brand_id, position="autonomy timelines overstated",
                             stance="physical AI is far from general autonomy")

    agents = OrchestratorAgents(
        research=ResearchAgent(ctx), memory=MemoryAgent(epi, sem, nar), timing=TimingAgent(ctx),
        strategy=StrategyAgent(gate), copy=CopyAgent(ctx), prompt_engineer=PromptEngineerAgent(ctx),
        production=ProductionAgent(MockProductionBackend(), cg, gate, brand.brand_id),
        quality=QualityAgent(ctx),
        distribution=DistributionAgent([XAdapter(link_mode="reply"), InstagramAdapter(), YouTubeAdapter()],
                                       cg, gate, brand.brand_id, dead_letter_sink=dl))
    orch = Orchestrator(agents=agents, job_store=InMemoryJobStore(), cost_guard=cg, brand=brand)

    print(f"\nTOPIC: {topic}  (entity={entity})\n")
    res = await orch.run(topic=topic, entity=entity, trigger="manual")
    job = await orch._jobs.get(res.job_id)

    print(f"JOB {res.job_id}")
    print(f"  status={res.status}  tier_reached={job.current_tier}  cost=${res.cost_usd:.4f}")
    if res.status == "staged_for_review":
        a = res.artifacts
        print(f"  chosen_angle: {a['packet'].chosen_angle[:90]}")
        print(f"  timing: velocity={a['timing'].velocity.verdict.value} "
              f"citation={a['timing'].citation.presence.value} "
              f"open_gaps={sum(1 for g in a['timing'].gaps.gaps if g.is_open)}")
        print(f"  copy: {len(a['copy'].x_thread)} tweets | quality={a['quality'].overall} "
              f"route={a['quality'].route.value} auto_eligible={a['auto_eligible']}")
        print(f"  render(mock): {a['asset'].status} {a['asset'].asset_url}")
        print(f"  staged platforms: {a['bundle'].plan.platforms} published={a['bundle'].plan.published}")
    else:
        print(f"  reason: {res.reason}")

    print("\n  per-agent cost:")
    for e in cost.entries:
        print(f"    {e['agent']:18} {e['model']:16} ${e['usd']:.4f}")
    print(f"  dead-letters: {len(dl.records)}")


if __name__ == "__main__":
    asyncio.run(main())
