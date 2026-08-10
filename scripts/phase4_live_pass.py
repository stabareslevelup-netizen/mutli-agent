"""
ONE capped live Tier-1 pipeline pass against the real Anthropic API.

Proves Research + Timing work live (web_search) and prints what a full Tier-1
pass costs. Memory/fusion/Strategy are deterministic (no token cost). Run:

    python -m scripts.phase4_live_pass "your topic" [entity]

Caps spend to ~3 web_search-backed Opus calls (Research x1, Timing x2).
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv("/home/user/mutli-agent/.env")

from engine.agents.base import AgentContext              # noqa: E402
from engine.agents.memory import MemoryAgent             # noqa: E402
from engine.agents.research import ResearchAgent         # noqa: E402
from engine.agents.strategy import StrategyAgent, StrategyBlocked  # noqa: E402
from engine.agents.timing import TimingAgent             # noqa: E402
from engine.core.brand_loader import load_brand          # noqa: E402
from engine.core.cost_guard import CostGuard, InMemoryCostSink  # noqa: E402
from engine.core.dead_letter import InMemoryDeadLetterSink      # noqa: E402
from engine.core.embeddings import build_embedding_provider     # noqa: E402
from engine.core.fusion import fuse                      # noqa: E402
from engine.core.llm import LLMClient                    # noqa: E402
from engine.core.validation_gate import ValidationGate   # noqa: E402
from engine.memory.backend import InMemoryBackend        # noqa: E402
from engine.memory.episodic import EpisodicMemory        # noqa: E402
from engine.memory.narrative import NarrativeMemory      # noqa: E402
from engine.memory.semantic import SemanticMemory        # noqa: E402


async def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "Figure 03 humanoid robot factory deployment"
    entity = sys.argv[2] if len(sys.argv) > 2 else "Figure"
    job_id = "live-pass-1"

    brand = load_brand()
    cost_sink = InMemoryCostSink()
    cost_guard = CostGuard(daily_budget_usd=brand.budget.daily_usd, cost_sink=cost_sink)
    gate = ValidationGate(dead_letter_sink=InMemoryDeadLetterSink())
    ctx = AgentContext(llm=LLMClient(max_retries=3), cost_guard=cost_guard,
                       gate=gate, brand_id=brand.brand_id)

    # memory (offline; degraded embeddings here) — seed one staked position
    emb = build_embedding_provider()
    be = InMemoryBackend()
    epi, sem, nar = EpisodicMemory(be, emb), SemanticMemory(be, emb), NarrativeMemory(be, emb)
    await nar.stake_position(brand_id=brand.brand_id,
                             position="autonomy timelines are overstated",
                             stance="physical AI is far from general autonomy")
    memory_agent = MemoryAgent(epi, sem, nar)

    print(f"\nTOPIC: {topic}   (entity={entity})   embeddings={emb.name}\n")

    research = await ResearchAgent(ctx).run(topic=topic, job_id=job_id)
    print(f"RESEARCH: {len(research.angles)} angles")
    for a in research.angles:
        print(f"  - ({a.confidence:.2f}) {a.angle}")

    memory = await memory_agent.query(brand_id=brand.brand_id, text=topic, k=5)
    print(f"\nMEMORY: episodic={len(memory.episodic)} semantic={len(memory.semantic)} "
          f"constraints={len(memory.narrative)}")

    timing = await TimingAgent(ctx).run(
        topic=topic, job_id=job_id,
        pillars=[p.model_dump() for p in brand.pillars],
        brand_aliases=[brand.display_name, brand.brand_id, brand.character_name],
        entity=entity)
    print(f"\nTIMING:\n  velocity: {timing.velocity.verdict.value} "
          f"(conf {timing.velocity.confidence}, src {timing.velocity.source.value}, "
          f"numeric {timing.velocity.numeric_rate})")
    print(f"  gaps (open):")
    for g in timing.gaps.gaps:
        if g.is_open:
            tag = f"[{g.gap_type.value}]"
            print(f"    {tag} {g.angle}  (density {g.coverage_density}; {g.evidence})")
    print(f"  citation: {timing.citation.presence.value}  incumbents={timing.citation.incumbents[:4]}")
    if timing.citation.disambiguation.risk:
        print(f"            ! {timing.citation.disambiguation.risk}")

    fused = fuse(research=research, memory=memory, timing=timing, weights=brand.fusion_weights)
    print(f"\nFUSION (weights {brand.fusion_weights}):")
    for f in fused:
        print(f"  {f.score:.3f}  {f.angle}   {f.digest}")

    try:
        packet = await StrategyAgent(gate).decide(
            fused=fused, constraints=memory.narrative, brand=brand, job_id=job_id)
        print(f"\nSTRATEGY PACKET:\n  chosen_angle: {packet.chosen_angle}\n  "
              f"formats: {packet.formats}\n  rationale: {packet.rationale}")
    except StrategyBlocked as e:
        print(f"\nSTRATEGY HALTED (hard constraint): {e}")

    print("\n===== COST LEDGER (this pass) =====")
    total = 0.0
    for e in cost_sink.entries:
        total += e["usd"]
        print(f"  {e['agent']:9} {e['model']:18} in={e['input_tokens']:>6} "
              f"out={e['output_tokens']:>5}  ${e['usd']:.4f}")
    print(f"  {'-'*52}\n  TOTAL Tier-1 pass: ${total:.4f}  "
          f"(spent_today=${cost_guard.spent_today():.4f} / budget ${brand.budget.daily_usd})")


if __name__ == "__main__":
    asyncio.run(main())
