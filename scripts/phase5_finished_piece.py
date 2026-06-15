"""
Cost of ONE finished piece, on top of the $0.64 Tier-1 pass.

Real calls: Copy (Sonnet), Prompt Engineer (Sonnet), Quality (Opus). Render is
MOCKED (Higgsfield MCP writes require approval not grantable autonomously; the
adapter is MCP-shaped and ready). Distribution stages X (reply mode) to log the
per-post X cost. Nothing publishes.

    python -m scripts.phase5_finished_piece
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv("/home/user/mutli-agent/.env")

from engine.agents.base import AgentContext                  # noqa: E402
from engine.agents.copy_agent import CopyAgent               # noqa: E402
from engine.agents.distribution import DistributionAgent     # noqa: E402
from engine.agents.production import ProductionAgent         # noqa: E402
from engine.agents.prompt_engineer import PromptEngineerAgent  # noqa: E402
from engine.agents.quality import QualityAgent               # noqa: E402
from engine.core.brand_loader import load_brand              # noqa: E402
from engine.core.cost_guard import CostGuard, InMemoryCostSink  # noqa: E402
from engine.core.dead_letter import InMemoryDeadLetterSink   # noqa: E402
from engine.core.llm import LLMClient                        # noqa: E402
from engine.core.models import PostingMode, StrategyPacket   # noqa: E402
from engine.core.validation_gate import ValidationGate       # noqa: E402
from engine.tools.higgsfield_mcp import MockProductionBackend  # noqa: E402
from engine.tools.social_apis import XAdapter                # noqa: E402

TIER1_USD = 0.6377  # measured in scripts/phase4_live_pass.py


async def main():
    brand = load_brand()
    cost = InMemoryCostSink()
    cg = CostGuard(daily_budget_usd=brand.budget.daily_usd, cost_sink=cost)
    gate = ValidationGate(InMemoryDeadLetterSink())
    ctx = AgentContext(llm=LLMClient(max_retries=3), cost_guard=cg, gate=gate, brand_id=brand.brand_id)
    job_id = "finished-piece-1"

    packet = StrategyPacket(
        chosen_angle=("Figure's BotQ factory flipped from prototyping to true mass production — "
                      "humanoids built like cars (one robot per hour)"),
        rationale="top fused angle from Tier-1", formats=brand.formats,
        fusion_weights=brand.fusion_weights, hard_constraints=[], inputs_digest={})

    copy = await CopyAgent(ctx).run(packet=packet, brand=brand, job_id=job_id)
    print(f"COPY: x_thread={len(copy.x_thread)} tweets, ig_caption={len(copy.ig_caption)} chars")

    prompt = await PromptEngineerAgent(ctx).run(packet=packet, brand=brand, job_id=job_id)
    print(f"PROMPT ENGINEER: placeholder embedded = {brand.character.placeholder in prompt.higgsfield_prompt}")

    backend = MockProductionBackend(cost_usd=0.30)  # MOCK render (see header)
    asset = await ProductionAgent(backend, cg, gate, brand.brand_id).run(
        prompt=prompt, character_element_id=brand.character.higgsfield_element_id, job_id=job_id)
    print(f"PRODUCTION (mock): status={asset.status} asset={asset.asset_url}")

    score, auto = await QualityAgent(ctx).evaluate(content=copy, brand=brand, job_id=job_id, has_video=True)
    print(f"QUALITY: overall={score.overall} route={score.route.value} auto_eligible={auto} (video -> human review)")

    dist = DistributionAgent([XAdapter(link_mode="reply")], cg, gate, brand.brand_id,
                             dead_letter_sink=InMemoryDeadLetterSink())
    content = {"x_thread": copy.x_thread, "ig_caption": copy.ig_caption,
               "youtube_script": copy.youtube_script, "link": "https://madredemaquinas.example/post"}
    bundle = await dist.stage(content=content, posting_mode=PostingMode.confirm, job_id=job_id)
    print(f"DISTRIBUTION: staged={bundle.plan.platforms} published={bundle.plan.published} "
          f"(confirm-mode; X cost ${bundle.staged['x'].estimated_cost_usd})")

    print("\n===== FINISHED-PIECE COST LEDGER =====")
    total = 0.0
    for e in cost.entries:
        total += e["usd"]
        print(f"  {e['agent']:18} {e['model']:16} in={e['input_tokens']:>5} out={e['output_tokens']:>5}  ${e['usd']:.4f}")
    print(f"  {'-'*60}")
    print(f"  Finished-piece subtotal:        ${total:.4f}")
    print(f"  + Tier-1 pass (measured):       ${TIER1_USD:.4f}")
    print(f"  = FULL PIPELINE / piece:        ${total + TIER1_USD:.4f}")
    print(f"  (daily budget ${brand.budget.daily_usd} -> ~{int(brand.budget.daily_usd / (total + TIER1_USD))} pieces/day before cap)")


if __name__ == "__main__":
    asyncio.run(main())
