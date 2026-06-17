"""
Phase 9 tests — FREE (fakes, no spend). The three fixes + the feedback loop.

Fix 1  Copy verification gate: requires_hedging flips from timing; the
       verification directive is injected into the Copy prompt.
Fix 2  Skeptic agent: structured review; confidence_adjustment clamped <= 0;
       orchestrator enriches the packet before Copy.
Fix 3  Prompt Engineer: scene matches the article (pillar visual_mood injected)
       and OCHO placeholder always present.
Phase 9 feedback loop: ingest -> episodic + semantic auto-update; a 5+ job
       pattern surfaces a guarded procedural proposal (never auto-applies).
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from engine.agents.base import AgentContext
from engine.agents.copy_agent import CopyAgent
from engine.agents.prompt_engineer import PromptEngineerAgent
from engine.agents.skeptic import SkepticAgent
from engine.agents.strategy import StrategyAgent, choose_content_format, match_pillar
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_SONNET, model_for
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.embeddings import NullEmbeddingProvider
from engine.core.feedback_service import FeedbackService, InMemoryFeedbackStore, engagement_score
from engine.core.llm import Usage
from engine.core.models import (
    AgentAttribution, CitationPresence, CitationSignal, ContentFormat, DisambiguationGuard,
    NarrativeGapSignal, ReviewItem, SkepticReview, StrategyPacket, TimingSignal,
    VelocitySignal, VelocitySource, VelocityVerdict,
)
from engine.core.review_store import InMemoryReviewStore
from engine.core.validation_gate import ValidationGate
from engine.memory.backend import InMemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.procedural import ProceduralMemory
from engine.memory.semantic import SemanticMemory

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


class CapturingLLM:
    def __init__(self, resp):
        self.resp = resp
        self.last_user = self.last_system = None

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, output_schema=None, **kw):
        self.last_system, self.last_user = system, user
        return self.resp, Usage(input_tokens=10, output_tokens=10)


def _ctx(llm):
    return AgentContext(llm=llm, cost_guard=CostGuard(daily_budget_usd=25, cost_sink=InMemoryCostSink()),
                        gate=ValidationGate(InMemoryDeadLetterSink()), brand_id="madre_de_maquinas")


def _timing(verdict, confidence, citation):
    guard = DisambiguationGuard(matched_context_verified=(citation == CitationPresence.present))
    return TimingSignal(
        velocity=VelocitySignal(topic="t", verdict=verdict, confidence=confidence,
                                source=VelocitySource.web_search_ordinal),
        gaps=NarrativeGapSignal(topic="t", gaps=[]),
        citation=CitationSignal(query="t", presence=citation, disambiguation=guard))


# --- Fix 1: Copy verification gate -----------------------------------------
async def test_fix1_hedging_flag_and_injection():
    brand = load_brand()
    gate = ValidationGate(InMemoryDeadLetterSink())
    from engine.core.fusion import FusedAngle
    fused = [FusedAngle(angle="a robot deployment claim", score=0.8, research=0.8, memory=0, timing=0,
                        digest={"research": 0.8, "memory": 0.0, "timing": 0.0, "score": 0.8})]

    # ambiguous citation -> requires_hedging True
    p_amb = await StrategyAgent(gate).decide(fused=fused, constraints=[], brand=brand, job_id="j",
                                             timing=_timing(VelocityVerdict.surging, 0.9, CitationPresence.ambiguous))
    check("ambiguous citation -> requires_hedging", p_amb.requires_hedging is True)
    # low velocity confidence -> requires_hedging True
    p_low = await StrategyAgent(gate).decide(fused=fused, constraints=[], brand=brand, job_id="j",
                                             timing=_timing(VelocityVerdict.steady, 0.4, CitationPresence.absent))
    check("low velocity confidence -> requires_hedging", p_low.requires_hedging is True)
    # solid signal -> no hedging
    p_ok = await StrategyAgent(gate).decide(fused=fused, constraints=[], brand=brand, job_id="j",
                                            timing=_timing(VelocityVerdict.steady, 0.9, CitationPresence.present))
    check("solid signal -> no hedging", p_ok.requires_hedging is False)

    # Copy injects the verification directive only when hedging is required
    llm = CapturingLLM('{"x_thread":["t"],"ig_caption":"c","youtube_script":"y"}')
    await CopyAgent(_ctx(llm)).run(packet=p_amb, brand=brand, job_id="j")
    check("hedging on -> VERIFICATION MODE injected into copy prompt", "VERIFICATION MODE" in llm.last_user)
    check("copy system carries the hedging rule", "reportedly" in llm.last_system.lower())
    llm2 = CapturingLLM('{"x_thread":["t"],"ig_caption":"c","youtube_script":"y"}')
    await CopyAgent(_ctx(llm2)).run(packet=p_ok, brand=brand, job_id="j")
    check("hedging off -> no verification directive", "VERIFICATION MODE" not in llm2.last_user)


# --- Fix 2: Skeptic agent ---------------------------------------------------
async def test_fix2_skeptic():
    check("skeptic runs on Sonnet tier", model_for("skeptic") == MODEL_SONNET)
    # model returns a positive adjustment -> clamped to 0.0 (never raises confidence)
    resp = ('{"disputes":"d","hidden_assumptions":"h","alternative_explanations":"a",'
            '"overstatement":"o","skeptic_summary":"single-source claim","confidence_adjustment":0.5}')
    rev = await SkepticAgent(_ctx(CapturingLLM(resp))).review(
        packet=StrategyPacket(chosen_angle="x"), job_id="j")
    check("skeptic returns SkepticReview", isinstance(rev, SkepticReview))
    check("confidence_adjustment clamped to <= 0", rev.confidence_adjustment == 0.0, str(rev.confidence_adjustment))
    check("skeptic_summary populated", rev.skeptic_summary != "")


# --- Fix 3: Prompt Engineer scene matches article --------------------------
async def test_fix3_prompt_engineer():
    brand = load_brand()
    # incident_file pillar has an amber/high-stakes visual_mood in the config
    packet = StrategyPacket(chosen_angle="a humanoid fails on the line", pillar_id="incident_file")
    llm = CapturingLLM('{"higgsfield_prompt":"a tense scene"}')  # model omits placeholder on purpose
    out = await PromptEngineerAgent(_ctx(llm)).run(packet=packet, brand=brand, job_id="j")
    mood = next(p.visual_mood for p in brand.pillars if p.id == "incident_file")
    check("scene prompt receives the pillar visual mood", mood[:20] in llm.last_user)
    check("scene prompt receives the article angle", "humanoid fails" in llm.last_user)
    check("OCHO placeholder always present (injected if model omits)",
          brand.character.placeholder in out.higgsfield_prompt)


# --- Phase 9 feedback loop --------------------------------------------------
def _review_item(job_id, pillar, fmt, velocity, angle):
    return ReviewItem(job_id=job_id, brand_id="madre_de_maquinas", status="approved",
                      content_format=ContentFormat(fmt), pillar_id=pillar, chosen_angle=angle,
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

    # one ingest -> episodic + semantic auto-update
    rs.put(_review_item("J0", "incident_file", "video", "surging", "an incident"))
    r0 = await fb.ingest(job_id="J0", engagement={"x": {"views": 1000, "likes": 50, "shares": 10}})
    check("ingest returns engagement score", r0["engagement_score"] > 0)
    check("episodic auto-updated", r0["episodic_updated"] and len(await epi.query(brand_id=brand.brand_id, text="incident", k=9)) >= 1)
    check("semantic auto-updated", r0["semantic_updated"] and len(await sem.query(brand_id=brand.brand_id, text="performance", k=9)) >= 1)
    check("single ingest does not surface a proposal yet", r0["proposal_surfaced"] is None)

    # 5 strong incident_file+video jobs, 3 weak company_intel+image jobs
    proposal = None
    for i in range(1, 5):
        rs.put(_review_item(f"A{i}", "incident_file", "video", "surging", "incident angle"))
        r = await fb.ingest(job_id=f"A{i}", engagement={"x": {"views": 5000, "likes": 400, "shares": 80, "saves": 60}})
        proposal = proposal or r["proposal_surfaced"]
    for i in range(1, 4):
        rs.put(_review_item(f"B{i}", "company_intel", "image", "steady", "company angle"))
        r = await fb.ingest(job_id=f"B{i}", engagement={"x": {"views": 400, "likes": 5}})
        proposal = proposal or r["proposal_surfaced"]

    check("pattern across 5+ jobs surfaces a procedural proposal", proposal is not None)
    pending = await proc.pending_proposals()
    check("proposal is pending (NOT auto-applied)", len(pending) == 1 and pending[0].status == "pending")
    check("proposal targets strategy with performance data",
          pending[0].agent == "strategy" and "best" in pending[0].performance_data)
    check("proposal carries a voice-similarity check", pending[0].voice_similarity is not None)
    # the live procedural prompt is unchanged until a human approves
    check("nothing auto-applied to procedural memory",
          await proc.active_prompt(brand_id=brand.brand_id, agent="strategy") is None)


async def main_async() -> int:
    await test_fix1_hedging_flag_and_injection()
    await test_fix2_skeptic()
    await test_fix3_prompt_engineer()
    await test_feedback_loop()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
