"""
Phase 8 tests — FREE (fake LLM / mock render, no spend, no network).

Covers the two scope additions + the dashboard backend:
  - format routing (Strategy choose_content_format: velocity/depth/pillar)
  - format-aware Production (video/image render, text_only skip) + cost_log tag
  - review service: queue, one-tap approve, one-tap reject -> dead-letter
  - agent attribution populated on staged jobs
  - procedural proposals: list, approve, voice-regression block, reject, rollback
  - full HTTP surface (POST /jobs -> queue -> approve, dashboard served)
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

from engine.agents.production import ProductionAgent
from engine.agents.strategy import choose_content_format
from engine.core.assembly import build_orchestrator
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.llm import Usage
from engine.core.models import ContentFormat, PromptEngineerOutput, VelocityVerdict
from engine.core.review_service import ReviewService
from engine.core.validation_gate import ValidationGate
from engine.tools.higgsfield_mcp import MockProductionBackend

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


# canned LLM responses keyed by system-prompt marker (reused shape from phase 6)
_R = ('{"angles":[{"angle":"robots on factory floors, the deployment reality","rationale":"r","sources":[{"url":"http://a","title":"t"}],"confidence":0.9},'
      '{"angle":"alt angle two","rationale":"r","sources":[{"url":"http://b","title":"t"}],"confidence":0.7},'
      '{"angle":"alt angle three","rationale":"r","sources":[{"url":"http://c","title":"t"}],"confidence":0.6}]}')
_S = '{"results":[{"title":"steady update","snippet":"quietly improving","published":"2026","url":"http://x"}]}'
_C = '{"answer_text":"The Robot Report is cited.","cited_sources":["The Robot Report"],"collision_terms":[]}'
_COPY = '{"x_thread":["t1","t2","t3"],"ig_caption":"cap","youtube_script":"yt"}'
_PROMPT = '{"higgsfield_prompt":"a cinematic scene"}'
_Q = '{"voice":0.9,"narrative":0.9,"format":0.9,"hook":0.9,"coherence":0.9,"reasons":["ok"]}'


class SmartFakeLLM:
    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, output_schema=None, **kw):
        s = system.lower()
        text = ("{}" if True else "")
        if "research analyst" in s: text = _R
        elif "web research tool" in s: text = _S
        elif "answer-engine" in s: text = _C
        elif "copywriter" in s: text = _COPY
        elif "prompt engineer" in s: text = _PROMPT
        elif "quality reviewer" in s: text = _Q
        return text, Usage(input_tokens=200, output_tokens=60)


# --- format routing --------------------------------------------------------
def test_format_routing():
    brand = load_brand()
    p = brand.pillars
    # company_intel pillar (hint=image), breaking velocity, shallow -> image (speed)
    f1 = choose_content_format(VelocityVerdict.surging,
                               "what companies are building and deciding next", p)
    check("breaking + company_intel -> image", f1 == ContentFormat.image, f1.value)
    # incident pillar (hint=video) + deep cue -> video even when surging (depth wins)
    f2 = choose_content_format(VelocityVerdict.surging,
                               "when physical ai fails what we learn, an incident", p)
    check("breaking + incident/deep -> video", f2 == ContentFormat.video, f2.value)
    # low velocity, company_intel -> image baseline
    f3 = choose_content_format(VelocityVerdict.steady,
                               "what companies are building and deciding", p)
    check("steady + company_intel -> image baseline", f3 == ContentFormat.image, f3.value)
    # unknown/default -> video
    f4 = choose_content_format(VelocityVerdict.steady, "a deep deployment reality story", p)
    check("deep deployment -> video", f4 == ContentFormat.video, f4.value)


# --- format-aware production + cost_log tag --------------------------------
async def test_production_routing():
    brand = load_brand()
    po = PromptEngineerOutput(higgsfield_prompt=f"{brand.character.placeholder} scene",
                              character_placeholder=brand.character.placeholder)

    for fmt, expect_calls, expect_status in [
            (ContentFormat.video, 1, "ready"), (ContentFormat.image, 1, "ready"),
            (ContentFormat.text_only, 0, "skipped")]:
        cost = InMemoryCostSink()
        backend = MockProductionBackend()
        agent = ProductionAgent(backend, CostGuard(daily_budget_usd=25, cost_sink=cost),
                                ValidationGate(InMemoryDeadLetterSink()), brand.brand_id)
        prompt = None if fmt == ContentFormat.text_only else po
        res = await agent.run(prompt=prompt, character_element_id="x", job_id="j", content_format=fmt)
        check(f"{fmt.value}: render calls={expect_calls}", backend.calls == expect_calls)
        check(f"{fmt.value}: status={expect_status}", res.status == expect_status)
        check(f"{fmt.value}: cost_log tagged with format",
              cost.entries[0]["model"] == f"mock:{fmt.value}", cost.entries[0]["model"])
    # image cheaper than video
    cost_v, cost_i = InMemoryCostSink(), InMemoryCostSink()
    for sink, fmt in [(cost_v, ContentFormat.video), (cost_i, ContentFormat.image)]:
        a = ProductionAgent(MockProductionBackend(), CostGuard(daily_budget_usd=25, cost_sink=sink),
                            ValidationGate(InMemoryDeadLetterSink()), brand.brand_id)
        await a.run(prompt=po, character_element_id="x", job_id="j", content_format=fmt)
    check("image render cheaper than video", cost_i.entries[0]["usd"] < cost_v.entries[0]["usd"])


# --- review service + attribution (integration with fake LLM) --------------
async def test_review_service():
    brand = load_brand()
    eng = build_orchestrator(brand, llm=SmartFakeLLM(), production_backend=MockProductionBackend())
    res = await eng.orchestrator.run(topic="humanoid robots", entity="Figure")
    check("job staged", res.status == "staged_for_review", res.status)

    svc = ReviewService(review_store=eng.review_store, distribution=eng.distribution,
                        procedural=eng.procedural, dead_letter_sink=eng.dead_letter, x_http=None)
    q = await svc.queue()
    check("review queue has the staged job", len(q) == 1)
    item = q[0]
    a = item.attribution
    check("attribution: research chosen + alternatives", a.research_chosen and len(a.research_alternatives) >= 1)
    check("attribution: timing velocity present", a.timing_velocity != "")
    check("attribution: quality 5 dims + overall", set(a.quality) >= {"voice","narrative","format","hook","coherence","overall"})
    check("attribution: copy output time recorded", a.copy_output_seconds >= 0)

    appr = await svc.approve(job_id=item.job_id, confirmed=True)
    check("approve -> approved (publish pending, no live client)", appr["status"] == "approved")
    check("approved job leaves the queue", len(await svc.queue()) == 0)

    # reject a fresh job -> dead-letter
    res2 = await eng.orchestrator.run(topic="another topic", entity="Figure")
    rej = await svc.reject(job_id=res2.job_id, reason="off-voice")
    check("reject -> rejected", rej["status"] == "rejected")
    check("reject dead-letters with reason",
          any(r["step"] == "review.reject" for r in eng.dead_letter.records))


# --- procedural proposals via review service -------------------------------
async def test_proposals_via_review():
    brand = load_brand()
    eng = build_orchestrator(brand, llm=SmartFakeLLM(), production_backend=MockProductionBackend())
    proc = eng.procedural
    await proc.seed(brand_id=brand.brand_id, agent="copy", prompt_text="seed prompt")
    on_voice = "dark editorial authoritative cinematic physical ai perspective never corporate always specific"
    off_voice = "buy now click here top 10 robot hacks number 7 will shock you"
    p_on = await proc.propose(brand_id=brand.brand_id, agent="copy", proposed_prompt=on_voice, performance_data={"ctr": 0.9})
    p_off = await proc.propose(brand_id=brand.brand_id, agent="copy", proposed_prompt=off_voice, performance_data={"ctr": 2.0})

    svc = ReviewService(review_store=eng.review_store, distribution=eng.distribution,
                        procedural=proc, dead_letter_sink=eng.dead_letter, x_http=None)
    views = await svc.proposals()
    check("proposals listed with similarity + threshold", len(views) == 2 and all(v.voice_threshold for v in views))

    blocked = await svc.approve_proposal(proposal_id=p_off.id, approved_by="reviewer")
    check("off-voice proposal blocked (voice regression)", blocked["status"] == "blocked_voice_regression")
    ok = await svc.approve_proposal(proposal_id=p_on.id, approved_by="reviewer")
    check("on-voice proposal approved -> new version", ok["status"] == "approved" and ok["new_version"] == 2)
    rb = await svc.rollback(agent="copy", brand_id=brand.brand_id)
    check("rollback returns to prior version", rb["status"] == "rolled_back" and rb["active_version"] == 1)


# --- full HTTP surface with fake LLM ---------------------------------------
def test_http_surface():
    from fastapi.testclient import TestClient
    import main
    from engine.core.assembly import build_orchestrator as real_build

    def fake_build(brand, **kw):
        return real_build(brand, llm=SmartFakeLLM(), production_backend=MockProductionBackend())

    with patch("engine.core.assembly.build_orchestrator", fake_build):
        with TestClient(main.app) as c:
            job = c.post("/jobs", json={"topic": "humanoid robots", "entity": "Figure"})
            check("POST /jobs 200 + staged", job.status_code == 200 and job.json()["status"] == "staged_for_review")
            q = c.get("/review/queue").json()
            check("GET /review/queue returns staged item", len(q) == 1)
            jid = q[0]["job_id"]
            detail = c.get(f"/review/job/{jid}").json()
            check("GET /review/job has attribution", "attribution" in detail and detail["attribution"]["research_chosen"])
            appr = c.post(f"/review/job/{jid}/approve").json()
            check("POST approve -> approved", appr["status"] == "approved")
            check("GET / serves dashboard html", "THE 10 AGENTS BUILT THIS" in c.get("/").text)


async def main_async() -> int:
    test_format_routing()
    await test_production_routing()
    await test_review_service()
    await test_proposals_via_review()
    test_http_surface()
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
