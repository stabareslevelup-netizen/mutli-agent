"""
Phase 5 unit tests — FREE (fake LLM / mock backends, no API spend, no network).

Covers Copy, Quality, Distribution (text-only pipeline: no Prompt Engineer /
Production / Higgsfield — removed when video generation was dropped from scope):
  - cost_guard tiering (Copy=Sonnet, Quality=Opus)
  - validation_gate on every handoff
  - Quality: text auto-eligible only at >=0.85
  - Distribution confirm-mode only; X cost link vs reply; IG/YT not live
    (both correctly excluded from staging when there's no media asset);
    auto/scheduled dormant; resilience + dead-letter on publish failure
"""
from __future__ import annotations

import asyncio
import os

from engine.agents.base import AgentContext
from engine.agents.copy_agent import CopyAgent
from engine.agents.distribution import DistributionAgent, PostingModeDisabled
from engine.agents.quality import QualityAgent
from engine.core.brand_loader import load_brand
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_OPUS, MODEL_SONNET
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.llm import Usage
from engine.core.models import CopyOutput, DistributionPlan, PostingMode, QualityRoute, StrategyPacket
from engine.core.validation_gate import HandoffHalted, ValidationGate
from engine.tools.social_apis import (
    InstagramAdapter, PublishBlocked, XAdapter, YouTubeAdapter,
)

os.environ["BRAND_CONFIG_PATH"] = "brands/madre_de_maquinas.yaml"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


class FakeLLM:
    def __init__(self, text):
        self._text = text
        self.calls = []

    async def complete(self, *, model, system, user, tools=None, max_tokens=2048, **kw):
        self.calls.append(model)
        return self._text, Usage(input_tokens=200, output_tokens=80)


def _ctx(llm, cost, dl, brand):
    return AgentContext(llm=llm,
                        cost_guard=CostGuard(daily_budget_usd=brand.budget.daily_usd, cost_sink=cost),
                        gate=ValidationGate(dead_letter_sink=dl), brand_id=brand.brand_id)


def _packet(brand):
    return StrategyPacket(chosen_angle="robots building robots", rationale="top",
                          formats=brand.formats, fusion_weights=brand.fusion_weights,
                          hard_constraints=[], inputs_digest={})


async def test_copy():
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    text = '{"x_thread":["t1","t2"],"ig_caption":"cap","youtube_script":"script"}'
    agent = CopyAgent(_ctx(FakeLLM(text), cost, dl, brand))
    out = await agent.run(packet=_packet(brand), brand=brand, job_id="j")
    check("copy returns CopyOutput", isinstance(out, CopyOutput) and len(out.x_thread) == 2)
    check("copy on Sonnet tier", cost.entries[0]["model"] == MODEL_SONNET)


async def test_quality_routing():
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    hi = '{"voice":0.9,"narrative":0.9,"format":0.9,"hook":0.9,"coherence":0.9,"reasons":["on voice"]}'
    content = CopyOutput(x_thread=["a"], ig_caption="b", youtube_script="c")

    q_text, auto_text = await QualityAgent(_ctx(FakeLLM(hi), cost, dl, brand)).evaluate(
        content=content, brand=brand, job_id="j")
    check("quality on Opus tier", cost.entries[0]["model"] == MODEL_OPUS)
    check("high text -> publish_queue", q_text.route == QualityRoute.publish_queue)
    check("high text -> auto eligible (>=0.85)", auto_text is True)

    lo = '{"voice":0.3,"narrative":0.3,"format":0.3,"hook":0.3,"coherence":0.3,"reasons":["off"]}'
    cost3 = InMemoryCostSink()
    q_lo, auto_lo = await QualityAgent(_ctx(FakeLLM(lo), cost3, dl, brand)).evaluate(
        content=content, brand=brand, job_id="j")
    check("low score -> reject", q_lo.route == QualityRoute.reject)
    check("low score -> not auto", auto_lo is False)


async def test_quality_malformed_halts():
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    halted = False
    try:
        await QualityAgent(_ctx(FakeLLM("no json here"), cost, dl, brand)).evaluate(
            content=CopyOutput(x_thread=["a"]), brand=brand, job_id="j")
    except HandoffHalted:
        halted = True
    check("malformed quality output halts + dead-letters", halted and len(dl.records) == 1)


# --- Distribution ----------------------------------------------------------
class FakeXHttp:
    def __init__(self, fail=False):
        self.fail = fail
        self.posted = None

    async def post_thread(self, posts):
        if self.fail:
            raise ConnectionError("x api down")
        self.posted = posts
        return {"ids": [f"x{i}" for i in range(len(posts))]}


async def test_distribution_confirm_only():
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    gate = ValidationGate(dl)
    adapters = [XAdapter(link_mode="reply"), InstagramAdapter(), YouTubeAdapter()]
    dist = DistributionAgent(adapters, CostGuard(daily_budget_usd=25.0, cost_sink=cost),
                             gate, brand.brand_id, dead_letter_sink=dl)
    # text-only content: no media asset -> IG and YouTube correctly can't stage
    # (both platforms require a media file to publish; that's a real constraint,
    # not a bug — see social_apis.py)
    content = {"x_thread": ["robots are here", "thread part 2"], "ig_caption": "cap",
               "youtube_script": "yt", "link": "https://madre.example/post"}

    bundle = await dist.stage(content=content, posting_mode=PostingMode.confirm, job_id="j")
    check("plan is confirm + staged + not published",
          bundle.plan.posting_mode == PostingMode.confirm and bundle.plan.staged and not bundle.plan.published)
    check("only X stages for text-only content (IG/YT need media)", set(bundle.staged) == {"x"})
    check("IG/YT dead-lettered as no-media, not crashed",
          any("media" in r["error"].lower() or "video" in r["error"].lower() for r in dl.records))

    # X cost: reply mode -> 2 cheap text posts ($0.01) + 1 link reply ($0.20) = $0.22
    x_cost = bundle.staged["x"].estimated_cost_usd
    check("X reply-mode cost itemized (~$0.22)", abs(x_cost - 0.22) < 1e-6, f"x_cost={x_cost}")
    x_entries = [e for e in cost.entries if e["model"] == "x-api"]
    check("X per-post cost logged via cost_guard", x_entries and abs(x_entries[0]["usd"] - 0.22) < 1e-6)

    # inline mode puts the URL in the main post (one $0.20 post)
    inline = XAdapter(link_mode="inline").stage(content)
    check("inline mode -> URL in main post", inline.contains_url and inline.cost_breakdown[0]["url"])

    # confirm gate: nothing publishes without explicit human confirm
    blocked = False
    try:
        await dist.confirm_publish(bundle=bundle, platform="x", http=FakeXHttp(),
                                   confirmed=False, job_id="j")
    except PublishBlocked:
        blocked = True
    check("publish blocked without human confirm", blocked)

    # X live + confirmed -> publishes
    ok = await dist.confirm_publish(bundle=bundle, platform="x", http=FakeXHttp(),
                                    confirmed=True, job_id="j")
    check("X publishes after confirm", ok.published is True)

    # X failure -> resilience + dead-letter, no crash
    pre = len(dl.records)
    fail = await dist.confirm_publish(bundle=bundle, platform="x", http=FakeXHttp(fail=True),
                                      confirmed=True, job_id="j")
    check("X publish failure handled (not published)", fail.published is False)
    check("publish failure dead-lettered", len(dl.records) > pre)


async def test_distribution_with_media_stages_ig():
    # if an asset URL IS present (e.g. a manually-added image), IG can stage —
    # proves the platform rule is about media presence, not a hardcoded skip
    brand = load_brand()
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    gate = ValidationGate(dl)
    dist = DistributionAgent([InstagramAdapter()], CostGuard(daily_budget_usd=25.0, cost_sink=cost),
                             gate, brand.brand_id, dead_letter_sink=dl)
    content = {"ig_caption": "cap", "media_url": "https://cdn.example/x.jpg"}
    bundle = await dist.stage(content=content, posting_mode=PostingMode.confirm, job_id="j")
    check("IG stages when media_url is present", "instagram" in bundle.staged)


async def test_auto_mode_dormant():
    brand = load_brand()
    dist = DistributionAgent([XAdapter()], CostGuard(daily_budget_usd=25.0),
                             ValidationGate(), brand.brand_id)
    disabled = False
    try:
        await dist.stage(content={"x_thread": ["x"]}, posting_mode=PostingMode.auto, job_id="j")
    except PostingModeDisabled:
        disabled = True
    check("auto/scheduled mode is dormant in v1", disabled)


async def main() -> int:
    await test_copy()
    await test_quality_routing()
    await test_quality_malformed_halts()
    await test_distribution_confirm_only()
    await test_distribution_with_media_stages_ig()
    await test_auto_mode_dormant()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
