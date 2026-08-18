"""
Phase 5 unit tests — FREE (fake LLM, no API spend, no live DB).

Covers Copy (write_v2()) and Quality (evaluate_v2()) -- Phase 2, X-agent
migration. Same fake-transport pattern as test_phase4.py: FakeLLM returns
canned text, no real network/API calls anywhere.
"""
from __future__ import annotations

import asyncio
import json

from engine.agents.base import AgentContext
from engine.agents.copy_agent import CopyAgent
from engine.agents.quality import QualityAgent
from engine.core.cost_guard import CostGuard, InMemoryCostSink, MODEL_OPUS, MODEL_SONNET
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.llm import Usage
from engine.core.models import PostFormat, PostingSlot, StrategyOutput, TimingDecision
from engine.core.validation_gate import HandoffHalted, ValidationGate

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
        self.calls.append({"model": model, "user": user})
        return self._text, Usage(input_tokens=120, output_tokens=40)


def _ctx(llm, cost_sink, dl_sink):
    return AgentContext(
        llm=llm,
        cost_guard=CostGuard(daily_budget_usd=5.0, cost_sink=cost_sink),
        gate=ValidationGate(dead_letter_sink=dl_sink),
        brand_id="b",
    )


def _strategy(format=PostFormat.pov_post, thread_spine=None):
    return StrategyOutput(chosen_angle="Epirus wins $66M Army contract",
                          format=format, must_include=["$66M", "Epirus"],
                          must_avoid=["hype"], hashtags=["#DefenseTech"],
                          thread_spine=thread_spine)


def _timing(hedge=False):
    return TimingDecision(item_id="i1", recommended_slot=PostingSlot.slot_3pm,
                          urgency_note="note", citation_hedge_required=hedge)


# ===========================================================================
# Copy agent: write_v2()
# ===========================================================================
async def test_copy_happy_path():
    resp = json.dumps({"main_post": "Epirus just won a $66M Army contract for directed energy.",
                       "reply_link": "Source: https://sam.gov/opp/1",
                       "format_used": "pov_post", "hashtags_used": ["#DefenseTech"]})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.write_v2(strategy=_strategy(), timing=_timing(), job_id="j1")
    check("copy returns CopyOutput with main_post", out.main_post.startswith("Epirus"))
    check("copy carries reply_link", out.reply_link == "Source: https://sam.gov/opp/1")
    check("copy logged cost at Sonnet tier", cost.entries and cost.entries[0]["model"] == MODEL_SONNET)


async def test_copy_char_count_never_trusted():
    # LLM includes a bogus char_count -- must be ignored, real len() used instead
    from engine.agents.copy_agent import _SYSTEM as _COPY_SYSTEM
    text = "Epirus just won a $66M Army contract for directed energy systems."
    resp = json.dumps({"main_post": text, "reply_link": "Source: https://sam.gov/1",
                       "format_used": "pov_post", "char_count": 999999})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.write_v2(strategy=_strategy(), timing=_timing(), job_id="j1")
    check("char_count matches real len(), not the LLM's bogus 999999",
          out.char_count == len(text), f"got {out.char_count}, expected {len(text)}")
    check("prompt system text never requests char_count from the model",
          "char_count" not in _COPY_SYSTEM)
    check("the user turn (this call's actual content) never mentions char_count either",
          "char_count" not in agent.ctx.llm.calls[0]["user"])


async def test_copy_no_url_in_body_rejected():
    resp = json.dumps({"main_post": "Check the source at https://sam.gov/opp/1 for details.",
                       "reply_link": "Source: https://sam.gov/1", "format_used": "pov_post"})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    halted = False
    try:
        await agent.write_v2(strategy=_strategy(), timing=_timing(), job_id="j1")
    except HandoffHalted:
        halted = True
    check("URL in main_post -> HandoffHalted (structural rejection)", halted)
    check("URL rejection dead-lettered", len(dl.records) == 1)


async def test_copy_thread_tweet_over_280_rejected():
    long_tweet = "x" * 281
    spine = ["hook", "evidence", "context", "stakes"]
    resp = json.dumps({"main_post": "hook", "reply_link": "Source: https://sam.gov/1",
                       "format_used": "thread", "thread_tweets": ["hook", long_tweet]})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    halted = False
    try:
        await agent.write_v2(strategy=_strategy(format=PostFormat.thread, thread_spine=spine),
                             timing=_timing(), job_id="j1")
    except HandoffHalted:
        halted = True
    check("thread_tweets element over 280 chars -> HandoffHalted", halted)


async def test_copy_citation_hedged_from_code_not_llm():
    # LLM explicitly self-reports citation_hedged=False; timing says hedge IS required
    resp = json.dumps({"main_post": "Per the solicitation, DARPA is funding this.",
                       "reply_link": "Source: https://darpa.mil/1", "format_used": "pov_post",
                       "citation_hedged": False})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.write_v2(strategy=_strategy(), timing=_timing(hedge=True), job_id="j1")
    check("citation_hedged forced True from timing, overriding the LLM's self-reported False",
          out.citation_hedged is True)


async def test_copy_thread_main_post_normalized():
    spine = ["hook", "evidence", "context", "stakes"]
    resp = json.dumps({
        "main_post": "a completely different, wrong hook the LLM wrote",
        "reply_link": "Source: https://sam.gov/1", "format_used": "thread",
        "thread_tweets": ["the REAL first tweet / hook", "second tweet", "third tweet"]})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = CopyAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.write_v2(strategy=_strategy(format=PostFormat.thread, thread_spine=spine),
                               timing=_timing(), job_id="j1")
    check("main_post normalized to thread_tweets[0], not the LLM's mismatched main_post",
          out.main_post == "the REAL first tweet / hook", out.main_post)


async def test_copy_one_call_writes_whole_thread():
    spine = ["hook", "evidence", "context", "stakes", "prediction"]
    resp = json.dumps({
        "main_post": "hook", "reply_link": "Source: https://sam.gov/1", "format_used": "thread",
        "thread_tweets": ["hook", "t2", "t3", "t4", "t5"]})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    llm = FakeLLM(resp)
    agent = CopyAgent(_ctx(llm, cost, dl))
    out = await agent.write_v2(strategy=_strategy(format=PostFormat.thread, thread_spine=spine),
                               timing=_timing(), job_id="j1")
    check("exactly one LLM call writes the whole 5-tweet thread, not one per segment",
          len(llm.calls) == 1, f"got {len(llm.calls)} calls")
    check("all 5 thread tweets present in one shot", len(out.thread_tweets) == 5)


# ===========================================================================
# Quality agent: evaluate_v2()
# ===========================================================================
def _copy_output():
    from engine.core.models import CopyOutput
    return CopyOutput(main_post="Epirus just won a $66M Army contract for directed energy.",
                      reply_link="Source: https://sam.gov/1", format_used=PostFormat.pov_post)


def _scores(**overrides):
    base = {"source_specificity": 8, "differentiation": 8, "hook_strength": 8,
            "format_compliance": 8, "thesis_alignment": 8}
    base.update(overrides)
    return base


async def test_quality_happy_path():
    resp = json.dumps({"scores": _scores(), "notes": [], "approval_note": "Looks clean."})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = QualityAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.evaluate_v2(copy=_copy_output(), job_id="j1")
    check("quality all-8s -> pass verdict", out.verdict.value == "pass")
    check("quality total sums the 5 scores", out.total == 40, out.total)
    check("quality blocking_dimension is None on pass", out.blocking_dimension is None)
    check("quality logged cost at Opus tier", cost.entries and cost.entries[0]["model"] == MODEL_OPUS)


async def test_quality_verdict_never_trusted_from_llm():
    # LLM self-reports pass/40/None despite genuinely low scores (should compute to reject)
    low_scores = _scores(source_specificity=2)
    resp = json.dumps({"scores": low_scores, "verdict": "pass", "total": 999,
                       "blocking_dimension": None, "notes": [], "approval_note": "fine"})
    cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
    agent = QualityAgent(_ctx(FakeLLM(resp), cost, dl))
    out = await agent.evaluate_v2(copy=_copy_output(), job_id="j1")
    check("LLM's self-reported verdict='pass' is IGNORED -- real verdict is reject",
          out.verdict.value == "reject", out.verdict.value)
    check("LLM's self-reported total=999 is IGNORED -- real total computed from scores",
          out.total == sum(low_scores.values()), out.total)
    check("LLM's self-reported blocking_dimension=None is IGNORED -- real one is computed",
          out.blocking_dimension == "source_specificity", out.blocking_dimension)


async def test_quality_format_compliance_prompt_scopes_to_banned_words():
    # can't verify a live LLM's judgment without a real model call -- what IS
    # verifiable is that the prompt actually tells it to ignore already-
    # guaranteed checks and score only on banned words, per the design.
    from engine.agents.quality import _SYSTEM
    check("prompt tells the model char/URL/hashtag are pre-guaranteed",
          "already" in _SYSTEM.lower() and "guaranteed" in _SYSTEM.lower())
    check("prompt scopes format_compliance to banned words only",
          "banned words" in _SYSTEM.lower() or "revolutionary" in _SYSTEM.lower())


async def test_quality_verdict_thresholds():
    # confirmed against models.py's actual _compute_gate: lowest<5 -> reject,
    # 5<=lowest<7 -> revise, lowest>=7 -> pass (gated on the MIN dimension).
    cases = [(4, "reject"), (5, "revise"), (6, "revise"), (7, "pass")]
    for lowest, expected in cases:
        resp = json.dumps({"scores": _scores(hook_strength=lowest), "notes": [], "approval_note": "x"})
        cost, dl = InMemoryCostSink(), InMemoryDeadLetterSink()
        agent = QualityAgent(_ctx(FakeLLM(resp), cost, dl))
        out = await agent.evaluate_v2(copy=_copy_output(), job_id="j1")
        check(f"lowest dimension={lowest} -> verdict={expected}",
              out.verdict.value == expected, out.verdict.value)


async def main() -> int:
    await test_copy_happy_path()
    await test_copy_char_count_never_trusted()
    await test_copy_no_url_in_body_rejected()
    await test_copy_thread_tweet_over_280_rejected()
    await test_copy_citation_hedged_from_code_not_llm()
    await test_copy_thread_main_post_normalized()
    await test_copy_one_call_writes_whole_thread()
    await test_quality_happy_path()
    await test_quality_verdict_never_trusted_from_llm()
    await test_quality_format_compliance_prompt_scopes_to_banned_words()
    await test_quality_verdict_thresholds()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_phase5():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_phase5`) is
    unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "phase 5 suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
