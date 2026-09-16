"""
Content-pillar measurement tests — FREE (fakes, no spend).

Covers the new per-draft pillar tag end to end: the ContentPillar enum,
StrategyOutput.pillar (LLM-assigned, defaults to "other" for backward
compat), SkepticOutput.pillar_flag (code-computed from strategy.pillar,
never LLM self-report -- same philosophy as narrative_conflict_flag),
ReviewItem.pillar (populated by Orchestrator._build_review_item(), backward
compatible with rows/fixtures that predate this field), ReviewStore.list_all()
on both the Protocol and InMemoryReviewStore, and build_pillar_report().

Does NOT cover "heavily edited before approval" -- that instrumentation
(edit-capture) is a separate, later change; this report only covers
generated/approved/rejected, which the review store can already tell us.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from engine.agents.base import AgentContext
from engine.agents.orchestrator import Orchestrator
from engine.agents.skeptic import SkepticAgent
from engine.core.cost_guard import CostGuard, InMemoryCostSink
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.llm import Usage
from engine.core.brand_loader import load_brand
from engine.core.models import (
    ContentPillar, CopyOutput, MemoryQueryResult, PostFormat, PostingSlot,
    QualityDimensions, QualityOutput, ResearchItem, ReviewItem, SkepticOutput,
    SkepticVerdict, SourceKind, StrategyOutput, TimingDecision,
)
from engine.core.pillar_report import build_pillar_report
from engine.core.review_store import InMemoryReviewStore, ReviewStore
from engine.core.validation_gate import ValidationGate

import os
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


def _item():
    return ResearchItem(item_id="i1", source=SourceKind.sam_gov, date_found=date.today(),
                        headline="Epirus wins $66M Army contract", raw_detail="180 employees, directed energy",
                        novelty_score=9, post_angle="post-Ukraine doctrine in procurement form",
                        source_url="https://sam.gov/opp/1")


def _timing():
    return TimingDecision(item_id="i1", recommended_slot=PostingSlot.slot_3pm,
                          urgency_note="note", citation_hedge_required=False)


def _strategy_out(pillar=ContentPillar.defense_procurement):
    return StrategyOutput(chosen_angle="Epirus wins $66M Army contract",
                          format=PostFormat.pov_post, must_include=["$66M", "Epirus"],
                          pillar=pillar)


def _copy_out():
    return CopyOutput(main_post="Epirus just won a $66M Army contract for directed energy.",
                      reply_link="https://sam.gov/opp/1", format_used=PostFormat.pov_post)


def _quality_out():
    return QualityOutput(scores=QualityDimensions(source_specificity=9, differentiation=8,
                         hook_strength=8, format_compliance=9, thesis_alignment=9),
                         approval_note="Dollar figure confirmed against the SAM.gov listing.")


# ===========================================================================
# ContentPillar enum
# ===========================================================================
def test_content_pillar_enum_values():
    expected = {"physical_ai_readiness", "defense_procurement", "training_performance",
               "plant_based_fuel", "building_with_ai", "other"}
    check("ContentPillar has exactly the six spec'd values",
          {p.value for p in ContentPillar} == expected, str({p.value for p in ContentPillar}))


# ===========================================================================
# StrategyOutput.pillar
# ===========================================================================
def test_strategy_output_pillar_default_and_explicit():
    out = StrategyOutput(chosen_angle="a", format=PostFormat.pov_post, must_include=["f"])
    check("StrategyOutput without a pillar kwarg defaults to other (backward compat)",
          out.pillar == ContentPillar.other)

    out2 = StrategyOutput(chosen_angle="a", format=PostFormat.pov_post, must_include=["f"],
                          pillar=ContentPillar.training_performance)
    check("StrategyOutput accepts an explicit valid pillar",
          out2.pillar == ContentPillar.training_performance)

    raised = False
    try:
        StrategyOutput(chosen_angle="a", format=PostFormat.pov_post, must_include=["f"],
                       pillar="not_a_real_pillar")
    except Exception:
        raised = True
    check("StrategyOutput rejects an invalid pillar string", raised)


async def test_strategy_prompt_lists_the_pillars():
    from engine.agents.strategy import StrategyAgent
    resp = '{"chosen_angle":"a","format":"pov_post","must_include":["f"],"pillar":"defense_procurement"}'
    llm = CapturingLLM(resp)
    strat = StrategyAgent(_ctx(llm))
    out = await strat.decide_v2(item=_item(), timing=_timing(), memory=MemoryQueryResult(narrative=[]), job_id="j")
    check("Strategy's prompt lists all five real pillars for the LLM to choose from",
          all(p.value in llm.last_system for p in ContentPillar if p != ContentPillar.other))
    check("LLM-assigned pillar comes through on the output", out.pillar == ContentPillar.defense_procurement)


# ===========================================================================
# SkepticOutput.pillar_flag -- code-computed, both directions
# ===========================================================================
async def test_skeptic_pillar_flag_set_true_when_strategy_pillar_is_other():
    llm = CapturingLLM('{"verdict":"approved","critique":"fine"}')
    out = await SkepticAgent(_ctx(llm)).review_v2(
        item=_item(), strategy=_strategy_out(pillar=ContentPillar.other), job_id="j")
    check("pillar_flag True when Strategy tagged the draft 'other'", out.pillar_flag is True)
    check("pillar_flag does not change the verdict (measurement only)",
          out.verdict == SkepticVerdict.approved)


async def test_skeptic_pillar_flag_false_for_a_real_pillar():
    llm = CapturingLLM('{"verdict":"approved","critique":"fine"}')
    out = await SkepticAgent(_ctx(llm)).review_v2(
        item=_item(), strategy=_strategy_out(pillar=ContentPillar.building_with_ai), job_id="j")
    check("pillar_flag False for a real (non-other) pillar", out.pillar_flag is False)


async def test_skeptic_pillar_flag_self_report_overridden_both_directions():
    # ADVERSARIAL, mirroring test_phase9's narrative_conflict self-report test:
    # the LLM tries to self-report pillar_flag directly; code must ignore it
    # and compute the real value from strategy.pillar every time.
    llm_lie_true = CapturingLLM('{"verdict":"approved","critique":"fine","pillar_flag":true}')
    out1 = await SkepticAgent(_ctx(llm_lie_true)).review_v2(
        item=_item(), strategy=_strategy_out(pillar=ContentPillar.building_with_ai), job_id="j")
    check("LLM self-reporting pillar_flag=true is overridden to False for a real pillar",
          out1.pillar_flag is False)

    llm_lie_false = CapturingLLM('{"verdict":"approved","critique":"fine","pillar_flag":false}')
    out2 = await SkepticAgent(_ctx(llm_lie_false)).review_v2(
        item=_item(), strategy=_strategy_out(pillar=ContentPillar.other), job_id="j")
    check("LLM self-reporting pillar_flag=false is overridden to True when pillar is 'other'",
          out2.pillar_flag is True)


# ===========================================================================
# ReviewItem.pillar -- backward compat + Orchestrator wiring
# ===========================================================================
def test_review_item_pillar_default_backward_compat():
    item = ReviewItem(job_id="j", brand_id="b", status="staged_for_review")
    check("ReviewItem without a pillar kwarg defaults to other", item.pillar == ContentPillar.other)

    dumped = item.model_dump(mode="json")
    restored = ReviewItem.model_validate(dumped)
    check("pillar round-trips through model_dump/model_validate",
          restored.pillar == ContentPillar.other)

    # simulates an old JSONB blob written before this field existed
    old_blob = {"job_id": "j", "brand_id": "b", "status": "approved"}
    old_blob.pop("pillar", None)
    restored_old = ReviewItem.model_validate(old_blob)
    check("a pre-existing JSON blob with no 'pillar' key still validates (defaults to other)",
          restored_old.pillar == ContentPillar.other)


def test_orchestrator_build_review_item_carries_strategy_pillar():
    orch = Orchestrator(agents=None, job_store=None, cost_guard=None, brand=load_brand())
    bundle = SimpleNamespace(plan=SimpleNamespace(platforms=["x"]))
    item = orch._build_review_item(
        "j1", _item(), MemoryQueryResult(), _strategy_out(pillar=ContentPillar.plant_based_fuel),
        SkepticOutput(verdict=SkepticVerdict.approved, critique="fine"), _timing(),
        _copy_out(), _quality_out(), bundle, 1.2)
    check("ReviewItem returned is the real model", isinstance(item, ReviewItem))
    check("ReviewItem.pillar carries Strategy's actual pillar, not a default",
          item.pillar == ContentPillar.plant_based_fuel)
    check("legacy pillar_id is left untouched (still empty)", item.pillar_id == "")


# ===========================================================================
# ReviewStore.list_all()
# ===========================================================================
async def test_list_all_returns_every_status_inmemory():
    store = InMemoryReviewStore()
    check("InMemoryReviewStore satisfies the ReviewStore protocol (list_all included)",
          isinstance(store, ReviewStore))
    await store.add(item=ReviewItem(job_id="a", brand_id="b", status="staged_for_review"))
    await store.add(item=ReviewItem(job_id="b", brand_id="b", status="approved"))
    await store.add(item=ReviewItem(job_id="c", brand_id="b", status="rejected"))
    all_items = await store.list_all()
    check("list_all returns every item regardless of status",
          {i.job_id for i in all_items} == {"a", "b", "c"}, str([i.job_id for i in all_items]))
    staged_only = await store.list_staged()
    check("list_staged still only returns staged_for_review (unaffected by list_all)",
          {i.job_id for i in staged_only} == {"a"})


# ===========================================================================
# build_pillar_report()
# ===========================================================================
def _mk(job_id, pillar, status, created_at=None):
    return ReviewItem(job_id=job_id, brand_id="b", status=status, pillar=pillar,
                      created_at=created_at or datetime.now(timezone.utc))


async def test_build_pillar_report_counts_by_pillar_and_status():
    store = InMemoryReviewStore()
    for it in [
        _mk("d1", ContentPillar.defense_procurement, "staged_for_review"),
        _mk("d2", ContentPillar.defense_procurement, "approved"),
        _mk("d3", ContentPillar.defense_procurement, "approved"),
        _mk("d4", ContentPillar.defense_procurement, "rejected"),
        _mk("p1", ContentPillar.plant_based_fuel, "rejected"),
        _mk("p2", ContentPillar.plant_based_fuel, "rejected"),
        _mk("o1", ContentPillar.other, "rejected"),
    ]:
        await store.add(item=it)

    report = await build_pillar_report(store)
    check("report has a row for every ContentPillar, even zero-draft ones",
          set(report.keys()) == {p.value for p in ContentPillar}, str(report.keys()))

    dp = report["defense_procurement"]
    check("defense_procurement: generated counts all 4 drafts", dp.generated == 4, str(dp))
    check("defense_procurement: approved counts only the approved ones", dp.approved == 2, str(dp))
    check("defense_procurement: rejected counts only the rejected one", dp.rejected == 1, str(dp))

    pbf = report["plant_based_fuel"]
    check("plant_based_fuel: 2 generated, 0 approved, 2 rejected -- the boredom signal",
          (pbf.generated, pbf.approved, pbf.rejected) == (2, 0, 2), str(pbf))

    empty_pillar = report["training_performance"]
    check("a pillar with zero drafts is a real zero row, not omitted",
          (empty_pillar.generated, empty_pillar.approved, empty_pillar.rejected) == (0, 0, 0))

    other = report["other"]
    check("'other' is tracked like any real pillar", other.generated == 1 and other.rejected == 1)


async def test_build_pillar_report_since_filters_by_created_at():
    store = InMemoryReviewStore()
    now = datetime.now(timezone.utc)
    await store.add(item=_mk("old", ContentPillar.building_with_ai, "approved", now - timedelta(days=30)))
    await store.add(item=_mk("new", ContentPillar.building_with_ai, "approved", now - timedelta(hours=1)))

    all_time = await build_pillar_report(store)
    check("since=None (all-time) counts both drafts",
          all_time["building_with_ai"].generated == 2)

    weekly = await build_pillar_report(store, since=now - timedelta(days=7))
    check("since=7 days excludes the 30-day-old draft",
          weekly["building_with_ai"].generated == 1, str(weekly["building_with_ai"]))


async def main_async() -> int:
    test_content_pillar_enum_values()
    test_strategy_output_pillar_default_and_explicit()
    await test_strategy_prompt_lists_the_pillars()
    await test_skeptic_pillar_flag_set_true_when_strategy_pillar_is_other()
    await test_skeptic_pillar_flag_false_for_a_real_pillar()
    await test_skeptic_pillar_flag_self_report_overridden_both_directions()
    test_review_item_pillar_default_backward_compat()
    test_orchestrator_build_review_item_carries_strategy_pillar()
    await test_list_all_returns_every_status_inmemory()
    await test_build_pillar_report_counts_by_pillar_and_status()
    await test_build_pillar_report_since_filters_by_created_at()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_pillar():
    """pytest entrypoint — runs the suite above and asserts real success.
    Direct-run entrypoint (`python -m tests.test_pillar`) is unaffected below."""
    import asyncio
    assert asyncio.run(main_async()) == 0, "pillar suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
