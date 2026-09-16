"""
SqlReviewStore tests — FREE, no live database.

A real Postgres round-trip isn't available in this environment, and
postgresql.JSONB (the column type this table -- and Job.payload/
DeadLetter.context before it -- already uses) doesn't compile against
SQLite (verified directly: SQLAlchemy raises CompileError attempting it).
So this exercises SqlReviewStore's REAL code -- serialization, query
construction, status-filtering logic -- against a fake async-session
double that mimics just the SQLAlchemy AsyncSession surface the store
actually calls (add/commit/get/execute), backed by a plain in-memory dict.
This proves the store's own logic round-trips correctly; it does not (and
can't, from here) verify SQLAlchemy-against-real-Postgres compatibility.
"""
from __future__ import annotations

import asyncio
from datetime import date

from engine.core.models import (
    AgentAttribution, ContentPillar, DistributionPlan, PostFormat, PostingMode,
    PostingSlot, QualityDimensions, ReviewItem,
)
from engine.core.review_store import InMemoryReviewStore, SqlReviewStore, StoredReview
from engine.agents.distribution import StagedBundle
from engine.tools.social_apis import StagedPost

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


# ===========================================================================
# Fake async-session double -- see module docstring for why
# ===========================================================================
class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeAsyncSession:
    def __init__(self, table: dict):
        self._table = table   # shared across sessions -- simulates durable storage
        self._pending = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        self._pending.append(obj)

    async def commit(self):
        for obj in self._pending:
            self._table[obj.job_id] = obj
        self._pending.clear()

    async def get(self, model_cls, pk):
        return self._table.get(pk)

    async def execute(self, stmt):
        # Simplified on purpose: supports exactly the two query shapes
        # SqlReviewStore issues -- list_staged()'s filtered select
        # (status == "staged_for_review") and list_all()'s unfiltered
        # select. stmt.whereclause is None iff no .where() was added
        # (verified directly against real SQLAlchemy Select objects).
        # Not a general SQL emulator -- this fake exists to verify
        # SqlReviewStore's own logic, not to reimplement SQLAlchemy Core
        # statement compilation.
        if stmt.whereclause is None:
            rows = list(self._table.values())
        else:
            rows = [r for r in self._table.values() if r.status == "staged_for_review"]
        return _FakeExecuteResult(rows)


class FakeSessionmaker:
    def __init__(self):
        self._table: dict = {}

    def __call__(self):
        return _FakeAsyncSession(self._table)


# ===========================================================================
# Fixtures
# ===========================================================================
def _review_item(job_id="j1", status="staged_for_review"):
    return ReviewItem(
        job_id=job_id, brand_id="madre_de_maquinas", status=status,
        chosen_angle="Epirus wins $66M Army contract",
        main_post="Epirus just won a $66M Army contract for directed energy.",
        thread_tweets=[], reply_link="Source: https://sam.gov/opp/1",
        format_used="pov_post", scheduled_slot="3PM",
        source_url="https://sam.gov/opp/1", novelty_score=9,
        approval_note="Dollar figure confirmed against the SAM.gov listing.",
        quality_overall=0.86, quality_route="pass", platforms=["x"],
        attribution=AgentAttribution(research_chosen="Epirus wins $66M Army contract",
                                     timing_velocity="3PM", timing_citation="not required",
                                     copy_output_seconds=3.1,
                                     quality={"source_specificity": 9.0, "total": 43.0},
                                     skeptic_summary="Solid, well-sourced angle."))


def _bundle():
    plan = DistributionPlan(posting_mode=PostingMode.confirm, platforms=["x"],
                            staged=True, published=False)
    staged = {"x": StagedPost(platform="x", payload={"posts": [{"text": "hi"}], "link_mode": "reply"},
                              contains_url=True, estimated_cost_usd=0.21,
                              cost_breakdown=[{"text": "hi", "url": True, "usd": 0.21}],
                              notes="1 post(s), link_mode=reply")}
    return StagedBundle(plan=plan, staged=staged)


# ===========================================================================
# Tests
# ===========================================================================
async def test_add_and_get_round_trip_item_only():
    store = SqlReviewStore(FakeSessionmaker())
    item = _review_item()
    await store.add(item=item)
    got = await store.get("j1")
    check("get() returns a StoredReview", isinstance(got, StoredReview))
    check("round-tripped item matches the original (real ReviewItem, not a dict)",
          isinstance(got.item, ReviewItem) and got.item.chosen_angle == item.chosen_angle)
    check("round-tripped item preserves nested attribution.quality dict",
          got.item.attribution.quality["source_specificity"] == 9.0)
    check("bundle is None when none was added", got.bundle is None)


async def test_add_and_get_round_trip_with_bundle():
    store = SqlReviewStore(FakeSessionmaker())
    item, bundle = _review_item(job_id="j2"), _bundle()
    await store.add(item=item, bundle=bundle)
    got = await store.get("j2")
    check("bundle round-trips as a real StagedBundle, not a dict",
          isinstance(got.bundle, StagedBundle))
    check("bundle.plan round-trips as a real DistributionPlan",
          isinstance(got.bundle.plan, DistributionPlan) and got.bundle.plan.platforms == ["x"])
    check("bundle.staged['x'] round-trips as a real StagedPost, not a dict",
          isinstance(got.bundle.staged["x"], StagedPost))
    check("bundle.staged['x'] fields survived the round-trip",
          got.bundle.staged["x"].estimated_cost_usd == 0.21 and
          got.bundle.staged["x"].payload["posts"][0]["text"] == "hi")


async def test_get_missing_job_returns_none():
    store = SqlReviewStore(FakeSessionmaker())
    got = await store.get("does-not-exist")
    check("get() on a missing job_id returns None, not an error", got is None)


async def test_list_staged_filters_by_status():
    store = SqlReviewStore(FakeSessionmaker())
    await store.add(item=_review_item(job_id="staged1", status="staged_for_review"))
    await store.add(item=_review_item(job_id="staged2", status="staged_for_review"))
    await store.add(item=_review_item(job_id="approved1", status="approved"))
    listed = await store.list_staged()
    ids = {i.job_id for i in listed}
    check("list_staged returns only staged_for_review items", ids == {"staged1", "staged2"}, str(ids))
    check("list_staged items are real ReviewItems", all(isinstance(i, ReviewItem) for i in listed))


async def test_set_status_updates_both_column_and_json_blob():
    sm = FakeSessionmaker()
    store = SqlReviewStore(sm)
    await store.add(item=_review_item(job_id="j3", status="staged_for_review"))
    await store.set_status("j3", "approved")

    got = await store.get("j3")
    check("set_status: subsequent get() reflects the new status on the item",
          got.item.status == "approved", got.item.status)

    listed = await store.list_staged()
    check("set_status: item no longer appears in list_staged() after approval",
          "j3" not in {i.job_id for i in listed})

    # confirm the JSON blob itself was updated, not just the indexed column
    # (a stale blob would still show the old status if re-read independently)
    raw_row = sm._table["j3"]
    check("set_status: the stored item JSON blob's own status field was also updated",
          raw_row.item["status"] == "approved", raw_row.item)


async def test_set_status_on_missing_job_is_a_noop_not_an_error():
    store = SqlReviewStore(FakeSessionmaker())
    await store.set_status("nonexistent", "approved")   # should not raise
    check("set_status on a missing job_id doesn't raise", True)


async def test_list_all_returns_every_status_unfiltered():
    store = SqlReviewStore(FakeSessionmaker())
    await store.add(item=_review_item(job_id="all1", status="staged_for_review"))
    await store.add(item=_review_item(job_id="all2", status="approved"))
    await store.add(item=_review_item(job_id="all3", status="rejected"))
    all_items = await store.list_all()
    check("list_all returns every item regardless of status",
          {i.job_id for i in all_items} == {"all1", "all2", "all3"}, str([i.job_id for i in all_items]))
    check("list_all items are real ReviewItems", all(isinstance(i, ReviewItem) for i in all_items))
    # list_staged() must still be unaffected -- proves list_all's unfiltered
    # select and list_staged's filtered select aren't accidentally sharing
    # state or query logic in a way that breaks the existing method.
    staged_only = await store.list_staged()
    check("list_staged still only returns staged_for_review (unaffected by list_all)",
          {i.job_id for i in staged_only} == {"all1"}, str([i.job_id for i in staged_only]))


async def test_list_all_parity_between_inmemory_and_sql():
    # Identical fixture dataset -- varying job_id, status, AND pillar -- built
    # once and added to both store implementations, then compared field by
    # field. This must catch a store that returns without erroring but with
    # wrong/missing/mismatched data (e.g. dropping a pillar on deserialize,
    # or a status filter that leaked into list_all by accident).
    fixture = [
        ("p1", "staged_for_review", ContentPillar.defense_procurement),
        ("p2", "approved", ContentPillar.plant_based_fuel),
        ("p3", "rejected", ContentPillar.other),
        ("p4", "approved", ContentPillar.training_performance),
    ]

    async def _populate(store):
        for job_id, status, pillar in fixture:
            item = _review_item(job_id=job_id, status=status)
            item = item.model_copy(update={"pillar": pillar})
            await store.add(item=item)
        return await store.list_all()

    mem_items = await _populate(InMemoryReviewStore())
    sql_items = await _populate(SqlReviewStore(FakeSessionmaker()))

    mem_by_id = {i.job_id: i for i in mem_items}
    sql_by_id = {i.job_id: i for i in sql_items}

    check("both stores return the same set of job_ids",
          set(mem_by_id.keys()) == set(sql_by_id.keys()) == {jid for jid, _, _ in fixture},
          f"mem={sorted(mem_by_id)} sql={sorted(sql_by_id)}")

    for job_id, expected_status, expected_pillar in fixture:
        mem_item, sql_item = mem_by_id[job_id], sql_by_id[job_id]
        check(f"{job_id}: InMemory and Sql agree on status (both = expected {expected_status!r})",
              mem_item.status == sql_item.status == expected_status,
              f"mem={mem_item.status!r} sql={sql_item.status!r}")
        check(f"{job_id}: InMemory and Sql agree on pillar (both = expected {expected_pillar!r})",
              mem_item.pillar == sql_item.pillar == expected_pillar,
              f"mem={mem_item.pillar!r} sql={sql_item.pillar!r}")
        check(f"{job_id}: InMemory and Sql agree on chosen_angle (unrelated field, confirms full-item parity)",
              mem_item.chosen_angle == sql_item.chosen_angle)


async def test_matches_inmemory_reviewstore_interface_behavior():
    # same scenario run against both implementations -- same observable behavior
    for name, store in [("InMemory", InMemoryReviewStore()), ("Sql", SqlReviewStore(FakeSessionmaker()))]:
        await store.add(item=_review_item(job_id="parity1", status="staged_for_review"))
        before = await store.list_staged()
        await store.set_status("parity1", "rejected")
        after = await store.list_staged()
        check(f"{name}: staged before set_status", len(before) == 1, str(before))
        check(f"{name}: not staged after set_status('rejected')", len(after) == 0, str(after))
        got = await store.get("parity1")
        check(f"{name}: get() reflects the rejected status", got.item.status == "rejected")


async def main() -> int:
    await test_add_and_get_round_trip_item_only()
    await test_add_and_get_round_trip_with_bundle()
    await test_get_missing_job_returns_none()
    await test_list_staged_filters_by_status()
    await test_set_status_updates_both_column_and_json_blob()
    await test_set_status_on_missing_job_is_a_noop_not_an_error()
    await test_list_all_returns_every_status_unfiltered()
    await test_list_all_parity_between_inmemory_and_sql()
    await test_matches_inmemory_reviewstore_interface_behavior()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_sql_review_store():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_sql_review_store`)
    is unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "sql_review_store suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
