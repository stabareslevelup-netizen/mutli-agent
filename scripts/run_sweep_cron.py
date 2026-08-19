"""
scripts/run_sweep_cron.py — one-shot Orchestrator.run_sweep() invocation for
a scheduled trigger (see .github/workflows/run_sweep.yml).

This is new code, not a revival of the deleted engine/triggers/calendar_cron.py
-- that file was built entirely around single-topic firing, a premise Phase 2
retired. run_sweep() takes zero topic input by design.

Wires the same Sql-backed stores main.py wires when DATABASE_URL is set, so
whatever this staging run produces is visible to a human via the long-lived
dashboard server -- a separate OS process with no shared memory otherwise.
Without DATABASE_URL, this silently falls back to InMemory stores that vanish
the moment this process exits, i.e. a no-op from the dashboard's point of
view. DATABASE_URL is required for a scheduled run to actually matter.

    python -m scripts.run_sweep_cron
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from engine.core.assembly import build_orchestrator   # noqa: E402
from engine.core.brand_loader import load_brand       # noqa: E402

# Every status _process_item() can return a JobResult with (grepped directly
# from orchestrator.py, not guessed) -- printed in this fixed order so the
# Actions log summary is stable run to run; anything unrecognized still
# prints, just after this list.
_KNOWN_STATUSES = (
    "staged_for_review", "narrative_conflict", "quality_rejected",
    "quality_revise_exhausted", "skeptic_rejected", "halted_budget", "failed",
)


async def _run() -> int:
    brand = load_brand()

    job_store = dead_letter_sink = review_store = None   # InMemory defaults inside build_orchestrator()
    if os.getenv("DATABASE_URL"):
        from engine.core.database import get_sessionmaker, init_db
        await init_db()
        from engine.core.dead_letter import SqlDeadLetterSink
        from engine.core.job_manager import SqlJobStore
        from engine.core.review_store import SqlReviewStore
        sm = get_sessionmaker()
        job_store = SqlJobStore(sm)
        dead_letter_sink = SqlDeadLetterSink(sm)
        review_store = SqlReviewStore(sm)
    else:
        print("WARNING: DATABASE_URL not set -- running against in-memory stores. "
              "Anything staged in this run vanishes when this process exits and "
              "will NOT appear on the dashboard.", file=sys.stderr)

    eng = build_orchestrator(brand, job_store=job_store,
                             dead_letter_sink=dead_letter_sink, review_store=review_store)

    try:
        results = await eng.orchestrator.run_sweep()
    except Exception as exc:
        # _process_item() already catches its own per-item failures (returns
        # a JobResult with status="failed" and dead-letters itself -- see
        # orchestrator.py's defensive `except Exception` around the per-job
        # try block) -- so an exception reaching here can only come from
        # research.sweep() or timing.assign(), the two top-level calls
        # run_sweep() makes before the per-item loop even starts. Record it
        # with full context, then re-raise: a failed sweep should make this
        # Actions run go red, not exit 0 having silently done nothing.
        await eng.dead_letter.record(
            job_id=None, step="run_sweep",
            error=f"{type(exc).__name__}: {exc}",
            context={"brand_id": brand.brand_id})
        raise

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print(f"run_sweep complete: {len(results)} item(s) processed")
    for status in _KNOWN_STATUSES:
        if status in counts:
            print(f"  {status}: {counts[status]}")
    for status, n in counts.items():
        if status not in _KNOWN_STATUSES:
            print(f"  {status}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
