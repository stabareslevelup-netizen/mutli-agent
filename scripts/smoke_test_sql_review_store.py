"""
scripts/smoke_test_sql_review_store.py — one-shot live-Postgres round-trip
check for SqlReviewStore. Run only from
.github/workflows/smoke_test_db.yml (workflow_dispatch, manual trigger
only) -- never on push/PR, never part of the scheduled sweep.

SqlReviewStore (PR #8) has only ever been exercised against a fake
async-session double in tests/test_sql_review_store.py -- no live Postgres
instance was reachable from that development environment. This is the real
round-trip check PR #8's own description flagged as still owed.

Writes ONE throwaway ReviewItem with add(), confirms list_staged() and
get() both see exactly what was written, then flips its status off
staged_for_review so it never shows up in the real review dashboard queue
-- regardless of whether the check passed or failed.

Exits 0 on a clean round trip, 1 (with the specific mismatch printed) on
anything else.

    DATABASE_URL=... python -m scripts.smoke_test_sql_review_store
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

from engine.core.models import ContentPillar, PostFormat, ReviewItem


async def _run() -> int:
    if not os.getenv("DATABASE_URL"):
        print("FAIL: DATABASE_URL is not set -- nothing to smoke-test against.", file=sys.stderr)
        return 1

    from engine.core.database import get_sessionmaker, init_db
    from engine.core.review_store import SqlReviewStore

    await init_db()
    store = SqlReviewStore(get_sessionmaker())

    job_id = f"smoke-test-{uuid.uuid4().hex[:12]}"
    written = ReviewItem(
        job_id=job_id, brand_id="smoke_test", status="staged_for_review",
        pillar=ContentPillar.other, chosen_angle="smoke test round trip",
        main_post="this is a throwaway smoke-test row, safe to ignore/delete",
        format_used=PostFormat.pov_post.value,
        approval_note="smoke test -- not real content",
    )

    failures: list[str] = []
    await store.add(item=written)

    staged = await store.list_staged()
    if job_id not in {i.job_id for i in staged}:
        failures.append(f"list_staged() did not return {job_id} right after add()")

    got = await store.get(job_id)
    if got is None:
        failures.append(f"get({job_id}) returned None right after add()")
    else:
        for field in ("job_id", "brand_id", "status", "pillar", "chosen_angle",
                      "main_post", "format_used", "approval_note"):
            written_val, got_val = getattr(written, field), getattr(got.item, field)
            if written_val != got_val:
                failures.append(f"field '{field}' mismatch: wrote {written_val!r}, got {got_val!r}")

    # tidy up regardless of outcome -- take it off the live queue so it
    # never shows up in the real review dashboard.
    await store.set_status(job_id, "smoke_test_verified")

    if failures:
        print("FAIL: SqlReviewStore live-Postgres round trip did not match:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS: SqlReviewStore round trip OK against a real Postgres connection (job_id={job_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
