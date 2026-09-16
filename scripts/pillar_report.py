"""
scripts/pillar_report.py — weekly per-pillar report (drafts generated /
approved / rejected) for deciding a niche by data instead of by what sounds
strategic. See engine/core/pillar_report.py for what the numbers mean.

Uses whatever ReviewStore build_orchestrator() wires up -- today that's
always a fresh InMemoryReviewStore, so this only reflects whatever this one
process staged (usually nothing). It becomes a real report once PR #8's
persistent SqlReviewStore merges, build_orchestrator() gets a review_store
override wired the same way scripts/run_sweep_cron.py wires one, and a
scheduled sweep is actually running.

    python -m scripts.pillar_report [--days N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from engine.core.assembly import build_orchestrator          # noqa: E402
from engine.core.brand_loader import load_brand               # noqa: E402
from engine.core.pillar_report import build_pillar_report     # noqa: E402


async def _run(days: int) -> int:
    brand = load_brand()
    eng = build_orchestrator(brand)

    if not os.getenv("DATABASE_URL"):
        print("WARNING: no persistent review store wired in yet (needs PR #8's "
              "SqlReviewStore, a review_store override on build_orchestrator, and a "
              "scheduled sweep actually running) -- this report only reflects whatever "
              "this process staged, which is generally nothing.", file=sys.stderr)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    stats = await build_pillar_report(eng.review_store, since=since)

    header = f"{'PILLAR':<28}{'GENERATED':>10}{'APPROVED':>10}{'REJECTED':>10}"
    print(f"Pillar report -- trailing {days} day(s)\n")
    print(header)
    print("-" * len(header))
    for s in sorted(stats.values(), key=lambda s: s.generated, reverse=True):
        print(f"{s.pillar:<28}{s.generated:>10}{s.approved:>10}{s.rejected:>10}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args.days)))
