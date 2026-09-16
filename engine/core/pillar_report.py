"""
engine/core/pillar_report.py — weekly per-pillar report: drafts generated /
approved / rejected, so a niche gets picked by data (which pillars actually
get approved vs. rejected) instead of by what sounds strategic.

Reads the review store directly; no new state of its own. Shared by
scripts/pillar_report.py (CLI) and main.py's GET /review/pillar_report so
there is exactly one place that defines what a pillar's numbers mean.

NOT included yet: "heavily edited before approval". Nothing in the review
flow captures an edit today (approve/reject are the only actions) -- that
instrumentation is a separate, later change; this report only covers what
the review store can already tell us.

Data only accumulates once PR #8's persistent SqlReviewStore is merged and a
scheduled sweep is actually running -- until then whatever process built
this report only sees its own in-memory queue.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from engine.core.models import ContentPillar
from engine.core.review_store import ReviewStore


@dataclass
class PillarStats:
    pillar: str
    generated: int = 0
    approved: int = 0
    rejected: int = 0


async def build_pillar_report(review_store: ReviewStore, *,
                              since: Optional[datetime] = None) -> dict[str, PillarStats]:
    """One row per ContentPillar, including pillars with zero drafts -- a
    pillar nobody generated anything for this week is itself a data point,
    not something to omit. `since` filters by ReviewItem.created_at; None
    means all-time."""
    stats: dict[str, PillarStats] = {p.value: PillarStats(pillar=p.value) for p in ContentPillar}
    for item in await review_store.list_all():
        if since is not None and item.created_at < since:
            continue
        key = item.pillar.value
        s = stats.setdefault(key, PillarStats(pillar=key))
        s.generated += 1
        if item.status == "approved":
            s.approved += 1
        elif item.status == "rejected":
            s.rejected += 1
    return stats
