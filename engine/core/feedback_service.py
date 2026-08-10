"""
engine/core/feedback_service.py — Phase 9 feedback loop.

Closes the system: real post engagement -> memory learns -> patterns surface as
guarded procedural proposals for human approval.

  - ingest(job_id, engagement): accepts per-platform metrics (views/likes/
    shares/saves/replies). Works with MANUAL input too (paste numbers after
    posting by hand) — no live API needed.
  - episodic memory auto-updates with the engagement (safe, no approval).
  - semantic memory auto-updates with the performance signal — which pillar /
    velocity performed (safe, no approval).
  - when a pattern holds across >= min_jobs jobs (e.g. incident_file
    consistently outperforms company_intel), a procedural PROPOSAL is surfaced
    to the dashboard. NEVER auto-applies — a human approves/rejects via the
    proposals UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# engagement weights: shares/saves/replies are worth more than a view
_WEIGHTS = {"views": 0.01, "likes": 1.0, "shares": 3.0, "saves": 2.0, "replies": 2.0}


def engagement_score(engagement: dict[str, dict[str, float]], weights=_WEIGHTS) -> float:
    total = 0.0
    for _platform, metrics in (engagement or {}).items():
        for k, w in weights.items():
            total += float(metrics.get(k, 0) or 0) * w
    return round(total, 3)


@dataclass
class FeedbackRecord:
    job_id: str
    pillar: str
    velocity: str
    angle: str
    score: float


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self.records: list[FeedbackRecord] = []
        self.proposed_combos: set[str] = set()   # dedupe procedural proposals (by pillar)

    def add(self, rec: FeedbackRecord) -> None:
        self.records.append(rec)


class FeedbackService:
    def __init__(self, *, review_store, episodic, semantic, procedural, brand_id: str,
                 feedback_store: Optional[InMemoryFeedbackStore] = None,
                 min_jobs: int = 5, outperform_ratio: float = 1.2):
        self._reviews = review_store
        self._epi = episodic
        self._sem = semantic
        self._proc = procedural
        self._brand_id = brand_id
        self._store = feedback_store or InMemoryFeedbackStore()
        self._min_jobs = min_jobs
        self._ratio = outperform_ratio

    async def ingest(self, *, job_id: str, engagement: dict[str, dict[str, float]]) -> dict:
        score = engagement_score(engagement)

        # correlate with the job's metadata (pillar / velocity / angle)
        pillar = velocity = ""
        angle = job_id
        stored = await self._reviews.get(job_id) if self._reviews else None
        if stored is not None:
            it = stored.item
            pillar, angle = it.pillar_id, it.chosen_angle
            velocity = it.attribution.timing_velocity

        # episodic auto-update (safe) — what happened for THIS job
        await self._epi.record(
            brand_id=self._brand_id,
            content=f"job {job_id}: '{angle[:80]}' [{pillar}] engagement_score={score}",
            meta={"job_id": job_id, "engagement": engagement, "score": score,
                  "pillar": pillar, "velocity": velocity},
            job_id=job_id)

        # semantic auto-update (safe) — the generalizable performance signal
        await self._sem.record(
            brand_id=self._brand_id,
            content=f"performance: pillar={pillar} velocity={velocity} -> score {score}",
            meta={"pillar": pillar, "velocity": velocity, "score": score})

        self._store.add(FeedbackRecord(job_id, pillar, velocity, angle, score))
        proposal = await self._maybe_propose()

        return {"job_id": job_id, "engagement_score": score, "pillar": pillar,
                "episodic_updated": True, "semantic_updated": True,
                "proposal_surfaced": proposal}

    async def _maybe_propose(self) -> Optional[dict]:
        # group by pillar; need enough jobs to trust the pattern
        groups: dict[str, list[float]] = {}
        for r in self._store.records:
            if r.pillar:
                groups.setdefault(r.pillar, []).append(r.score)
        ranked = sorted(
            [(k, sum(v) / len(v), len(v)) for k, v in groups.items()],
            key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return None
        (best_pillar, best_avg, best_n) = ranked[0]
        (worst_pillar, worst_avg, _wn) = ranked[-1]
        # only propose on a strong, well-supported gap, and only once per pillar
        if best_n < self._min_jobs or worst_avg <= 0 or best_avg < worst_avg * self._ratio:
            return None
        if best_pillar in self._store.proposed_combos:
            return None
        self._store.proposed_combos.add(best_pillar)

        pct = round((best_avg / worst_avg - 1) * 100)
        instruction = (
            f"Performance signal: {best_pillar} content outperforms "
            f"{worst_pillar} by ~{pct}% across {best_n} jobs. "
            f"Bias Strategy toward {best_pillar} for comparable topics.")
        perf = {"best": {"pillar": best_pillar, "avg_score": round(best_avg, 2), "jobs": best_n},
                "worst": {"pillar": worst_pillar, "avg_score": round(worst_avg, 2)}}
        proposal = await self._proc.propose(
            brand_id=self._brand_id, agent="strategy",
            proposed_prompt=instruction, performance_data=perf)
        return {"proposal_id": proposal.id, "agent": "strategy",
                "voice_similarity": proposal.voice_similarity, "performance": perf}
