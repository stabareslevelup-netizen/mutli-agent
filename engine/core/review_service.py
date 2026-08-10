"""
engine/core/review_service.py — backend for the review dashboard.

Wraps the review store + distribution (one-tap approve -> confirm publish),
the dead-letter (one-tap reject), and the guarded ProceduralMemory (proposal
approve/reject/rollback). Confirm-mode stays intact: approve is the human
confirm; nothing publishes without it.
"""
from __future__ import annotations

from typing import Any, Optional

from engine.agents.distribution import DistributionAgent
from engine.core.models import ProposalView, ReviewItem
from engine.core.review_store import ReviewStore
from engine.memory.procedural import (
    ApprovalRequired, NoVersionToRollback, ProceduralMemory, ProposalNotFound,
    VoiceRegressionBlocked,
)
from engine.tools.social_apis import PublishBlocked


class ReviewService:
    def __init__(self, *, review_store: ReviewStore, distribution: DistributionAgent,
                 procedural: ProceduralMemory, dead_letter_sink: Any,
                 x_http: Any = None):
        self._store = review_store
        self._dist = distribution
        self._proc = procedural
        self._dl = dead_letter_sink
        self._x_http = x_http   # live X client; None => publish pending credentials

    # --- content review queue ----------------------------------------------
    async def queue(self) -> list[ReviewItem]:
        return await self._store.list_staged()

    async def get(self, job_id: str) -> Optional[ReviewItem]:
        s = await self._store.get(job_id)
        return s.item if s else None

    async def approve(self, *, job_id: str, confirmed: bool = True) -> dict:
        """One-tap approve = the human confirm. Publishes live platforms when a
        live client is configured; otherwise marks approved/publish-pending."""
        if not confirmed:
            return {"status": "blocked", "reason": "approval requires explicit confirm"}
        s = await self._store.get(job_id)
        if s is None:
            return {"status": "not_found", "job_id": job_id}

        if self._x_http is None:
            await self._store.set_status(job_id, "approved")
            return {"status": "approved", "published": [],
                    "note": "confirmed; publish pending live platform credentials",
                    "platforms": s.item.platforms}

        published, errors = [], []
        for plat in s.item.platforms:
            try:
                r = await self._dist.confirm_publish(
                    bundle=s.bundle, platform=plat, http=self._x_http,
                    confirmed=True, job_id=job_id)
                (published if r.published else errors).append(plat)
            except PublishBlocked as exc:
                errors.append(f"{plat}: {exc}")
        await self._store.set_status(job_id, "approved")
        return {"status": "approved", "published": published, "errors": errors}

    async def reject(self, *, job_id: str, reason: str) -> dict:
        if self._dl is not None:
            await self._dl.record(job_id=job_id, step="review.reject", error=reason,
                                  context={"action": "reject"})
        await self._store.set_status(job_id, "rejected")
        return {"status": "rejected", "job_id": job_id, "reason": reason}

    # --- procedural proposals (guarded self-improvement) -------------------
    async def proposals(self) -> list[ProposalView]:
        out = []
        for p in await self._proc.pending_proposals():
            out.append(ProposalView(
                proposal_id=p.id, agent_name=p.agent, current_version=p.current_version,
                proposed_prompt=p.proposed_prompt, performance_data=p.performance_data,
                voice_similarity=p.voice_similarity, voice_threshold=self._proc.voice_threshold,
                status=p.status))
        return out

    async def approve_proposal(self, *, proposal_id: int, approved_by: str) -> dict:
        try:
            rec = await self._proc.approve(proposal_id=proposal_id, approved_by=approved_by)
        except VoiceRegressionBlocked as exc:
            return {"status": "blocked_voice_regression", "similarity": exc.similarity,
                    "threshold": exc.threshold}
        except ApprovalRequired:
            return {"status": "blocked", "reason": "human approver required"}
        except ProposalNotFound as exc:
            return {"status": "not_found", "detail": str(exc)}
        return {"status": "approved", "agent": rec.agent, "new_version": rec.version}

    async def reject_proposal(self, *, proposal_id: int) -> dict:
        await self._proc.reject(proposal_id=proposal_id)
        return {"status": "rejected", "proposal_id": proposal_id}

    async def rollback(self, *, agent: str, brand_id: str) -> dict:
        try:
            rec = await self._proc.rollback(brand_id=brand_id, agent=agent)
        except NoVersionToRollback as exc:
            return {"status": "no_op", "detail": str(exc)}
        return {"status": "rolled_back", "agent": agent, "active_version": rec.version}
