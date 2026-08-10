"""
engine/memory/procedural.py — GUARDED self-improvement (the dangerous part).

Procedural memory = the system rewriting agent prompts from performance data.
Left autonomous it drifts the brand voice toward engagement bait. So in v1:

  - Changes are PROPOSED, never auto-applied. `propose()` writes a pending
    proposal and changes NOTHING live.
  - `approve()` is the ONLY path that activates a new prompt version, and it
    requires (a) a human `approved_by` and (b) the proposed prompt passing a
    held-out voice-similarity check. If voice similarity drops below threshold
    the change is BLOCKED and the proposal rejected.
  - `rollback()` reverts to the previous version in one action.

The voice checker is injectable (Protocol). The default is lexical (offline,
no embeddings); an embedding-based checker can swap in without touching this
logic.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from engine.memory.backend import MemoryBackend, ProceduralRecord, ProposalRecord


class ProposalNotFound(Exception):
    pass


class ApprovalRequired(Exception):
    """approve() was called without a human approver."""


class VoiceRegressionBlocked(Exception):
    """Proposed prompt's voice similarity fell below the threshold."""

    def __init__(self, similarity: float, threshold: float):
        self.similarity = similarity
        self.threshold = threshold
        super().__init__(f"voice similarity {similarity:.3f} < threshold {threshold:.3f}; change blocked")


class NoVersionToRollback(Exception):
    pass


@runtime_checkable
class VoiceChecker(Protocol):
    def similarity(self, candidate: str, reference_set: list[str]) -> float: ...


class LexicalVoiceChecker:
    """Offline default: mean token Jaccard overlap against the reference set.
    A proxy for 'does this still sound like the brand'; swap for an
    embedding-based checker when embeddings are reachable."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", text.lower()))

    def similarity(self, candidate: str, reference_set: list[str]) -> float:
        if not reference_set:
            return 1.0
        cand = self._tokens(candidate)
        sims = []
        for ref in reference_set:
            r = self._tokens(ref)
            union = cand | r
            sims.append(len(cand & r) / len(union) if union else 0.0)
        return sum(sims) / len(sims)


class ProceduralMemory:
    def __init__(self, backend: MemoryBackend, voice_checker: Optional[VoiceChecker] = None,
                 reference_set: Optional[list[str]] = None, voice_threshold: float = 0.6):
        self._backend = backend
        self._voice = voice_checker or LexicalVoiceChecker()
        self._refs = reference_set or []
        self._threshold = voice_threshold

    @property
    def voice_threshold(self) -> float:
        return self._threshold

    # --- read --------------------------------------------------------------
    async def active_prompt(self, *, brand_id: str, agent: str) -> Optional[str]:
        rec = await self._backend.active_procedural(brand_id=brand_id, agent=agent)
        return rec.prompt_text if rec else None

    async def versions(self, *, brand_id: str, agent: str) -> list[ProceduralRecord]:
        return await self._backend.procedural_versions(brand_id=brand_id, agent=agent)

    async def pending_proposals(self) -> list[ProposalRecord]:
        return await self._backend.list_proposals(status="pending")

    # --- seed the initial (human-authored) prompt --------------------------
    async def seed(self, *, brand_id: str, agent: str, prompt_text: str) -> ProceduralRecord:
        return await self._backend.insert_procedural(
            brand_id=brand_id, agent=agent, prompt_text=prompt_text,
            version=1, active=True, approved_by="seed")

    # --- propose (NEVER applies) -------------------------------------------
    async def propose(self, *, brand_id: str, agent: str, proposed_prompt: str,
                      performance_data: dict,
                      reference_set: Optional[list[str]] = None) -> ProposalRecord:
        refs = reference_set if reference_set is not None else self._refs
        sim = self._voice.similarity(proposed_prompt, refs)
        cur = await self._backend.active_procedural(brand_id=brand_id, agent=agent)
        return await self._backend.create_proposal(
            brand_id=brand_id, agent=agent,
            current_version=cur.version if cur else 0,
            proposed_prompt=proposed_prompt, performance_data=performance_data,
            voice_similarity=sim)

    # --- approve (the ONLY activation path) --------------------------------
    async def approve(self, *, proposal_id: int, approved_by: str) -> ProceduralRecord:
        if not approved_by:
            raise ApprovalRequired("approve() requires a human approver")
        p = await self._backend.get_proposal(proposal_id=proposal_id)
        if p is None:
            raise ProposalNotFound(str(proposal_id))
        if p.status != "pending":
            raise ProposalNotFound(f"proposal {proposal_id} is {p.status}, not pending")

        sim = p.voice_similarity if p.voice_similarity is not None else 1.0
        if sim < self._threshold:
            await self._backend.set_proposal_status(proposal_id=proposal_id, status="rejected")
            raise VoiceRegressionBlocked(sim, self._threshold)

        cur = await self._backend.active_procedural(brand_id=p.brand_id, agent=p.agent)
        new_version = (cur.version + 1) if cur else 1
        if cur:
            await self._backend.set_procedural_flags(rec_id=cur.id, active=False)
        rec = await self._backend.insert_procedural(
            brand_id=p.brand_id, agent=p.agent, prompt_text=p.proposed_prompt,
            version=new_version, active=True, approved_by=approved_by)
        await self._backend.set_proposal_status(proposal_id=proposal_id, status="approved")
        return rec

    async def reject(self, *, proposal_id: int) -> None:
        await self._backend.set_proposal_status(proposal_id=proposal_id, status="rejected")

    # --- rollback (one action) ---------------------------------------------
    async def rollback(self, *, brand_id: str, agent: str) -> ProceduralRecord:
        versions = await self._backend.procedural_versions(brand_id=brand_id, agent=agent)
        actives = [v for v in versions if v.active]
        if not actives:
            raise NoVersionToRollback(f"no active version for {agent}")
        current = max(actives, key=lambda v: v.version)
        previous = [v for v in versions if v.version < current.version]
        if not previous:
            raise NoVersionToRollback(f"no prior version for {agent} to roll back to")
        target = max(previous, key=lambda v: v.version)
        await self._backend.set_procedural_flags(rec_id=current.id, active=False, rolled_back=True)
        await self._backend.set_procedural_flags(rec_id=target.id, active=True)
        return target
