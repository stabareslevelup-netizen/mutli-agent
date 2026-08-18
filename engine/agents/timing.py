"""
engine/agents/timing.py — Timing agent [Phase 2, X-agent migration].

Assigns each ResearchItem a posting slot (or rejects it) across the WHOLE
batch at once -- "one item per slot per day" is a cross-item constraint,
enforced at the code level below (LLM slot self-enforcement isn't trusted;
see _dedup_slots).

velocity_probe.py and citation_monitor.py are NOT reused here -- they
answer different questions than the Phase 2 fields need: the old
citation_monitor.py checks whether OUR BRAND is cited by answer engines
(brand-visibility tracking), while citation_hedge_required asks whether a
specific claim needs hedging (verification confidence) -- unrelated
questions despite the similar name. velocity_probe.py's structured ordinal
tool was built for a single topic with a fetched corpus; TimingDecision has
no velocity-verdict field anymore, just free-text urgency_note. Both files
are orphaned by this migration (see their own docstrings), not deleted.

duplicate_check and narrative_gap_check use PostHistoryStore, which has
real state those old tools never had access to. Velocity/urgency judgment
and citation-hedge judgment fold into the one batch LLM call below, same
cost-bounding reasoning as Research.sweep().
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

from engine.agents.base import AgentContext, BaseAgent
from engine.core.embeddings import safe_embed
from engine.core.llm import WEB_SEARCH_TOOL
from engine.core.models import PostingSlot, ResearchItem, TimingDecision
from engine.core.validation_gate import HandoffHalted

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are the Timing Agent for an X posting pipeline. You receive a list of "
    "ResearchItems that passed novelty scoring. Decide WHEN each should post and "
    "WHETHER conditions are right to post it now.\n\n"
    "POSTING SLOTS (ET):\n"
    "7AM -- short_hook only (<=100 chars). Most punchy item.\n"
    "9AM -- pov_post (150-240 chars). Overnight developments.\n"
    "12PM -- thread (4-7 tweets). Deepest item.\n"
    "3PM -- data_drop (200-260 chars). Contract award, funding, job signal.\n"
    "6PM -- pov_post. End-of-day analysis.\n\n"
    "For each item, you're given is_duplicate and topic_seen_recently flags "
    "(precomputed against real post history). is_duplicate=true items must be "
    "recommended_slot='reject' -- exact same source already posted within 72h, no "
    "exceptions. topic_seen_recently=true items should be 'reject' UNLESS this item "
    "directly contradicts or advances the prior post -- use judgment, explain in "
    "urgency_note either way.\n\n"
    "citation_hedge_required: does this item's claim rest on something preliminary, "
    "inferred, or a single unverified document? If yes, set true -- Copy will hedge "
    "the language. A primary-source document (a real SAM.gov award, a real bill "
    "text) being the ONLY source does not by itself require hedging -- primary "
    "sources are authoritative. Inference, rumor, or a single blog's interpretation "
    "of a primary source does.\n\n"
    "Assign AT MOST ONE item per slot -- if two genuinely compete for the same slot, "
    "you may recommend the same slot for both; a deterministic tie-break by "
    "novelty_score runs after your output, so don't worry about enforcing uniqueness "
    "perfectly yourself.\n\n"
    "Return ONLY a JSON array, no prose, no fences. Each element:\n"
    '{"item_id":"<pass through>","recommended_slot":"7AM|9AM|12PM|3PM|6PM|reject",'
    '"rejection_reason":"<required if reject, else null>","urgency_note":"<one '
    'sentence>","citation_hedge_required":<bool>}'
)

_USER = """Items (with precomputed duplicate/topic-recency flags):
{items}

Today's date: {today}

Return the JSON array as specified."""


class TimingAgent(BaseAgent):
    name = "timing"

    def __init__(self, ctx: AgentContext, *, post_history=None, embeddings=None):
        super().__init__(ctx)
        self._post_history = post_history   # PostHistoryStore; None -> both checks degrade to False
        self._embeddings = embeddings       # EmbeddingProvider; None -> topic_posted_recently always False

    async def assign(self, *, items: list[ResearchItem], job_id: Optional[str]) -> list[TimingDecision]:
        if not items:
            return []
        precheck = await self._precheck(items)
        payload = [{"item_id": i.item_id, "headline": i.headline, "post_angle": i.post_angle,
                   "novelty_score": i.novelty_score, "source": i.source.value,
                   **precheck[i.item_id]} for i in items]
        raw = await self._complete_json(
            system=_SYSTEM,
            user=_USER.format(items=json.dumps(payload, default=str), today=date.today().isoformat()),
            job_id=job_id or "sweep", tools=[WEB_SEARCH_TOOL], max_tokens=3000)

        candidates = raw if isinstance(raw, list) else (
            raw.get("decisions", []) if isinstance(raw, dict) else [])
        decisions: list[TimingDecision] = []
        for c in candidates:
            try:
                d = await self._validate(TimingDecision, c, job_id=job_id or "sweep",
                                         step="timing.assign")
            except HandoffHalted:
                continue   # dead-lettered inside validate(); one bad item can't kill the batch
            decisions.append(d)

        decisions = self._enforce_duplicate_override(decisions, precheck)
        by_id = {i.item_id: i for i in items}
        decisions = self._dedup_slots(decisions, by_id)
        return decisions

    async def _precheck(self, items: list[ResearchItem]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for item in items:
            is_dup = False
            topic_seen = False
            if self._post_history is not None:
                try:
                    is_dup = await self._post_history.was_posted_recently(
                        source_url=str(item.source_url), within_hours=72)
                except Exception as exc:
                    logger.warning("timing.assign: duplicate_check failed for %s -- %s: %s",
                                   item.item_id, type(exc).__name__, exc)
                embedding = None
                if self._embeddings is not None:
                    embedding = (await safe_embed(self._embeddings, [item.post_angle]))[0]
                try:
                    topic_seen = await self._post_history.topic_posted_recently(
                        post_angle_embedding=embedding, within_hours=48)
                except Exception as exc:
                    logger.warning("timing.assign: narrative_gap_check failed for %s -- %s: %s",
                                   item.item_id, type(exc).__name__, exc)
            out[item.item_id] = {"is_duplicate": is_dup, "topic_seen_recently": topic_seen}
        return out

    def _enforce_duplicate_override(self, decisions: list[TimingDecision],
                                    precheck: dict[str, dict]) -> list[TimingDecision]:
        out = []
        for d in decisions:
            if precheck.get(d.item_id, {}).get("is_duplicate") and d.recommended_slot != PostingSlot.reject:
                d = d.model_copy(update={"recommended_slot": PostingSlot.reject,
                                         "rejection_reason": "duplicate source_url posted within 72h"})
            out.append(d)
        return out

    def _dedup_slots(self, decisions: list[TimingDecision], by_id: dict) -> list[TimingDecision]:
        by_slot: dict[PostingSlot, list[int]] = {}
        for idx, d in enumerate(decisions):
            if d.recommended_slot != PostingSlot.reject:
                by_slot.setdefault(d.recommended_slot, []).append(idx)
        for idxs in by_slot.values():
            if len(idxs) <= 1:
                continue
            ranked = sorted(idxs, key=lambda i: by_id[decisions[i].item_id].novelty_score, reverse=True)
            for loser in ranked[1:]:
                decisions[loser] = decisions[loser].model_copy(update={
                    "recommended_slot": PostingSlot.reject,
                    "rejection_reason": "slot collision -- a higher novelty_score item took this slot"})
        return decisions
