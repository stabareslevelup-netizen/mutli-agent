"""
engine/agents/strategy.py — Strategy agent [Phase 2, X-agent migration].

The narrative-conflict check is NOT trusted to LLM self-report -- same
philosophy as CopyOutput.char_count / QualityOutput's gate math elsewhere
in this migration, and how the OLD Strategy agent did this exact check
(100% code-driven, zero LLM involvement). The LLM picks chosen_angle/
format/etc freely; the existing checker/LexicalConstraintChecker then runs
against the LLM's ACTUAL chosen_angle, and narrative_conflict_flag/note are
overridden with the real computed result regardless of what the LLM said.

The conflict SINK WRITE happens here too, right next to the check that
triggers it -- not in the Orchestrator. narrative_conflict_flag stays pure
data (never an exception): Strategy still just returns a flagged
StrategyOutput. The Orchestrator still owns halting the JOB (JobStore
status, JobResult) since only it has job-state access -- see
orchestrator.py's _halt_narrative_conflict(), trimmed to just that.

Old Strategy made zero LLM calls (pure code over pre-fused angles from
fusion.py). New Strategy genuinely needs LLM judgment (angle framing,
format selection, must_include/hashtags) since there's no fusion step
anymore -- one ResearchItem in, one StrategyOutput out.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from engine.agents.base import AgentContext, BaseAgent
from engine.core.models import (
    MemoryQueryResult, NarrativeConstraint, ResearchItem, StrategyOutput, TimingDecision,
)

# --- checker mechanism: UNCHANGED from the old strategy.py ---------------
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "of", "to", "in", "on", "for", "with", "as", "at", "by", "we",
    "it", "this", "that", "from", "our", "their", "its", "will", "would",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower())
            if t not in _STOP and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


@runtime_checkable
class ConstraintChecker(Protocol):
    def violates(self, angle_text: str, constraint: NarrativeConstraint) -> bool: ...


class LexicalConstraintChecker:
    """Conservative default: an angle that is about a staked position but does
    not align with its stance is treated as a contradiction."""

    def __init__(self, position_overlap: float = 0.18, stance_overlap: float = 0.12):
        self._pos = position_overlap
        self._stance = stance_overlap

    def violates(self, angle_text: str, c: NarrativeConstraint) -> bool:
        at = _tokens(angle_text)
        about_position = _jaccard(at, _tokens(c.position)) >= self._pos
        aligns_stance = _jaccard(at, _tokens(c.stance)) >= self._stance
        return about_position and not aligns_stance


# --- Phase 2 prompt --------------------------------------------------------
_SYSTEM = (
    "You are the Strategy Agent for an X posting pipeline. You receive a single "
    "ResearchItem that passed timing checks. Choose the exact angle and set "
    "constraints Copy will use to write the post.\n\n"
    "ACCOUNT THESIS -- never deviate:\n"
    "This account publishes primary-source intelligence on physical AI, defense tech, "
    "and autonomous systems -- before it becomes mainstream news.\n"
    "Identity: \"The account that finds what's being built before it's announced.\"\n\n"
    "ANGLE PRIORITY (pick highest applicable):\n"
    "1. Something being built that nobody has reported yet\n"
    "2. A contract or funding signal that predicts a future announcement\n"
    "3. A failure or setback that contradicts the prevailing hype narrative\n"
    "4. A policy or regulatory move that will affect the industry\n"
    "5. A technical paper with real-world implications most people missed\n\n"
    "FORMAT SELECTION:\n"
    "- short_hook: Raw fact so surprising it needs no context. Target 71-100 chars.\n"
    "- pov_post: Fact needs one sentence of interpretation. Target 150-240 chars.\n"
    "- thread: Story has 3+ connected facts or requires explanation. 4-7 tweets.\n"
    "- data_drop: Contract award, funding round, headcount signal. Target 200-260 chars.\n\n"
    "If a skeptic critique is included below, address it directly in a revised angle "
    "and must_include -- don't repeat the same mistake.\n\n"
    "PILLAR -- tag the angle with the single content pillar it actually belongs to:\n"
    "physical_ai_readiness, defense_procurement, training_performance, plant_based_fuel, "
    "building_with_ai. Use \"other\" ONLY when the angle genuinely fits none of the five -- "
    "don't force a fit just to avoid \"other\".\n\n"
    "Return ONLY a JSON object, no prose, no fences:\n"
    '{"chosen_angle":"<one sentence>","format":"short_hook|pov_post|thread|data_drop",'
    '"must_include":["<1-3 specific facts>"],"must_avoid":["<framings to avoid>"],'
    '"hashtags":["<0-2 from: #AI #PhysicalAI #Robotics #DefenseTech #AutonomousSystems '
    '#AIAgents #FutureOfWar #DARPA>"],"thread_spine":["<4-7 bullets, ONLY if format is '
    'thread, omit otherwise>"],"pillar":"physical_ai_readiness|defense_procurement|'
    'training_performance|plant_based_fuel|building_with_ai|other"}'
)

_USER = """Item:
headline: {headline}
raw_detail: {raw_detail}
post_angle (Research's proposed framing -- you may refine it): {post_angle}
source: {source}
novelty_score: {novelty_score}
recommended_slot: {slot}
citation_hedge_required: {hedge}

{critique_block}
Return the JSON object as specified."""


class StrategyAgent(BaseAgent):
    name = "strategy"

    def __init__(self, ctx: AgentContext, *, checker: Optional[ConstraintChecker] = None,
                 narrative_conflict_sink=None):
        super().__init__(ctx)
        self._checker = checker or LexicalConstraintChecker()
        self._nc = narrative_conflict_sink   # sink write lives here, next to the check

    async def decide_v2(self, *, item: ResearchItem, timing: TimingDecision,
                        memory: MemoryQueryResult, job_id: str,
                        skeptic_critique: Optional[str] = None) -> StrategyOutput:
        critique_block = (f"Skeptic's critique of your previous attempt: {skeptic_critique}\n"
                          f"Address this directly." if skeptic_critique
                          else "(first attempt, no critique yet)")
        user = _USER.format(headline=item.headline, raw_detail=item.raw_detail,
                            post_angle=item.post_angle, source=item.source.value,
                            novelty_score=item.novelty_score,
                            slot=timing.recommended_slot.value,
                            hedge=timing.citation_hedge_required,
                            critique_block=critique_block)
        raw = await self._complete_json(system=_SYSTEM, user=user, job_id=job_id, max_tokens=1500)
        out = await self._validate(StrategyOutput, raw, job_id=job_id, step="strategy.decide_v2")

        # code-level check on the LLM's ACTUAL chosen_angle -- not LLM self-report.
        # Set UNCONDITIONALLY from the computed result (both directions), not just
        # overridden on the positive case -- otherwise a self-reported
        # narrative_conflict_flag=True with no real violation leaks through unchecked.
        violating = [c for c in memory.narrative if self._checker.violates(out.chosen_angle, c)]
        note = (f"contradicts staked position: {violating[0].position} ({violating[0].stance})"
               if violating else None)
        out = out.model_copy(update={"narrative_conflict_flag": bool(violating),
                                     "narrative_conflict_note": note})
        if violating and self._nc is not None:
            await self._nc.record(
                job_id=job_id, item_id=item.item_id, chosen_angle=out.chosen_angle,
                conflict_note=note, source_url=str(item.source_url))
        return out
