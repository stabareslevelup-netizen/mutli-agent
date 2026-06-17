"""
engine/agents/strategy.py — Strategy agent [PROVEN].

Applies weighted fusion (Tier 1) and selects the chosen angle, with NARRATIVE
MEMORY AS A HARD CONSTRAINT: it will never select an angle that contradicts an
active staked position. If every candidate violates, it HALTS rather than
contradict (StrategyBlocked) — wired to the dead-letter at the orchestrator.

The contradiction check is an injectable ConstraintChecker. Default is lexical
and conservative: if an angle is clearly about a staked position but does not
align with that staked stance, it's treated as a violation. An embedding/LLM
checker can swap in without touching this logic.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from engine.core.brand_loader import BrandConfig
from engine.core.fusion import FusedAngle
from engine.core.models import ContentFormat, NarrativeConstraint, StrategyPacket, VelocityVerdict
from engine.core.validation_gate import ValidationGate

# cues that mark a deep / narrative angle (favors video)
_DEEP_CUES = {"why", "inside", "story", "reality", "fails", "failure", "incident",
              "learn", "lesson", "investigation", "aftermath", "postmortem", "deep"}


def choose_content_format(velocity: VelocityVerdict, angle_text: str, pillars) -> ContentFormat:
    """Format routing: pillar hint (config) is the baseline; a deep/narrative
    angle favors video; high velocity (breaking) favors speed (image/text_only)
    EXCEPT when the pillar leans video (depth wins). Default video."""
    at = _tokens(angle_text)
    matched, best = None, 0.0
    for p in pillars:
        score = _jaccard(at, _tokens(f"{p.id} {p.desc}"))
        if score > best:
            best, matched = score, p
    base = (matched.format if (matched and matched.format) else ContentFormat.video)
    is_deep = bool(at & _DEEP_CUES) or base == ContentFormat.video
    breaking = velocity == VelocityVerdict.surging
    if breaking and not is_deep:                       # speed wins for non-deep breaking angles
        return ContentFormat.text_only if base == ContentFormat.text_only else ContentFormat.image
    return base                                         # low urgency or deep -> pillar baseline


class StrategyBlocked(Exception):
    """Every candidate angle contradicts a staked position; nothing is chosen."""


# Stopwords are filtered so overlap reflects content words, not "the"/"and".
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


class StrategyAgent:
    name = "strategy"

    def __init__(self, gate: ValidationGate, checker: Optional[ConstraintChecker] = None):
        self._gate = gate
        self._checker = checker or LexicalConstraintChecker()

    async def decide(self, *, fused: list[FusedAngle], constraints: list[NarrativeConstraint],
                     brand: BrandConfig, job_id: str,
                     velocity: VelocityVerdict = VelocityVerdict.unknown) -> StrategyPacket:
        for fa in fused:  # fusion already sorted best-first
            violating = [c for c in constraints if self._checker.violates(fa.angle, c)]
            if violating:
                continue
            fmt = choose_content_format(velocity, fa.angle, brand.pillars)
            packet = StrategyPacket(
                chosen_angle=fa.angle,
                rationale=(f"top fused score {fa.score} "
                           f"(research={fa.research:.2f}, memory={fa.memory:.2f}, "
                           f"timing={fa.timing:.2f}); format={fmt.value}; no staked-position conflict"),
                content_format=fmt,
                formats=brand.formats,
                fusion_weights=brand.fusion_weights,
                hard_constraints=constraints,
                inputs_digest=fa.digest,
            )
            return await self._gate.validate(StrategyPacket, packet.model_dump(),
                                             job_id=job_id, step="strategy->production")
        raise StrategyBlocked(
            "all candidate angles contradict an active staked narrative position")
