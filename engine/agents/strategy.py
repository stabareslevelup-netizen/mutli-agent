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
from engine.core.models import NarrativeConstraint, StrategyPacket
from engine.core.validation_gate import ValidationGate


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
                     brand: BrandConfig, job_id: str) -> StrategyPacket:
        for fa in fused:  # fusion already sorted best-first
            violating = [c for c in constraints if self._checker.violates(fa.angle, c)]
            if violating:
                continue
            packet = StrategyPacket(
                chosen_angle=fa.angle,
                rationale=(f"top fused score {fa.score} "
                           f"(research={fa.research:.2f}, memory={fa.memory:.2f}, "
                           f"timing={fa.timing:.2f}); no staked-position conflict"),
                formats=brand.formats,
                fusion_weights=brand.fusion_weights,
                hard_constraints=constraints,
                inputs_digest=fa.digest,
            )
            return await self._gate.validate(StrategyPacket, packet.model_dump(),
                                             job_id=job_id, step="strategy->production")
        raise StrategyBlocked(
            "all candidate angles contradict an active staked narrative position")
