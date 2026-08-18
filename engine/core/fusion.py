"""
engine/core/fusion.py — explicit, inspectable weighted fusion of Tier 1.

# ORPHANED - Phase 2 migration. Candidate for cleanup. Do not delete until
# after Phase 5 is complete. Unreachable from the new pipeline
# (run_sweep()/_process_item() never call fuse() -- Strategy now operates
# on a single ResearchItem, not a multi-angle blend). Also currently
# BROKEN, not just unused: it imports ResearchOutput, removed from
# models.py in the Phase 2 schema migration (replaced by ResearchItem).
# Left as-is rather than "fixed" since fixing dead code with no caller
# would be pointless churn.

Not emergent: Research / Memory / Timing are combined by a deterministic
function using the brand's fusion_weights. Each candidate angle (from Research)
gets a score and a per-signal contribution breakdown so the decision is fully
auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from engine.core.models import (
    MemoryQueryResult, ResearchOutput, TimingSignal, VelocityVerdict,
)

_VELOCITY_WEIGHT = {
    VelocityVerdict.surging: 1.0,
    VelocityVerdict.steady: 0.5,
    VelocityVerdict.declining: 0.2,
    VelocityVerdict.dead: 0.0,
    VelocityVerdict.unknown: 0.3,
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


@dataclass
class FusedAngle:
    angle: str
    score: float
    research: float
    memory: float
    timing: float
    digest: dict[str, float] = field(default_factory=dict)


def _memory_relevance(angle: str, memory: MemoryQueryResult) -> float:
    at = _tokens(angle)
    best = 0.0
    for item in (*memory.episodic, *memory.semantic):
        best = max(best, _jaccard(at, _tokens(item.content)))
    return best


def _timing_score(timing: TimingSignal) -> float:
    vel = _VELOCITY_WEIGHT.get(timing.velocity.verdict, 0.3)
    gaps = timing.gaps.gaps
    gap_open = (sum(1 for g in gaps if g.is_open) / len(gaps)) if gaps else 0.0
    # absent from answer engines = white space to own; present = already saturated
    cite = {"absent": 1.0, "ambiguous": 0.5, "present": 0.2}.get(
        timing.citation.presence.value, 0.3)
    return round(0.5 * vel + 0.3 * gap_open + 0.2 * cite, 4)


def fuse(*, research: ResearchOutput, memory: MemoryQueryResult,
         timing: TimingSignal, weights: dict[str, float]) -> list[FusedAngle]:
    wr, wm, wt = weights["research"], weights["memory"], weights["timing"]
    t_score = _timing_score(timing)   # topic-level; shared across angles
    fused: list[FusedAngle] = []
    for a in research.angles:
        r = a.confidence
        m = _memory_relevance(a.angle, memory)
        score = round(wr * r + wm * m + wt * t_score, 4)
        fused.append(FusedAngle(
            angle=a.angle, score=score, research=r, memory=m, timing=t_score,
            digest={"research": round(wr * r, 4), "memory": round(wm * m, 4),
                    "timing": round(wt * t_score, 4), "score": score}))
    fused.sort(key=lambda f: f.score, reverse=True)
    return fused
