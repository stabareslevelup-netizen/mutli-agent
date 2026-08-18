"""
engine/tools/narrative_gap.py — find under-covered angles (white space).

# ORPHANED - Phase 2 migration. Candidate for cleanup. Do not delete until
# after Phase 5 is complete. No live caller: the old Timing agent used this
# for RULE 2 gap-finding against brand pillars, a concept the new
# TimingDecision schema doesn't have. Strategy's narrative-conflict check
# (which sounds similar) uses a separate checker/NarrativeConstraint
# mechanism instead — verified this module has no remaining callers.

RULE 2 (Phase 0): distinguish ENTITY-SPECIFIC white space from TOPIC-GENERIC
saturation. A candidate angle can be saturated at the topic level yet wide open
for a named entity. The corpus is gathered by the Timing agent via web_search;
this function is pure + brand-free (candidate angles are injected from the
brand pillars at runtime).
"""
from __future__ import annotations

import re
from typing import Optional

from engine.core.models import GapType, NarrativeGap, NarrativeGapSignal


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def find_gaps(*, topic: str, entity: Optional[str], corpus: list[dict],
              candidates: list[dict], threshold: float = 0.34) -> NarrativeGapSignal:
    """
    corpus: [{"title","snippet",...}]
    candidates: [{"angle", "gap_type": GapType, "entity": str|None, "terms": [..]}]
    """
    texts = [f"{c.get('title','')} {c.get('snippet','')}".lower() for c in corpus]
    gaps: list[NarrativeGap] = []
    for cand in candidates:
        terms = [t.lower() for t in cand["terms"]]
        hits = [t for t in texts if any(term in t for term in terms)]
        density = len(hits) / max(len(texts), 1)
        ent = cand.get("entity")
        entity_hits = [t for t in hits if ent and ent.lower() in t]
        # entity-specific white space: covered generically but not for the entity
        entity_specific_gap = bool(ent) and len(hits) > 0 and len(entity_hits) == 0
        is_open = density < threshold or entity_specific_gap
        note = ("covered generically but NOT for this entity" if entity_specific_gap
                else "thin coverage" if density < threshold else "saturated")
        gaps.append(NarrativeGap(
            angle=cand["angle"], gap_type=cand["gap_type"], entity=ent,
            coverage_density=round(density, 3), is_open=is_open, evidence=note))
    gaps.sort(key=lambda g: (not g.is_open, g.coverage_density))
    return NarrativeGapSignal(topic=topic, entity=entity, gaps=gaps)
