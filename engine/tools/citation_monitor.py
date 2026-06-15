"""
engine/tools/citation_monitor.py — is the brand surfaced/cited by answer engines?

RULE 3 (Phase 0): disambiguation-aware. The Phase-0 false positive came from a
naive substring match of the brand name inside an unrelated same-name context
(a Magic: The Gathering card). Production fix:

  - "present" requires the alias to appear in a CITED SOURCE *and* no unresolved
    name collision (verified context). Answer-text-only mention is not enough.
  - An unresolved collision (a same-name entity present, brand not in citations)
    is reported as "ambiguous", never present/absent.

Brand-free: aliases + collision terms are injected at runtime (from the brand
config and the live search), never hardcoded here.
"""
from __future__ import annotations

from typing import Optional

from engine.core.models import CitationPresence, CitationSignal, DisambiguationGuard


def assess_citation(*, query: str, brand_aliases: list[str], answer_text: str,
                    cited_sources: list[str],
                    collision_terms: Optional[list[str]] = None) -> CitationSignal:
    cited = [s.strip() for s in cited_sources if s and s.strip()]
    aliases = [a.lower() for a in brand_aliases if a]
    collisions = collision_terms or []

    alias_in_cited = any(a in s.lower() for s in cited for a in aliases)
    collision_present = any(c.lower() in answer_text.lower() for c in collisions)
    collision_unresolved = collision_present and not alias_in_cited

    rank = None
    for i, s in enumerate(cited, 1):
        if any(a in s.lower() for a in aliases):
            rank = i
            break

    verified = alias_in_cited and not collision_unresolved
    if verified:
        presence = CitationPresence.present
        sov = round(1 / len(cited), 3) if cited else 0.0
    elif collision_unresolved:
        presence = CitationPresence.ambiguous
        sov = 0.0
    else:
        presence = CitationPresence.absent
        sov = 0.0

    guard = DisambiguationGuard(
        collision_terms=collisions,
        matched_context_verified=verified,
        risk=("unresolved name collision — scope queries to disambiguate"
              if collision_unresolved else None))
    return CitationSignal(query=query, brand_aliases=brand_aliases, presence=presence,
                          rank=rank, share_of_voice=sov, incumbents=cited,
                          disambiguation=guard)
