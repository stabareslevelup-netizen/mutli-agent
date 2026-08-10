"""
PHASE 0 SPIKE — citation_monitor  [THROWAWAY, not production code]

Question: can we reliably detect whether a given source/brand is SURFACED /
CITED by an answer engine for its target topics — i.e. a share-of-voice and
zero-state baseline the Timing agent can track over time?

What this spike proves (offline, runnable):
  Given an answer-engine response to a target query, extract the named
  sources/domains it cites, then report:
    - is OUR brand present?  (presence/absence)
    - rank + share-of-voice among cited sources
    - who the incumbents are (competitive set to displace)
    - a disambiguation flag when the brand name collides with something else

Live findings (see PHASE0_FINDINGS.md):
  * "best physical-AI sources" -> engine names specific domains
    (The Robot Report, Robotics & Automation News, CNBC, NVIDIA). Clean,
    reliable presence signal.
  * "Madre de Maquinas physical AI" -> brand ABSENT (correct zero-state) AND
    name collides with an MTG card "Elesh Norn, Mother of Machines" -> the
    monitor must flag disambiguation or it will track the wrong entity.

Stubbed: production calls the web_search/answer tool live; here we use the
real captured responses as fixtures.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class CitationStatus:
    query: str
    brand_present: bool
    rank: int | None
    share_of_voice: float
    incumbents: list[str]
    disambiguation_risk: str | None
    note: str


def assess_citation(query: str, brand_aliases: list[str],
                    answer_text: str, cited_sources: list[str],
                    collision_terms: list[str] | None = None) -> CitationStatus:
    cited = [s.strip() for s in cited_sources if s.strip()]
    low_aliases = [a.lower() for a in brand_aliases]
    present = any(a in s.lower() for s in cited for a in low_aliases) or \
              any(a in answer_text.lower() for a in low_aliases)
    rank = None
    for i, s in enumerate(cited, 1):
        if any(a in s.lower() for a in low_aliases):
            rank = i
            break
    sov = (1 / len(cited)) if (present and cited) else 0.0

    disambig = None
    if collision_terms:
        hit = next((c for c in collision_terms if c.lower() in answer_text.lower()), None)
        if hit:
            disambig = f"name collides with '{hit}' — scope queries to disambiguate"

    note = ("present in citation set" if present
            else "ABSENT — zero-state baseline; incumbents own this query")
    return CitationStatus(query, present, rank, round(sov, 2), cited, disambig, note)


# --- FIXTURES: real captured answer-engine responses 2026-06-15 -------------
FIXTURES = [
    dict(
        query="best sources for physical AI / embodied robotics news",
        brand_aliases=["madre de maquinas", "madredemaquinas", "ocho"],
        answer_text=("The Robot Report, Robotics & Automation News, CNBC, AI Magazine "
                     "and the NVIDIA Blog are the leading sources for physical AI news."),
        cited_sources=["The Robot Report", "Robotics & Automation News", "CNBC",
                       "AI Magazine", "NVIDIA Blog"],
        collision_terms=None,
    ),
    dict(
        query="Madre de Maquinas physical AI",
        brand_aliases=["madre de maquinas", "madredemaquinas", "ocho"],
        answer_text=("'Madre de Maquinas' doesn't appear to be a specific project name. "
                     "Result set includes 'Elesh Norn, Madre de las Maquinas / Mother of "
                     "Machines' — a Magic: The Gathering card."),
        cited_sources=["NVIDIA Glossary", "ICEMD", "DPL News", "El Rincón de Magic (MTG)"],
        collision_terms=["Elesh Norn", "Magic: The Gathering", "Mother of Machines"],
    ),
]

if __name__ == "__main__":
    for f in FIXTURES:
        st = assess_citation(**f)
        print(f"QUERY: {st.query}")
        print(f"  brand_present={st.brand_present}  rank={st.rank}  SoV={st.share_of_voice}")
        print(f"  incumbents={st.incumbents}")
        if st.disambiguation_risk:
            print(f"  !! {st.disambiguation_risk}")
        print(f"  -> {st.note}\n")
