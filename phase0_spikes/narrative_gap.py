"""
PHASE 0 SPIKE — narrative_gap  [THROWAWAY, not production code]

Question: given a topic, can we reliably surface ANGLES NOT BEING COVERED
(white space the brand could own) — and are those gaps real, not hallucinated?

What this spike proves (offline, runnable):
  Given a topic's coverage corpus + a set of candidate angles (the brand
  pillars make good candidates), measure coverage DENSITY per angle and flag
  low-density angles as gaps. Crucially it separates ENTITY-SPECIFIC white
  space from TOPIC-GENERIC saturation — the trap found during the spike:
  "automation & jobs" is saturated generically, but "BotQ-specific labor
  impact" is wide open. A naive detector misses that.

Verification done live (see PHASE0_FINDINGS.md): the top gap it proposes for
Figure 03 — an "incident file" angle — was confirmed under-covered: the viral
malfunction footage is Unitree H1, NOT Figure; Figure 02->03 had only "minor
forearm issues." So the gap is genuine and checkable.

Stubbed: in production each angle's density comes from a fresh scoped
web_search count; here we use the real captured corpora as fixtures.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Gap:
    angle: str
    density: float          # 0..1 how covered this angle already is
    is_gap: bool
    entity_specific: bool   # True = white space is around the named entity
    note: str


def _density(angle_terms: list[str], corpus: list[dict], entity: str | None) -> tuple[float, bool]:
    """Fraction of corpus items that substantively hit the angle. If `entity`
    given, also check whether hits are entity-specific or only generic."""
    text_items = [f"{r.get('title','')} {r.get('snippet','')}".lower() for r in corpus]
    hits = [t for t in text_items if any(term in t for term in angle_terms)]
    density = len(hits) / max(len(text_items), 1)
    entity_hits = [t for t in hits if entity and entity.lower() in t]
    entity_specific_gap = bool(entity) and len(hits) > 0 and len(entity_hits) == 0
    return density, entity_specific_gap


def find_gaps(topic: str, entity: str, corpus: list[dict],
              candidate_angles: dict[str, list[str]], threshold: float = 0.34) -> list[Gap]:
    gaps = []
    for angle, terms in candidate_angles.items():
        density, ent_gap = _density(terms, corpus, entity)
        # a gap = thin overall coverage, OR coverage exists but never about the entity
        is_gap = density < threshold or ent_gap
        note = "thin coverage" if density < threshold else (
            "covered generically but NOT for this entity" if ent_gap else "saturated")
        gaps.append(Gap(angle, round(density, 2), is_gap, ent_gap, note))
    gaps.sort(key=lambda g: (not g.is_gap, g.density))
    return gaps


# --- FIXTURES: real captured corpora 2026-06-15 -----------------------------
# Figure 03 coverage is overwhelmingly about production THROUGHPUT.
FIGURE_CORPUS = [
    {"title": "Figure ramps production from one per day to one per hour", "snippet": "24x throughput, 350 robots delivered"},
    {"title": "Figure’s Factory Just Hit a Major Production Milestone", "snippet": "BotQ line up to 12,000/yr, 80% first-pass yield"},
    {"title": "Introducing Figure 03", "snippet": "third generation humanoid, Helix AI, hardware specs"},
    {"title": "Figure claims one humanoid robot production per hour", "snippet": "150 networked workstations, supplier qualification"},
    # NOTE: corpus contains ZERO Figure-03 incident items and ZERO BotQ-labor items.
]

CANDIDATE_ANGLES = {  # brand pillars from madre_de_maquinas.yaml make good candidates
    "throughput / production rate": ["production", "per hour", "throughput", "ramp", "yield"],
    "incident_file (failures in the field)": ["malfunction", "injury", "incident", "fail", "recall", "safety"],
    "deployment_reality (real environments)": ["deployed", "real-world", "customer site", "warehouse", "messy"],
    "company_intel (strategy/decisions)": ["funding", "strategy", "decision", "ceo", "roadmap", "supplier"],
    "labor impact (this entity specifically)": ["jobs", "workers", "labor", "displace", "wages"],
}

if __name__ == "__main__":
    gaps = find_gaps("Figure 03 production", "Figure", FIGURE_CORPUS, CANDIDATE_ANGLES)
    print("TOPIC: Figure 03 production ramp   (entity=Figure/BotQ)\n")
    for g in gaps:
        tag = "GAP " if g.is_gap else "    "
        es = " [entity-specific white space]" if g.entity_specific else ""
        print(f"{tag} density={g.density:<4} {g.angle}{es}\n       -> {g.note}")
