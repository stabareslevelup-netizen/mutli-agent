"""
engine/agents/timing.py — Timing agent [UNPROVEN, de-risked in Phase 0].

Wires the three signals through the server-side web_search tool:
  - velocity_probe  -> ordinal verdict (RULE 1: no fake-precise numbers until a
    real time-series source is allowlisted; numeric sources self-report
    unavailable and the probe degrades to ordinal)
  - narrative_gap   -> entity-specific vs topic-generic white space (RULE 2)
  - citation_monitor-> disambiguation-aware presence (RULE 3)

To cap spend: ONE web_search call gathers the topic corpus (shared by velocity
+ gap), and ONE answer-engine call drives citation. Opus tier (reasoning).
Brand-free: pillars + brand aliases are injected at runtime.
"""
from __future__ import annotations

import re
from typing import Optional

from engine.agents.base import BaseAgent
from engine.core.llm import WEB_SEARCH_TOOL
from engine.core.models import (
    GapType, PrimarySourceSignal, Source, TimingSignal,
)
from engine.tools.citation_monitor import assess_citation
from engine.tools.narrative_gap import find_gaps
from engine.tools.velocity_probe import build_default_probe

_SEARCH_SYS = "You are a web research tool. Use web_search. Return ONLY JSON, no prose."
_SEARCH_USER = ('Search for recent coverage of: {topic}\n'
                'Return ONLY {{"results":[{{"title":..,"snippet":..,'
                '"published":"YYYY-MM-DD or null","url":..}}]}} up to 8, recent first.')

_CITE_SYS = "You analyze answer-engine citations. Use web_search. Return ONLY JSON."
_CITE_USER = ('For the topic "{topic}", use web_search to find which sources/outlets are '
              'cited as authorities. Also check whether any of these brand names appear as a '
              'cited source: {aliases}. Note any unrelated well-known entity sharing those '
              'names (a naming collision).\n'
              'Return ONLY {{"answer_text":"<summary naming the sources>",'
              '"cited_sources":["<source/domain>",...],"collision_terms":["<same-name entity>",...]}}')


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2]


class TimingAgent(BaseAgent):
    name = "timing"

    async def run(self, *, topic: str, job_id: str, pillars: list[dict],
                  brand_aliases: list[str], entity: Optional[str] = None) -> TimingSignal:
        # --- 1 web_search call: topic corpus (shared by velocity + gap) ------
        corpus_raw = await self._complete_json(
            system=_SEARCH_SYS, user=_SEARCH_USER.format(topic=topic),
            job_id=job_id, tools=[WEB_SEARCH_TOOL], max_tokens=1800)
        corpus = corpus_raw.get("results", []) if isinstance(corpus_raw, dict) else []

        # velocity: ordinal via the corpus (numeric sources stay unavailable)
        probe = build_default_probe(search_fn=lambda _q: corpus)
        velocity = probe.probe(topic)

        # narrative gap: pillars as topic-generic candidates + an entity-specific one
        candidates = []
        for p in pillars:
            terms = _terms(f"{p.get('id','')} {p.get('desc','')}")
            candidates.append({"angle": p.get("desc") or p.get("id", ""),
                               "gap_type": GapType.topic_generic, "entity": None,
                               "terms": terms or [p.get("id", "x")]})
        if entity:
            candidates.append({"angle": f"{entity}: incident / failure file",
                               "gap_type": GapType.entity_specific, "entity": entity,
                               "terms": ["incident", "fail", "malfunction", "recall", "injury", "safety"]})
        gaps = find_gaps(topic=topic, entity=entity, corpus=corpus, candidates=candidates)

        # --- 1 answer-engine call: citation ---------------------------------
        cite_raw = await self._complete_json(
            system=_CITE_SYS,
            user=_CITE_USER.format(topic=topic, aliases=", ".join(brand_aliases)),
            job_id=job_id, tools=[WEB_SEARCH_TOOL], max_tokens=1200)
        cite_raw = cite_raw if isinstance(cite_raw, dict) else {}
        citation = assess_citation(
            query=topic, brand_aliases=brand_aliases,
            answer_text=cite_raw.get("answer_text", ""),
            cited_sources=cite_raw.get("cited_sources", []),
            collision_terms=cite_raw.get("collision_terms", []))

        # primary-source signal: results that carry a real URL
        primary = [Source(url=r["url"], title=r.get("title", ""))
                   for r in corpus if r.get("url")]
        primary_sig = PrimarySourceSignal(sources=primary[:5], has_primary=bool(primary))

        confidence = round(0.5 * velocity.confidence + 0.5 * (1.0 if corpus else 0.2), 3)
        signal = TimingSignal(velocity=velocity, gaps=gaps, citation=citation,
                              primary_sources=primary_sig, confidence_overall=confidence)
        return await self._validate(TimingSignal, signal.model_dump(), job_id=job_id,
                                    step="timing->fusion")
