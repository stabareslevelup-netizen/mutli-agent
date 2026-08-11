"""
engine/agents/research.py — Research agent [Phase 2, X-agent migration].

Sweeps primary-source government/patent/paper feeds for stories that haven't
been reported yet. Not a topic lookup — sweep() takes no input, it scans
everything configured and lets one LLM call score/angle the results.

Resilience: a down source (missing API key, request failure) is skipped,
not fatal to the sweep — but logged (logger.warning), not silent, so a
misconfigured key doesn't go unnoticed for days. A malformed item from the
LLM's own output is dead-lettered and dropped, not allowed to halt the
whole sweep — see the per-item validate loop in sweep().
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from typing import Optional

from engine.agents.base import BaseAgent
from engine.core.llm import WEB_SEARCH_TOOL
from engine.core.models import ResearchItem
from engine.core.validation_gate import HandoffHalted
from engine.tools.arxiv_search import search_arxiv
from engine.tools.congress_gov import build_congress_gov_client_from_env
from engine.tools.darpa_solicitations import search_darpa_solicitations
from engine.tools.federal_register import search_federal_register
from engine.tools.sam_gov import build_sam_gov_client_from_env

logger = logging.getLogger(__name__)

_SAM_KEYWORDS = ["artificial intelligence", "autonomous systems", "machine learning",
                "robotics", "unmanned"]
_FED_REGISTER_TERM = "artificial intelligence"
_CONGRESS_QUERY = "artificial intelligence autonomous weapons"
_ARXIV_CATEGORIES = ["cs.RO", "cs.AI"]

_SYSTEM = (
    "You are the Research Agent for an X account covering physical AI, defense tech, "
    "and autonomous systems. Your only job: find stories that haven't been reported yet "
    "-- or that mainstream tech press will cover in 24-72 hours but hasn't touched yet. "
    "You are looking for primary source signals, not news summaries.\n\n"
    "You will be given pre-fetched results from SAM.gov, Federal Register, Congress.gov, "
    "and arXiv directly -- do not re-search those, just score/angle what's given. For "
    "DARPA solicitations, patents, and job-posting signals, use web_search with the "
    "queries provided.\n\n"
    "NOVELTY SCORING GUIDE:\n"
    "9-10: Contract award or patent from a company with no press coverage\n"
    "7-8: DARPA solicitation or Congressional move not yet in tech press\n"
    "5-6: arXiv paper with real-world implications most people missed\n"
    "3-4: Story lightly covered but angle is fresh\n"
    "1-2: Already reported by TechCrunch, Wired, or The Verge\n\n"
    "Only output items with novelty_score 6 or higher. Discard the rest. Do not "
    "editorialize. Do not summarize existing news articles. Do not pull from OpenAI, "
    "Anthropic, or Google product announcements -- that is not this account's lane.\n\n"
    "Return ONLY a JSON array, no prose, no fences. Each element:\n"
    '{"source":"SAM.gov|DARPA|Patent|arXiv|Congress|FedRegister|JobSignal",'
    '"date_found":"<YYYY-MM-DD>","headline":"<one sentence>","raw_detail":"<who, what, '
    'dollar amount or scope>","novelty_score":<1-10>,"post_angle":"<counterintuitive '
    'framing>","source_url":"<direct URL to the primary source>"}'
)

_USER = """Pre-fetched results (score/angle these; do not re-search):
{prefetched}

Also use web_search for:
- DARPA solicitations: {darpa_query}
- Patents (site:patents.google.com): "autonomous weapon", "drone swarm", "physical AI", "humanoid robot", "AI targeting"
- Job signals: any defense/robotics company posting 5+ technical roles simultaneously

Today's date: {today}

Return the JSON array as specified."""


class ResearchAgent(BaseAgent):
    name = "research"

    async def sweep(self, *, job_id: Optional[str]) -> list[ResearchItem]:
        prefetched = await self._gather_prefetched()
        darpa = search_darpa_solicitations()
        raw = await self._complete_json(
            system=_SYSTEM,
            user=_USER.format(prefetched=json.dumps(prefetched, default=str),
                              darpa_query=darpa["query"], today=date.today().isoformat()),
            job_id=job_id or "sweep", tools=[WEB_SEARCH_TOOL], max_tokens=4000)

        candidates = raw if isinstance(raw, list) else (
            raw.get("items", []) if isinstance(raw, dict) else [])
        items: list[ResearchItem] = []
        for c in candidates:
            if isinstance(c, dict) and not c.get("item_id"):
                c["item_id"] = str(uuid.uuid4())
            try:
                item = await self._validate(ResearchItem, c, job_id=job_id or "sweep",
                                            step="research.sweep")
            except HandoffHalted:
                continue   # dead-lettered inside validate(); one bad item can't kill the sweep
            items.append(item)
        return items

    async def _gather_prefetched(self) -> list[dict]:
        hits: list[dict] = []

        sam = build_sam_gov_client_from_env()
        if sam is not None:
            try:
                hits += [{"source": "SAM.gov", **r}
                        for r in await sam.search(keywords=_SAM_KEYWORDS, days_back=1, dept_name="DoD")]
            except Exception as exc:
                logger.warning("research.sweep: %s source failed -- %s: %s",
                               "SAM.gov", type(exc).__name__, exc)

        try:
            hits += [{"source": "FedRegister", **r}
                    for r in await search_federal_register(term=_FED_REGISTER_TERM, days_back=7)]
        except Exception as exc:
            logger.warning("research.sweep: %s source failed -- %s: %s",
                           "FedRegister", type(exc).__name__, exc)

        congress = build_congress_gov_client_from_env()
        if congress is not None:
            try:
                hits += [{"source": "Congress", **r}
                        for r in await congress.search(query=_CONGRESS_QUERY, days_back=7)]
            except Exception as exc:
                logger.warning("research.sweep: %s source failed -- %s: %s",
                               "Congress", type(exc).__name__, exc)

        try:
            hits += [{"source": "arXiv", **r}
                    for r in await search_arxiv(categories=_ARXIV_CATEGORIES, days_back=2)]
        except Exception as exc:
            logger.warning("research.sweep: %s source failed -- %s: %s",
                           "arXiv", type(exc).__name__, exc)

        return hits
