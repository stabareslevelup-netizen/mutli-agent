"""
engine/agents/research.py — Research agent [PROVEN].

Surfaces 3 ranked angles with sources + confidence, using the server-side
web_search tool. Opus tier (reasoning). Output is schema-validated before it
flows to fusion.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.llm import WEB_SEARCH_TOOL
from engine.core.models import ResearchOutput

_SYSTEM = ("You are a sharp research analyst. Use web_search to ground every claim "
           "in real, recent sources. Return ONLY a JSON object — no prose, no fences.")

_USER = """Research this topic and return the 3 strongest, most distinct angles, ranked best-first.

Topic: {topic}

Return ONLY:
{{"angles":[{{"angle":"<one-line angle>","rationale":"<why it matters now>",
"sources":[{{"url":"<url>","title":"<title>"}}],"confidence":<0..1>}}]}}
Exactly 3 angles. Each angle needs >=1 real source from your search."""


class ResearchAgent(BaseAgent):
    name = "research"

    async def run(self, *, topic: str, job_id: str) -> ResearchOutput:
        raw = await self._complete_json(
            system=_SYSTEM, user=_USER.format(topic=topic), job_id=job_id,
            tools=[WEB_SEARCH_TOOL], max_tokens=2500)
        return await self._validate(ResearchOutput, raw, job_id=job_id, step="research->fusion")
