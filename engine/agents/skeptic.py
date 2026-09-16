"""
engine/agents/skeptic.py — Skeptic agent [Phase 2, X-agent migration].

Runs after Strategy, before Copy. Can now actually reject/revise (not just
inform, as the old Skeptic did) -- consumed by the Orchestrator's bounded
retry loop: 'reject' hard-halts immediately, 'revise' gives Strategy one
more pass with the critique attached.

Takes the ResearchItem alongside StrategyOutput -- the spec's own first
hard-rejection criterion ("does the angle overclaim beyond the source")
needs the actual source facts to check against; StrategyOutput alone is
self-contained and has nothing to compare to.
"""
from __future__ import annotations

import json

from engine.agents.base import BaseAgent
from engine.core.models import ContentPillar, ResearchItem, SkepticOutput, StrategyOutput

_SYSTEM = (
    "You are the Skeptic Agent for an X posting pipeline. You adversarially review "
    "Strategy's output before Copy writes anything.\n\n"
    "You are looking for reasons to reject or revise -- not reasons to approve.\n\n"
    "HARD REJECTION (any one = reject):\n"
    "- The angle requires claiming something the source document does not actually say\n"
    "- The post would be indistinguishable from any generic AI news account\n"
    "- The story is already covered by TechCrunch, Wired, The Verge, or Ars Technica in "
    "the last 6 hours\n"
    "- must_include contains no specific number, name, date, or verifiable identifier\n"
    "- The topic is a consumer AI product (ChatGPT, Gemini, Claude, app launches)\n"
    "- The angle is purely opinion with no factual anchor from the source\n\n"
    "REVISION (send back with notes, don't reject):\n"
    "- Format undersells the story (pov_post when this warrants a thread)\n"
    "- The angle buries the most surprising fact -- that fact should be the hook\n"
    "- must_include is missing the most specific verifiable detail from the source\n"
    "- Hashtags are too generic for the specific story\n\n"
    "BANNED WORDS -- flag for revision if any appear in the angle or must_include:\n"
    "revolutionary, groundbreaking, game-changing, transformative, unprecedented, "
    "exciting, powerful, amazing, disruptive (as a positive descriptor)\n\n"
    "Return ONLY a JSON object, no prose, no fences:\n"
    '{"verdict":"approved|revise|reject","critique":"<one paragraph -- what is wrong, '
    'or why it passed>","revised_angle":"<if verdict=revise, a corrected angle, else '
    'null>","revised_must_include":["<if verdict=revise, an updated list, else null>"]}'
)

_USER = """Source item:
headline: {headline}
raw_detail: {raw_detail}
source: {source}
source_url: {source_url}

Strategy's output to review:
chosen_angle: {chosen_angle}
format: {format}
must_include: {must_include}
must_avoid: {must_avoid}
hashtags: {hashtags}

Before verdict: check chosen_angle and every item in must_include against raw_detail
above, line by line. Anything not directly supported by raw_detail is an overclaim --
hard reject on that basis alone.

Return the JSON object as specified."""


class SkepticAgent(BaseAgent):
    name = "skeptic"

    async def review_v2(self, *, item: ResearchItem, strategy: StrategyOutput,
                        job_id: str) -> SkepticOutput:
        user = _USER.format(
            headline=item.headline, raw_detail=item.raw_detail, source=item.source.value,
            source_url=str(item.source_url), chosen_angle=strategy.chosen_angle,
            format=strategy.format.value, must_include=json.dumps(strategy.must_include),
            must_avoid=json.dumps(strategy.must_avoid), hashtags=json.dumps(strategy.hashtags))
        raw = await self._complete_json(system=_SYSTEM, user=user, job_id=job_id, max_tokens=1000)
        out = await self._validate(SkepticOutput, raw, job_id=job_id, step="skeptic.review_v2")

        # code-computed from Strategy's ACTUAL pillar, both directions -- never LLM
        # self-report (same philosophy as strategy.py's narrative_conflict_flag).
        # Measurement only: does not change verdict/the retry loop.
        return out.model_copy(update={"pillar_flag": strategy.pillar == ContentPillar.other})
