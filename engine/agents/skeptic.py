"""
engine/agents/skeptic.py — Skeptic agent (the 11th agent) [Tier 2].

Runs after Strategy, before Copy. Adversarially reviews the chosen angle and
answers four questions: (1) what an insider would dispute, (2) hidden/unstated
assumptions, (3) alternative explanations, (4) is scale/impact overstated.

Emits a 2-3 sentence `skeptic_summary` and a `confidence_adjustment`
(-0.1..0.0 — never raises confidence). It does NOT block the job; it informs
Copy's hedging. Sonnet tier, one LLM call, structured output enforced.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.models import SkepticReview, StrategyPacket

_SYSTEM = ("You are a skeptical robotics/industry analyst doing adversarial review. "
           "You do not hype. You find the weak point before it becomes the headline. "
           "Be specific and fair. Return ONLY a JSON object, no fences.")

_USER = """Adversarially review this chosen angle. Answer all four:
1) What would a robotics engineer or industry insider DISPUTE about this claim?
2) What assumptions are hidden or unstated?
3) What alternative explanations exist for the same facts?
4) Is the scale or impact being overstated?

Angle: {angle}
Rationale: {rationale}

Return ONLY:
{{"disputes":"...","hidden_assumptions":"...","alternative_explanations":"...",
"overstatement":"...","skeptic_summary":"<2-3 sentences, the key caution>",
"confidence_adjustment":<number between -0.1 and 0.0; 0.0 if the angle is solid>}}"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "disputes": {"type": "string"}, "hidden_assumptions": {"type": "string"},
        "alternative_explanations": {"type": "string"}, "overstatement": {"type": "string"},
        "skeptic_summary": {"type": "string"}, "confidence_adjustment": {"type": "number"},
    },
    "required": ["disputes", "hidden_assumptions", "alternative_explanations",
                 "overstatement", "skeptic_summary", "confidence_adjustment"],
    "additionalProperties": False,
}


class SkepticAgent(BaseAgent):
    name = "skeptic"   # not in REASONING_AGENTS -> Sonnet tier

    async def review(self, *, packet: StrategyPacket, job_id: str) -> SkepticReview:
        raw = await self._complete_json(
            system=_SYSTEM,
            user=_USER.format(angle=packet.chosen_angle, rationale=packet.rationale),
            job_id=job_id, max_tokens=600, output_schema=_SCHEMA)
        if isinstance(raw, dict) and "confidence_adjustment" in raw:
            # clamp to the allowed band — the skeptic never raises confidence
            try:
                raw["confidence_adjustment"] = max(-0.1, min(0.0, float(raw["confidence_adjustment"])))
            except (TypeError, ValueError):
                raw["confidence_adjustment"] = 0.0
        return await self._validate(SkepticReview, raw, job_id=job_id, step="skeptic->copy")
