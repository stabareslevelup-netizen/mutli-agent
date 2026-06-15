"""
engine/agents/copy_agent.py — Copy agent [PROVEN].

Writes the X thread / IG caption / YouTube script in {{voice}} from the brand
config. Sonnet tier (structured generation). Routes through cost_guard +
validation_gate. Brand-free: voice/formats come from the config at runtime.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.brand_loader import BrandConfig
from engine.core.models import CopyOutput, StrategyPacket

_SYSTEM = ("You are a brand copywriter. Write strictly in this VOICE:\n{voice}\n\n"
           "Never break voice. Return ONLY a JSON object, no prose, no fences.")

_USER = """Chosen angle: {angle}
Rationale: {rationale}
Hard narrative constraints (never contradict these staked positions):
{constraints}

Produce content for these formats: {formats}
Return ONLY:
{{"x_thread":["<tweet 1>","<tweet 2>","..."],
"ig_caption":"<caption>",
"youtube_script":"<short script>"}}"""


# Structured-output schema (no numeric constraints — API-compatible) so the
# shape is guaranteed, not hoped for.
_SCHEMA = {
    "type": "object",
    "properties": {
        "x_thread": {"type": "array", "items": {"type": "string"}},
        "ig_caption": {"type": "string"},
        "youtube_script": {"type": "string"},
    },
    "required": ["x_thread", "ig_caption", "youtube_script"],
    "additionalProperties": False,
}


class CopyAgent(BaseAgent):
    name = "copy"

    async def run(self, *, packet: StrategyPacket, brand: BrandConfig, job_id: str) -> CopyOutput:
        constraints = "\n".join(f"- {c.position}: {c.stance}" for c in packet.hard_constraints) or "- (none)"
        raw = await self._complete_json(
            system=_SYSTEM.format(voice=brand.voice),
            user=_USER.format(angle=packet.chosen_angle, rationale=packet.rationale,
                              constraints=constraints, formats=", ".join(packet.formats)),
            job_id=job_id, max_tokens=2000, output_schema=_SCHEMA)
        return await self._validate(CopyOutput, raw, job_id=job_id, step="copy->quality")
