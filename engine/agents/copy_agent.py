"""
engine/agents/copy_agent.py — Copy agent [PROVEN].

Writes the X thread / IG caption / YouTube script in {{voice}}.

Fix 1 — VERIFICATION GATE (non-negotiable): the brand positioning is "I find
the mistake before it becomes the headline." When the StrategyPacket flags
`requires_hedging` (citation ambiguous OR velocity confidence < 0.6), EVERY
factual claim must be hedged ("reportedly", "per the announcement", "if
confirmed", "according to the company") — never a declarative assertion
presented as verified fact. Enforced two ways: a permanent system-prompt rule
AND a pre-copy check that inspects the packet and injects an explicit
verification directive before generation. The Skeptic's summary is always
passed in so the copy addresses the weak point.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.brand_loader import BrandConfig
from engine.core.models import CopyOutput, StrategyPacket

_SYSTEM = ("You are a brand copywriter. Write strictly in this VOICE:\n{voice}\n\n"
           "POSITIONING: this brand finds the mistake before it becomes the headline — "
           "a careful analyst, never a hype account. VERIFICATION RULE: when asked to "
           "operate in verification mode, hedge EVERY factual claim with framing like "
           "'reportedly', 'per the announcement', 'if confirmed', or 'according to the "
           "company' — never present an unverified claim as established fact. "
           "Return ONLY a JSON object, no prose, no fences.")

_USER = """Chosen angle: {angle}
Rationale: {rationale}
Skeptic's caution (address this, don't ignore it): {skeptic}
Hard narrative constraints (never contradict these staked positions):
{constraints}
{verification}
Produce content for these formats: {formats}
Return ONLY:
{{"x_thread":["<tweet 1>","<tweet 2>","..."],
"ig_caption":"<caption>",
"youtube_script":"<short script>"}}"""

_VERIFICATION_ON = (
    "\n*** VERIFICATION MODE (citation={citation}, velocity_confidence={conf}): "
    "the underlying claim is NOT independently verified. Hedge EVERY factual assertion "
    "('reportedly', 'per the announcement', 'if confirmed', 'according to the company'). "
    "Do NOT state unverified specifics as fact. Read like a careful analyst. ***\n")

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
        # pre-copy check: inspect the packet and turn on the verification directive
        verification = ""
        if packet.requires_hedging:
            verification = _VERIFICATION_ON.format(
                citation=packet.citation_status or "unverified",
                conf=round(packet.velocity_confidence, 2))
        raw = await self._complete_json(
            system=_SYSTEM.format(voice=brand.voice),
            user=_USER.format(angle=packet.chosen_angle, rationale=packet.rationale,
                              skeptic=packet.skeptic_summary or "(none)",
                              constraints=constraints, verification=verification,
                              formats=", ".join(packet.formats)),
            job_id=job_id, max_tokens=2000, output_schema=_SCHEMA)
        return await self._validate(CopyOutput, raw, job_id=job_id, step="copy->quality")
