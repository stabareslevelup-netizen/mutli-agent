"""
engine/agents/prompt_engineer.py — Prompt Engineer agent [PROVEN].

Two rules, both required on every render (Fix 3) — stated generically so the
engine stays brand-free; the character identity is injected from the config:

  RULE A — the brand CHARACTER is ALWAYS the subject. The character element
    placeholder <<<...>>> is always present; no other character is generated.
    The character is the visual anchor of the brand.
  RULE B — the SCENE MATCHES THE ARTICLE. The agent receives the chosen angle
    and the matched pillar's visual mood (from config) and must build a scene
    that is visually SPECIFIC to that story — never a generic void or default
    environment. Pillar drives the mood; the angle drives the concrete scene.

Brand palette/aesthetic always applies; IP-safety stays (no real brands/logos/
products). Sonnet tier. Placeholder is force-injected if the model omits it.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.brand_loader import BrandConfig
from engine.core.models import PromptEngineerOutput, StrategyPacket

_SYSTEM = ("You are a cinematic prompt engineer for an image/video model.\n"
           "RULE A: the character is ALWAYS {character} — include the exact token "
           "{placeholder} verbatim where the character appears. Never invent another "
           "character; {character} is the brand's visual anchor.\n"
           "RULE B: build a scene that is visually SPECIFIC to the story you are given — "
           "never a generic void or default room. The pillar mood sets the atmosphere; "
           "the angle sets the concrete scene.\n"
           "IP SAFETY: no real-world company names, brands, products, named robots/vehicles, "
           "logos, trademarks, or identifiable real people; abstract the ENVIRONMENT, keep "
           "the character vivid. Always apply the brand palette/aesthetic. Return ONLY JSON, no fences.")

_USER = """Story angle (drives the specific scene): {angle}
Pillar visual mood (drives atmosphere): {mood}

Character: {character}   token (use verbatim): {placeholder}
Brand visual style: {style}
Palette: {palette}
Visual encoding cues: {encoding}

Write ONE cinematic prompt: {character} ({placeholder}) inside a scene that concretely reflects
this story's world (not a void), in the pillar's mood, brand palette, dark editorial, cinematic.
Return ONLY: {{"higgsfield_prompt":"<one prompt containing the token verbatim>"}}"""

_SCHEMA = {"type": "object", "properties": {"higgsfield_prompt": {"type": "string"}},
           "required": ["higgsfield_prompt"], "additionalProperties": False}


class PromptEngineerAgent(BaseAgent):
    name = "prompt_engineer"

    async def run(self, *, packet: StrategyPacket, brand: BrandConfig, job_id: str) -> PromptEngineerOutput:
        ch = brand.character
        # RULE B: look up the matched pillar's visual mood (config-driven, brand-free engine)
        mood = next((p.visual_mood for p in brand.pillars if p.id == packet.pillar_id), "")
        if not mood:
            mood = "dark editorial, cinematic, the character embodied in its environment"

        raw = await self._complete_json(
            system=_SYSTEM.format(character=ch.name, placeholder=ch.placeholder),
            user=_USER.format(angle=packet.chosen_angle, mood=mood, character=ch.name,
                              placeholder=ch.placeholder, style=ch.style,
                              palette=", ".join(ch.palette), encoding=ch.visual_encoding),
            job_id=job_id, max_tokens=900, output_schema=_SCHEMA)

        # RULE A guarantee: the placeholder is always embedded even if the model slipped.
        if isinstance(raw, dict):
            prompt = str(raw.get("higgsfield_prompt", "")).strip()
            if ch.placeholder not in prompt:
                prompt = f"{ch.placeholder} — {prompt}" if prompt else ch.placeholder
            raw = {"higgsfield_prompt": prompt, "character_placeholder": ch.placeholder}
        return await self._validate(PromptEngineerOutput, raw, job_id=job_id,
                                    step="prompt_engineer->production")
