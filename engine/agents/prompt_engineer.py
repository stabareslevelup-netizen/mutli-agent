"""
engine/agents/prompt_engineer.py — Prompt Engineer agent [PROVEN].

Writes a Higgsfield video/image prompt from the {{character}} config: the
backend rewrites the `<<<element_id>>>` placeholder into the reference
character, so the prompt MUST embed it. Sonnet tier. Routes through cost_guard
+ validation_gate. Brand-free: character/palette/style come from config.

Guarantee: the placeholder is always present in the final prompt (injected if
the model omits it), so the PromptEngineerOutput validator never trips on a
model slip.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.brand_loader import BrandConfig
from engine.core.models import PromptEngineerOutput, StrategyPacket

_SYSTEM = ("You are a cinematic prompt engineer for an image/video model. "
           "Write ONE vivid prompt. You MUST include the exact token {placeholder} "
           "verbatim where the character should appear. Return ONLY JSON, no fences.")

_USER = """Angle to depict: {angle}

Character token (use verbatim): {placeholder}
Visual style: {style}
Palette: {palette}
Visual encoding cues: {encoding}

Return ONLY: {{"higgsfield_prompt":"<one cinematic prompt that contains the token verbatim>"}}"""


class PromptEngineerAgent(BaseAgent):
    name = "prompt_engineer"

    async def run(self, *, packet: StrategyPacket, brand: BrandConfig, job_id: str) -> PromptEngineerOutput:
        ch = brand.character
        schema = {"type": "object",
                  "properties": {"higgsfield_prompt": {"type": "string"}},
                  "required": ["higgsfield_prompt"], "additionalProperties": False}
        raw = await self._complete_json(
            system=_SYSTEM.format(placeholder=ch.placeholder),
            user=_USER.format(angle=packet.chosen_angle, placeholder=ch.placeholder,
                              style=ch.style, palette=", ".join(ch.palette),
                              encoding=ch.visual_encoding),
            job_id=job_id, max_tokens=900, output_schema=schema)

        # Guarantee the placeholder is embedded even if the model dropped it.
        if isinstance(raw, dict):
            prompt = str(raw.get("higgsfield_prompt", "")).strip()
            if ch.placeholder not in prompt:
                prompt = f"{ch.placeholder} — {prompt}" if prompt else ch.placeholder
            raw = {"higgsfield_prompt": prompt, "character_placeholder": ch.placeholder}
        return await self._validate(PromptEngineerOutput, raw, job_id=job_id,
                                    step="prompt_engineer->production")
