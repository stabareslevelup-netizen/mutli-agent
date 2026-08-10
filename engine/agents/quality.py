"""
engine/agents/quality.py — Quality agent [PROVEN].

Scores voice / narrative / format / hook / coherence in [0,1], routes per
rules, and reports auto-publish eligibility: auto-eligible ONLY at overall
>= 0.85 and route == publish_queue.

(In v1 posting_mode=confirm everything waits for a human anyway; auto_eligible
is what gates the dormant auto/scheduled modes.)

Opus tier. Routes through cost_guard + validation_gate.
"""
from __future__ import annotations

from engine.agents.base import BaseAgent
from engine.core.brand_loader import BrandConfig
from engine.core.models import CopyOutput, QualityRoute, QualityScore

AUTO_PUBLISH_MIN = 0.85
PUBLISH_QUEUE_MIN = 0.60
REJECT_BELOW = 0.40

_SYSTEM = ("You are a strict quality reviewer. Score how well the content matches "
           "this VOICE:\n{voice}\n\nScore each dimension in [0,1]. Return ONLY JSON, no fences.")

_USER = """Hard narrative constraints (must not be contradicted):
{constraints}

Content to score:
X thread: {x}
IG caption: {ig}
YouTube script: {yt}

Return ONLY: {{"voice":0..1,"narrative":0..1,"format":0..1,"hook":0..1,"coherence":0..1,
"reasons":["<short reason>","..."]}}"""


class QualityAgent(BaseAgent):
    name = "quality"

    async def evaluate(self, *, content: CopyOutput, brand: BrandConfig,
                       job_id: str) -> tuple[QualityScore, bool]:
        constraints = "(provided to strategy; assume staked positions hold)"
        schema = {"type": "object",
                  "properties": {k: {"type": "number"} for k in
                                 ("voice", "narrative", "format", "hook", "coherence")}
                  | {"reasons": {"type": "array", "items": {"type": "string"}}},
                  "required": ["voice", "narrative", "format", "hook", "coherence", "reasons"],
                  "additionalProperties": False}
        raw = await self._complete_json(
            system=_SYSTEM.format(voice=brand.voice),
            user=_USER.format(constraints=constraints, x=" | ".join(content.x_thread),
                              ig=content.ig_caption, yt=content.youtube_script[:1200]),
            job_id=job_id, max_tokens=700, output_schema=schema)

        if isinstance(raw, dict):
            dims = {k: float(raw.get(k, 0.0)) for k in ("voice", "narrative", "format", "hook", "coherence")}
            overall = round(sum(dims.values()) / 5, 4)
            if overall < REJECT_BELOW:
                route = QualityRoute.reject
            elif overall < PUBLISH_QUEUE_MIN:
                route = QualityRoute.revise
            else:
                route = QualityRoute.publish_queue
            raw = {**dims, "overall": overall, "route": route.value,
                   "reasons": raw.get("reasons", [])}

        score = await self._validate(QualityScore, raw, job_id=job_id, step="quality->distribution")
        auto_eligible = score.overall >= AUTO_PUBLISH_MIN and score.route == QualityRoute.publish_queue
        return score, auto_eligible
