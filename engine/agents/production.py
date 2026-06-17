"""
engine/agents/production.py — Production agent [PROVEN].

Format-aware (Phase 8):
  - video / image: fire the Higgsfield render via the injected backend (fire
    once, bounded poll, never loop), passing `kind`.
  - text_only: skip rendering entirely (no visual).

Always logs a cost_log line tagged with the format ("<backend>:<format>") so we
can track which format runs most — text_only logs a $0 line for counting.
"""
from __future__ import annotations

from engine.core.cost_guard import CostGuard
from engine.core.models import ContentFormat, ProductionResult, PromptEngineerOutput
from engine.core.validation_gate import ValidationGate
from engine.tools.higgsfield_mcp import ProductionBackend


class ProductionAgent:
    name = "production"

    def __init__(self, backend: ProductionBackend, cost_guard: CostGuard,
                 gate: ValidationGate, brand_id: str):
        self._backend = backend
        self._cost = cost_guard
        self._gate = gate
        self._brand_id = brand_id

    async def run(self, *, prompt: PromptEngineerOutput | None, character_element_id: str,
                  job_id: str, content_format: ContentFormat = ContentFormat.video) -> ProductionResult:
        label = f"{self._backend.name}:{content_format.value}"   # format tracked in cost_log

        if content_format == ContentFormat.text_only:
            await self._cost.record_external(job_id=job_id, brand_id=self._brand_id,
                                             agent=self.name, usd=0.0, label=label)
            return await self._gate.validate(
                ProductionResult, {"asset_id": None, "asset_url": None, "status": "skipped"},
                job_id=job_id, step="production->quality")

        result = await self._backend.render(
            prompt=prompt.higgsfield_prompt, character_element_id=character_element_id,
            job_id=job_id, kind=content_format.value)
        await self._cost.record_external(job_id=job_id, brand_id=self._brand_id,
                                         agent=self.name, usd=result.cost_usd, label=label)
        payload = {"asset_id": result.asset_id, "asset_url": result.asset_url,
                   "status": result.status}
        return await self._gate.validate(ProductionResult, payload, job_id=job_id,
                                         step="production->quality")
