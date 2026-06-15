"""
engine/agents/production.py — Production agent [PROVEN].

Fires the Higgsfield render via the injected backend (fire once, bounded poll,
never loop), logs the render cost through cost_guard.record_external, and
returns a schema-validated ProductionResult. The backend choice (mock vs MCP)
is injected, so this agent is identical in tests and production.
"""
from __future__ import annotations

from engine.core.cost_guard import CostGuard
from engine.core.models import ProductionResult, PromptEngineerOutput
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

    async def run(self, *, prompt: PromptEngineerOutput, character_element_id: str,
                  job_id: str) -> ProductionResult:
        result = await self._backend.render(
            prompt=prompt.higgsfield_prompt, character_element_id=character_element_id,
            job_id=job_id)
        # render cost is a flat external charge (credits), not token-based
        await self._cost.record_external(
            job_id=job_id, brand_id=self._brand_id, agent=self.name,
            usd=result.cost_usd, label=self._backend.name)
        payload = {"asset_id": result.asset_id, "asset_url": result.asset_url,
                   "status": result.status}
        return await self._gate.validate(ProductionResult, payload, job_id=job_id,
                                         step="production->quality")
