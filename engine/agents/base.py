"""
engine/agents/base.py — shared agent plumbing.

Every LLM-backed agent call routes through:
  - cost_guard  (correct model tier via model_for(name); token+$ logged)
  - validation_gate (the agent's output is schema-checked before it flows on)
Nothing leaves an agent unvalidated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from engine.core.cost_guard import CostGuard, model_for
from engine.core.llm import LLMClient, extract_json
from engine.core.validation_gate import ValidationGate

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentContext:
    llm: LLMClient
    cost_guard: CostGuard
    gate: ValidationGate
    brand_id: str


class BaseAgent:
    name = "base"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    @property
    def model(self) -> str:
        return model_for(self.name)

    async def _complete_json(self, *, system: str, user: str, job_id: str,
                             tools: Optional[list] = None, max_tokens: int = 2048) -> Any:
        text, usage = await self.ctx.llm.complete(
            model=self.model, system=system, user=user, tools=tools, max_tokens=max_tokens)
        await self.ctx.cost_guard.record(
            job_id=job_id, brand_id=self.ctx.brand_id, agent=self.name,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            model=self.model)
        try:
            return extract_json(text)
        except ValueError:
            return text  # let the gate fail + dead-letter it

    async def _validate(self, model_cls: Type[T], raw: Any, *, job_id: str, step: str) -> T:
        return await self.ctx.gate.validate(model_cls, raw, job_id=job_id, step=step)
