"""
engine/core/cost_guard.py — model tiering, per-job cost logging, daily cap.

Model tiering (SPEC): reasoning agents (Research, Strategy, Quality, Memory,
Timing) run on Opus 4.8; structured/formatting agents (Copy, Prompt Engineer,
Production control, Distribution) run on Sonnet 4.6. Every job logs token cost
to cost_log; once the brand's daily budget is exceeded, non-critical
generation halts while critical work may proceed.

Pricing is authoritative as of 2026-06 (USD per 1M tokens) and overridable.
The cost sink is injectable so this is testable without a live DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Protocol, runtime_checkable

# --- model tiering ----------------------------------------------------------
MODEL_OPUS = "claude-opus-4-8"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5"

REASONING_AGENTS = {"research", "strategy", "quality", "memory", "timing"}
STRUCTURED_AGENTS = {"copy", "prompt_engineer", "production", "distribution"}


def model_for(agent: str) -> str:
    """Reasoning agents → Opus; everything else (structured/formatting) → Sonnet."""
    return MODEL_OPUS if agent.lower() in REASONING_AGENTS else MODEL_SONNET


# --- pricing (USD per 1,000,000 tokens): (input, output) --------------------
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    MODEL_OPUS: (5.00, 25.00),
    MODEL_SONNET: (3.00, 15.00),
    MODEL_HAIKU: (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             pricing: Optional[dict[str, tuple[float, float]]] = None) -> float:
    table = pricing or DEFAULT_PRICING
    in_rate, out_rate = table.get(model, (0.0, 0.0))
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


# --- cost sink --------------------------------------------------------------
@runtime_checkable
class CostSink(Protocol):
    async def record(self, *, job_id: Optional[str], brand_id: str, agent: str,
                     model: str, input_tokens: int, output_tokens: int,
                     usd: float) -> None: ...


class InMemoryCostSink:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    async def record(self, **kw) -> None:
        self.entries.append(kw)


class SqlCostSink:
    def __init__(self, sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def record(self, *, job_id, brand_id, agent, model,
                     input_tokens, output_tokens, usd) -> None:
        from engine.core.database import CostLog
        async with self._sessionmaker() as session:
            session.add(CostLog(job_id=job_id, brand_id=brand_id, agent=agent,
                                model=model, input_tokens=input_tokens,
                                output_tokens=output_tokens, usd=usd))
            await session.commit()


@dataclass
class CostEntry:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    usd: float


class BudgetExceeded(Exception):
    pass


@dataclass
class CostGuard:
    daily_budget_usd: float
    cost_sink: Optional[CostSink] = None
    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    today_fn: Callable[[], date] = date.today

    _spent: float = field(default=0.0, init=False)
    _day: Optional[date] = field(default=None, init=False)

    def _roll(self) -> None:
        today = self.today_fn()
        if self._day != today:
            self._day = today
            self._spent = 0.0

    def spent_today(self) -> float:
        self._roll()
        return round(self._spent, 6)

    def remaining(self) -> float:
        return round(self.daily_budget_usd - self.spent_today(), 6)

    def can_spend(self, estimated_usd: float = 0.0, *, critical: bool = False) -> bool:
        """Critical generation is always allowed; non-critical halts past the cap."""
        if critical:
            return True
        return (self.spent_today() + estimated_usd) <= self.daily_budget_usd

    async def record(self, *, job_id: Optional[str], brand_id: str, agent: str,
                     input_tokens: int, output_tokens: int,
                     model: Optional[str] = None) -> CostEntry:
        self._roll()
        model = model or model_for(agent)
        usd = cost_usd(model, input_tokens, output_tokens, self.pricing)
        self._spent += usd
        if self.cost_sink is not None:
            await self.cost_sink.record(
                job_id=job_id, brand_id=brand_id, agent=agent, model=model,
                input_tokens=input_tokens, output_tokens=output_tokens, usd=usd,
            )
        return CostEntry(agent, model, input_tokens, output_tokens, usd)

    async def record_external(self, *, job_id: Optional[str], brand_id: str, agent: str,
                              usd: float, label: str) -> CostEntry:
        """Log a non-token external cost (Higgsfield render, per-post X fee).
        `label` stands in for the model column (e.g. 'higgsfield', 'x-api')."""
        self._roll()
        self._spent += usd
        if self.cost_sink is not None:
            await self.cost_sink.record(
                job_id=job_id, brand_id=brand_id, agent=agent, model=label,
                input_tokens=0, output_tokens=0, usd=round(usd, 6),
            )
        return CostEntry(agent, label, 0, 0, round(usd, 6))
