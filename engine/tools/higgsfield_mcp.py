"""
engine/tools/higgsfield_mcp.py — Higgsfield production backend (DI).

Fire ONCE, poll a BOUNDED number of times, return the asset. Renders are never
looped: a single fire, then status polling with a hard attempt cap behind the
circuit breaker. MockProductionBackend powers tests; HiggsfieldMCPBackend is
the production shape (the MCP fire/poll calls are injected — the engine never
hardcodes the transport).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable

from engine.core.resilience import CircuitBreaker, RetryConfig, resilient_call


@dataclass
class RenderResult:
    asset_id: Optional[str]
    asset_url: Optional[str]
    status: str          # ready | failed | pending
    cost_usd: float = 0.0


@runtime_checkable
class ProductionBackend(Protocol):
    name: str
    async def render(self, *, prompt: str, character_element_id: str,
                     job_id: str, kind: str = "video") -> RenderResult: ...


class MockProductionBackend:
    """Deterministic stand-in. Returns a ready asset, one shot, with a
    format-specific render cost. Used when Higgsfield isn't reachable.

    Costs reflect measured/estimated reality: video (seedance_2_0 15s/720p) =
    67.5 credits ~= $3.21; image generation is cheaper (~$0.40 estimate)."""
    name = "mock"

    def __init__(self, video_cost: float = 3.21, image_cost: float = 0.40, status: str = "ready"):
        self.video_cost = video_cost
        self.image_cost = image_cost
        self.status = status
        self.calls = 0

    async def render(self, *, prompt, character_element_id, job_id, kind="video") -> RenderResult:
        self.calls += 1   # tests assert this is exactly 1 (never looped)
        cost = self.image_cost if kind == "image" else self.video_cost
        ext = "png" if kind == "image" else "mp4"
        return RenderResult(
            asset_id=f"mock-{job_id}", status=self.status, cost_usd=cost,
            asset_url=f"https://assets.local/mock/{job_id}.{ext}" if self.status == "ready" else None)


# fire: (prompt, element_id) -> job dict {"id": ...};  poll: (job_id) -> status dict
FireFn = Callable[[str, str], Awaitable[dict]]
PollFn = Callable[[str], Awaitable[dict]]


class HiggsfieldMCPBackend:
    """Production backend over the Higgsfield MCP. fire/poll are injected
    callables that bridge to the MCP tools. Polling is bounded + guarded by a
    circuit breaker; the render is fired exactly once."""
    name = "higgsfield"

    def __init__(self, *, fire_fn: FireFn, poll_fn: PollFn,
                 cost_per_render_usd: float = 0.30, poll_attempts: int = 12,
                 dead_letter_sink=None, breaker: Optional[CircuitBreaker] = None):
        self._fire = fire_fn
        self._poll = poll_fn
        self.cost_usd = cost_per_render_usd
        self._poll_attempts = poll_attempts
        self._dl = dead_letter_sink
        self._breaker = breaker or CircuitBreaker(name="higgsfield")

    async def render(self, *, prompt, character_element_id, job_id, kind="video") -> RenderResult:
        # fire ONCE — no retry on fire to avoid duplicate (paid) renders.
        # `kind` selects the MCP generation (video vs image) in the fire bridge.
        job = await self._fire(prompt, character_element_id)
        hf_id = job.get("id")
        # poll a bounded number of times; each poll is retry/breaker-guarded
        retry = RetryConfig(retries=self._poll_attempts, base_delay=2.0, max_delay=10.0)

        async def poll_until_done():
            status = await self._poll(hf_id)
            state = status.get("status")
            if state in ("ready", "completed", "succeeded"):
                return status
            if state in ("failed", "error"):
                raise RuntimeError(f"higgsfield render failed: {status}")
            raise _StillPending()  # triggers a bounded retry, never an infinite loop

        try:
            final = await resilient_call(
                poll_until_done, breaker=self._breaker, retry=retry,
                job_id=job_id, step="higgsfield.poll", dead_letter_sink=self._dl)
        except Exception:
            return RenderResult(asset_id=hf_id, asset_url=None, status="failed",
                                cost_usd=self.cost_usd)
        return RenderResult(asset_id=hf_id, asset_url=final.get("url") or final.get("asset_url"),
                            status="ready", cost_usd=self.cost_usd)


class _StillPending(Exception):
    pass
