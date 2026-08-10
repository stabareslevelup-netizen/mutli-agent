"""
engine/core/resilience.py — retry + circuit breaker for ALL external calls
(social APIs, web_search, and the future GDELT/Wikipedia sources).

- retry_async: exponential backoff with optional jitter.
- CircuitBreaker: after N consecutive failures the circuit OPENS and calls
  short-circuit (raise CircuitOpenError) instead of looping; after a cool-down
  it goes HALF_OPEN and a trial call decides whether to close or re-open.
- resilient_call: composes the two and lands the final failure in dead_letter.

Time is injectable (`time_fn`) so the breaker's cool-down is testable without
sleeping.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Sequence

from engine.core.dead_letter import DeadLetterSink


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the breaker is OPEN."""


@dataclass
class RetryConfig:
    retries: int = 3                 # total attempts
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    stop_on: tuple[type[BaseException], ...] = ()   # never retry these


async def retry_async(fn: Callable[[], Awaitable], config: RetryConfig):
    last: Optional[BaseException] = None
    for attempt in range(1, config.retries + 1):
        try:
            return await fn()
        except config.stop_on:
            raise
        except config.retry_on as exc:
            last = exc
            if attempt >= config.retries:
                break
            delay = min(config.base_delay * (2 ** (attempt - 1)), config.max_delay)
            if config.jitter:
                delay += random.uniform(0, config.base_delay)
            if delay > 0:
                await asyncio.sleep(delay)
    assert last is not None
    raise last


class _State(str, Enum):
    closed = "closed"
    open = "open"
    half_open = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    time_fn: Callable[[], float] = time.monotonic
    name: str = "breaker"

    _state: _State = field(default=_State.closed, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> str:
        return self._state.value

    def _can_attempt(self) -> bool:
        if self._state is _State.open:
            if self.time_fn() - self._opened_at >= self.reset_timeout:
                self._state = _State.half_open
                return True
            return False
        return True

    def _on_success(self) -> None:
        self._failures = 0
        self._state = _State.closed

    def _on_failure(self) -> None:
        self._failures += 1
        if self._state is _State.half_open or self._failures >= self.failure_threshold:
            self._state = _State.open
            self._opened_at = self.time_fn()

    async def call(self, fn: Callable[[], Awaitable]):
        if not self._can_attempt():
            raise CircuitOpenError(f"{self.name} is OPEN")
        try:
            result = await fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result


async def resilient_call(fn: Callable[[], Awaitable], *, breaker: CircuitBreaker,
                         retry: RetryConfig, job_id: Optional[str], step: str,
                         dead_letter_sink: Optional[DeadLetterSink] = None):
    """Retry `fn` through `breaker`; on terminal failure write a dead-letter."""
    # don't keep retrying once the breaker has tripped this call-chain
    cfg = RetryConfig(retries=retry.retries, base_delay=retry.base_delay,
                      max_delay=retry.max_delay, jitter=retry.jitter,
                      retry_on=retry.retry_on,
                      stop_on=retry.stop_on + (CircuitOpenError,))
    try:
        return await retry_async(lambda: breaker.call(fn), cfg)
    except Exception as exc:
        if dead_letter_sink is not None:
            await dead_letter_sink.record(
                job_id=job_id, step=step, error=f"{type(exc).__name__}: {exc}",
                context={"breaker_state": breaker.state},
            )
        raise
