"""
Phase 2 safety-rail isolation tests. Runnable with plain `python3 -m tests.test_phase2`
(no pytest, no live DB). Each rail is exercised on its own.

Covers the three scenarios the spec calls for:
  1. a malformed handoff is halted and dead-lettered (never passed downstream)
  2. a daily-budget breach halts non-critical generation but allows critical
  3. repeated external failures trip the circuit breaker (then it recovers)
Plus: model tiering and cost math.
"""
from __future__ import annotations

import asyncio

from engine.core.cost_guard import (
    CostGuard, InMemoryCostSink, MODEL_OPUS, MODEL_SONNET, cost_usd, model_for,
)
from engine.core.dead_letter import InMemoryDeadLetterSink
from engine.core.models import VelocitySignal, VelocitySource, VelocityVerdict
from engine.core.resilience import (
    CircuitBreaker, CircuitOpenError, RetryConfig, resilient_call, retry_async,
)
from engine.core.validation_gate import HandoffHalted, ValidationGate

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((PASS if cond else FAIL, name, detail))


# --- 1. validation gate: malformed handoff halts + dead-letters -------------
async def test_validation_gate() -> None:
    sink = InMemoryDeadLetterSink()
    gate = ValidationGate(dead_letter_sink=sink)

    # valid payload passes through
    good = await gate.validate(
        VelocitySignal,
        {"topic": "x", "verdict": "surging", "confidence": 0.9,
         "source": "web_search_ordinal"},
        job_id="job1", step="timing->fusion",
    )
    check("valid handoff returns model", isinstance(good, VelocitySignal))

    # malformed JSON halts
    halted = False
    try:
        await gate.validate(VelocitySignal, "{not json", job_id="job1", step="timing->fusion")
    except HandoffHalted:
        halted = True
    check("malformed JSON halts", halted)

    # schema-invalid (RULE 1 violation: ordinal source + numeric_rate) halts
    halted2 = False
    try:
        await gate.validate(
            VelocitySignal,
            {"topic": "x", "verdict": "surging", "confidence": 0.9,
             "source": "web_search_ordinal", "numeric_rate": 42.0},
            job_id="job1", step="timing->fusion",
        )
    except HandoffHalted:
        halted2 = True
    check("schema-invalid handoff halts", halted2)
    check("two failures dead-lettered", len(sink.records) == 2,
          f"records={len(sink.records)}")
    check("dead-letter carries step", all(r["step"] == "timing->fusion" for r in sink.records))


# --- 2. cost guard: tiering, cost math, budget breach -----------------------
async def test_cost_guard() -> None:
    check("reasoning agent -> Opus", model_for("research") == MODEL_OPUS)
    check("structured agent -> Sonnet", model_for("copy") == MODEL_SONNET)
    # 1M in + 1M out on Opus = $5 + $25 = $30
    check("cost math", cost_usd(MODEL_OPUS, 1_000_000, 1_000_000) == 30.0,
          f"got {cost_usd(MODEL_OPUS, 1_000_000, 1_000_000)}")

    sink = InMemoryCostSink()
    guard = CostGuard(daily_budget_usd=1.0, cost_sink=sink)
    check("starts under budget", guard.can_spend(0.5))

    # burn through the cap: 200k out on Opus = $5 > $1 budget
    await guard.record(job_id="j", brand_id="b", agent="research",
                       input_tokens=0, output_tokens=200_000)
    check("spend logged to cost_log", len(sink.entries) == 1)
    check("over budget after big spend", guard.spent_today() > 1.0,
          f"spent={guard.spent_today()}")
    check("non-critical halts past cap", guard.can_spend(0.0, critical=False) is False)
    check("critical still allowed", guard.can_spend(0.0, critical=True) is True)

    # day rollover resets the running total
    fake_day = ["2026-06-15"]
    import datetime
    g2 = CostGuard(daily_budget_usd=1.0,
                   today_fn=lambda: datetime.date.fromisoformat(fake_day[0]))
    asyncio.get_event_loop()
    await g2.record(job_id="j", brand_id="b", agent="research",
                    input_tokens=0, output_tokens=200_000)
    over = g2.spent_today() > 1.0
    fake_day[0] = "2026-06-16"
    check("budget resets next day", over and g2.spent_today() == 0.0)


# --- 3. circuit breaker: trips after N failures, then recovers --------------
async def test_circuit_breaker() -> None:
    clock = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0,
                             time_fn=lambda: clock[0], name="ext")
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise RuntimeError("downstream down")

    # 3 consecutive failures open the breaker
    for _ in range(3):
        try:
            await breaker.call(always_fails)
        except RuntimeError:
            pass
    check("breaker OPEN after threshold", breaker.state == "open", breaker.state)

    # further calls short-circuit WITHOUT invoking fn (no infinite looping)
    before = calls["n"]
    short = False
    try:
        await breaker.call(always_fails)
    except CircuitOpenError:
        short = True
    check("open breaker short-circuits", short)
    check("fn not invoked while open", calls["n"] == before)

    # after cool-down -> HALF_OPEN -> a success closes it
    clock[0] = 11.0

    async def succeeds():
        return "ok"

    res = await breaker.call(succeeds)
    check("breaker recovers to closed", res == "ok" and breaker.state == "closed",
          breaker.state)


# --- 3b. resilient_call ties retry + breaker + dead-letter together ----------
async def test_resilient_call() -> None:
    sink = InMemoryDeadLetterSink()
    breaker = CircuitBreaker(failure_threshold=10, name="api")
    retry = RetryConfig(retries=3, base_delay=0.0, jitter=False)
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("transient")
        return "recovered"

    out = await resilient_call(flaky, breaker=breaker, retry=retry,
                               job_id="j", step="higgsfield.poll",
                               dead_letter_sink=sink)
    check("retry recovers transient failure", out == "recovered", f"attempts={attempts['n']}")
    check("no dead-letter on eventual success", len(sink.records) == 0)

    async def always():
        raise ConnectionError("hard down")

    failed = False
    try:
        await resilient_call(always, breaker=breaker, retry=RetryConfig(retries=2, base_delay=0.0, jitter=False),
                             job_id="j", step="higgsfield.poll", dead_letter_sink=sink)
    except ConnectionError:
        failed = True
    check("terminal failure raises", failed)
    check("terminal failure dead-lettered", len(sink.records) == 1,
          f"records={len(sink.records)}")


async def main() -> int:
    await test_validation_gate()
    await test_cost_guard()
    await test_circuit_breaker()
    await test_resilient_call()

    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_phase2():
    """pytest entrypoint — runs the suite above (167-check style) and asserts
    real success. Direct-run entrypoint (`python -m tests.test_phase2`) is
    unaffected below."""
    import asyncio
    assert asyncio.run(main()) == 0, "phase 2 suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
