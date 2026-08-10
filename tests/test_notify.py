"""
Notify (Slack) tests — FREE (injected fake transport, no network).

Covers:
  - notify_staged() posts the expected message shape to the webhook URL
  - hedging note appears only when requires_hedging=True
  - any failure (bad status, exception) returns False and never raises —
    this is the property the Orchestrator's "never fail the job" wiring
    depends on (see test_phase6.py for the end-to-end proof)
  - build_notifier_from_env(): graceful degrade to None when unset
"""
from __future__ import annotations

import asyncio
import os

from engine.tools.notify import SlackNotifier, build_notifier_from_env

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


def test_notify_posts_expected_message():
    calls = []

    async def fake_post(url, body):
        calls.append((url, body))
        return 200

    n = SlackNotifier("https://hooks.slack.test/x", post_fn=fake_post,
                      review_base_url="http://localhost:8000")
    ok = asyncio.run(n.notify_staged(job_id="j1", chosen_angle="robots are here",
                                     quality_overall=0.81, requires_hedging=False,
                                     trigger="calendar"))
    check("returns True on 2xx", ok is True)
    check("posted to the configured webhook URL", calls[0][0] == "https://hooks.slack.test/x")
    text = calls[0][1]["text"]
    check("message includes the chosen angle", "robots are here" in text)
    check("message includes the quality score", "0.81" in text)
    check("message includes the trigger", "calendar" in text)
    check("message includes a link back to the job", "j1" in text)
    check("no hedging note when requires_hedging=False", "verification mode" not in text)


def test_notify_includes_hedging_note():
    async def fake_post(url, body):
        return 200
    n = SlackNotifier("https://hooks.slack.test/x", post_fn=fake_post)
    asyncio.run(n.notify_staged(job_id="j2", chosen_angle="a claim", quality_overall=0.7,
                                requires_hedging=True, trigger="manual"))
    # re-run capturing the body this time
    captured = {}
    async def fake_post2(url, body):
        captured.update(body)
        return 200
    n2 = SlackNotifier("https://hooks.slack.test/x", post_fn=fake_post2)
    asyncio.run(n2.notify_staged(job_id="j2", chosen_angle="a claim", quality_overall=0.7,
                                 requires_hedging=True, trigger="manual"))
    check("hedging note present when requires_hedging=True", "verification mode" in captured["text"])


def test_notify_never_raises_on_failure():
    async def failing_post(url, body):
        return 500
    n = SlackNotifier("https://hooks.slack.test/x", post_fn=failing_post)
    ok = asyncio.run(n.notify_staged(job_id="j3", chosen_angle="x", quality_overall=0.5,
                                     requires_hedging=False, trigger="manual"))
    check("non-2xx status -> returns False, does not raise", ok is False)

    async def raising_post(url, body):
        raise ConnectionError("slack is down")
    n2 = SlackNotifier("https://hooks.slack.test/x", post_fn=raising_post)
    raised = False
    ok2 = True
    try:
        ok2 = asyncio.run(n2.notify_staged(job_id="j4", chosen_angle="x", quality_overall=0.5,
                                           requires_hedging=False, trigger="manual"))
    except Exception:
        raised = True
    check("transport exception is swallowed, not raised", raised is False)
    check("transport exception -> returns False", ok2 is False)


def test_build_notifier_from_env_graceful_degrade():
    saved = os.environ.pop("SLACK_WEBHOOK_URL", None)
    try:
        check("no SLACK_WEBHOOK_URL -> None (not an error)", build_notifier_from_env() is None)
        os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.slack.test/y"
        n = build_notifier_from_env()
        check("SLACK_WEBHOOK_URL set -> a real notifier", isinstance(n, SlackNotifier))
    finally:
        if saved is not None:
            os.environ["SLACK_WEBHOOK_URL"] = saved
        else:
            os.environ.pop("SLACK_WEBHOOK_URL", None)


def main() -> int:
    test_notify_posts_expected_message()
    test_notify_includes_hedging_note()
    test_notify_never_raises_on_failure()
    test_build_notifier_from_env_graceful_degrade()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_notify():
    """pytest entrypoint (see pyproject.toml python_functions)."""
    assert main() == 0, "notify suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
