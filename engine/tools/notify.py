"""
engine/tools/notify.py — "something's waiting for you" alert.

The dashboard is pull (you have to check it); this is push, so the human stays
on the loop without babysitting the queue. Fires whenever a job reaches
staged_for_review — manual or cron-triggered — since either way a human still
has to tap Approve before anything publishes; this alert never bypasses that
gate, it only tells you the gate exists to be tapped.

Best-effort by design: a failed notification must never fail the job it's
about. One attempt, caught, logged — no retry/circuit-breaker machinery here
(unlike the platform adapters), because there's nothing to "resume" if a Slack
ping is dropped; the job already succeeded and is sitting in the queue either way.
"""
from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Optional

PostFn = Callable[[str, dict], Awaitable[int]]  # (url, json_body) -> status_code


async def _httpx_post(url: str, json_body: dict) -> int:
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=json_body)
        return resp.status_code


class SlackNotifier:
    def __init__(self, webhook_url: str, *, post_fn: Optional[PostFn] = None,
                 review_base_url: str = ""):
        self._url = webhook_url
        self._post = post_fn or _httpx_post
        self._review_base_url = review_base_url.rstrip("/")

    async def notify_staged(self, *, job_id: str, chosen_angle: str,
                            quality_overall: float, requires_hedging: bool,
                            trigger: str) -> bool:
        """Returns True if the notification was sent, False on any failure —
        never raises, so a Slack outage can't take down a job."""
        link = f"{self._review_base_url}/#{job_id}" if self._review_base_url else job_id
        hedge_note = " · verification mode (claims hedged)" if requires_hedging else ""
        text = (f":clipboard: New piece staged for review ({trigger} trigger)\n"
                f"*{chosen_angle}*\n"
                f"quality {quality_overall:.2f}{hedge_note} — {link}")
        try:
            status = await self._post(self._url, {"text": text})
            return status < 300
        except Exception:
            return False


def build_notifier_from_env() -> Optional[SlackNotifier]:
    """Graceful degrade: None when SLACK_WEBHOOK_URL isn't set — the
    orchestrator simply skips notifying, nothing else changes."""
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return None
    return SlackNotifier(url, review_base_url=os.getenv("REVIEW_DASHBOARD_URL", ""))
