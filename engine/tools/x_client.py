"""
engine/tools/x_client.py — real X (Twitter) API v2 client for posting.

Implements the `post_thread(posts) -> dict` contract that `XAdapter.publish()`
already calls (see social_apis.py) — this is a drop-in for the `http` param on
`DistributionAgent.confirm_publish()`. Nothing about the staging/cost logic
changes; this only makes the actual publish step real.

AUTH: X API v2 write endpoints (POST /2/tweets) need OAuth 1.0a USER CONTEXT,
not an app-only bearer token — a bearer token from app-only auth is read-only
for posting purposes. User-context needs four values from the X Developer
Portal (per-app, generated once, no interactive flow needed at request time):
  X_API_KEY, X_API_KEY_SECRET       (consumer key/secret)
  X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET   (user access token/secret, write scope)
This client hand-rolls RFC 5849 OAuth 1.0a HMAC-SHA1 signing (stdlib only —
no new dependency) since the project already uses httpx, not requests.

THREADING: a real X thread requires every tweet after the first to reply to
the tweet immediately before it. The staged `posts` list (see XAdapter.stage)
only flags the final link-reply post with `reply_to_prev` — it does not chain
the thread's own tweets. post_thread() always chains sequentially regardless
of that flag, which is what "thread" actually means on the platform.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, urlencode

API_BASE = "https://api.twitter.com"
POST_TWEET_PATH = "/2/tweets"

# injectable transport: (method, url, headers, json_body) -> (status_code, json)
RequestFn = Callable[[str, str, dict, Optional[dict]], Awaitable[tuple[int, dict]]]


def _percent_encode(s: str) -> str:
    # RFC 5849 3.6: unreserved = A-Z a-z 0-9 - . _ ~
    return quote(str(s), safe="-._~")


async def _httpx_request(method: str, url: str, headers: dict, json_body: Optional[dict]) -> tuple[int, dict]:
    import httpx
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.request(method, url, headers=headers, json=json_body)
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text}
        return resp.status_code, body


class XApiClient:
    """OAuth 1.0a-signed client for POST /2/tweets. Matches the `http` param
    contract `XAdapter.publish()` expects: `await http.post_thread(posts)`."""

    def __init__(self, *, consumer_key: str, consumer_secret: str,
                 access_token: str, access_token_secret: str,
                 base_url: str = API_BASE, request_fn: Optional[RequestFn] = None,
                 nonce_fn: Optional[Callable[[], str]] = None,
                 timestamp_fn: Optional[Callable[[], int]] = None):
        self._ck, self._cs = consumer_key, consumer_secret
        self._at, self._ats = access_token, access_token_secret
        self._base = base_url.rstrip("/")
        self._request = request_fn or _httpx_request
        self._nonce_fn = nonce_fn or (lambda: secrets.token_hex(16))
        self._timestamp_fn = timestamp_fn or (lambda: int(time.time()))

    def _oauth_header(self, method: str, url: str) -> str:
        """Build the OAuth 1.0a Authorization header. For a JSON POST body,
        the signature base string includes only the oauth_* params (X's JSON
        endpoints don't form-encode the body, so it isn't part of the base
        string per RFC 5849 — only query/form params would be)."""
        params = {
            "oauth_consumer_key": self._ck,
            "oauth_nonce": self._nonce_fn(),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(self._timestamp_fn()),
            "oauth_token": self._at,
            "oauth_version": "1.0",
        }
        sorted_params = "&".join(
            f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted(params.items()))
        base_string = "&".join([method.upper(), _percent_encode(url), _percent_encode(sorted_params)])
        signing_key = f"{_percent_encode(self._cs)}&{_percent_encode(self._ats)}"
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        ).decode()
        params["oauth_signature"] = signature
        header_params = ", ".join(
            f'{_percent_encode(k)}="{_percent_encode(v)}"' for k, v in sorted(params.items()))
        return f"OAuth {header_params}"

    async def _post_tweet(self, text: str, in_reply_to: Optional[str] = None) -> str:
        url = f"{self._base}{POST_TWEET_PATH}"
        body: dict[str, Any] = {"text": text}
        if in_reply_to:
            body["reply"] = {"in_reply_to_tweet_id": in_reply_to}
        headers = {"Authorization": self._oauth_header("POST", url),
                  "Content-Type": "application/json"}
        status, resp = await self._request("POST", url, headers, body)
        if status >= 300:
            raise RuntimeError(f"X API POST /2/tweets failed ({status}): {resp}")
        return resp["data"]["id"]

    async def post_thread(self, posts: list[dict]) -> dict:
        """Post each item in `posts` in order, each replying to the previous
        tweet's real ID — this is what makes it an actual thread on X."""
        ids: list[str] = []
        prev_id: Optional[str] = None
        for p in posts:
            tweet_id = await self._post_tweet(p["text"], in_reply_to=prev_id)
            ids.append(tweet_id)
            prev_id = tweet_id
        return {"ids": ids}


def build_x_client_from_env() -> Optional[XApiClient]:
    """Graceful degrade: returns None (not an error) when credentials are
    incomplete, matching build_embedding_provider()'s pattern. The caller
    (main.py) falls back to the existing 'approved, publish pending
    credentials' behavior — the human-confirm gate is unaffected either way."""
    ck = os.getenv("X_API_KEY")
    cs = os.getenv("X_API_KEY_SECRET")
    at = os.getenv("X_ACCESS_TOKEN")
    ats = os.getenv("X_ACCESS_TOKEN_SECRET")
    if not all([ck, cs, at, ats]):
        return None
    return XApiClient(consumer_key=ck, consumer_secret=cs,
                      access_token=at, access_token_secret=ats)
