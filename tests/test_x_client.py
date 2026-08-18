"""
X client tests — FREE (injected fake transport, no network, no real X creds).

Covers:
  - OAuth 1.0a header: correct RFC 5849 percent-encoding, deterministic given
    fixed nonce/timestamp, signature changes when the request changes
  - post_thread(): each post after the first replies to the PREVIOUS post's
    real returned id (the actual definition of a thread) — verified even
    though the staged payload only flags the last (link) post explicitly
  - non-2xx response raises, so a failed tweet doesn't silently continue
  - build_x_client_from_env(): graceful degrade to None on incomplete creds
"""
from __future__ import annotations

import asyncio
import os
import urllib.parse

from engine.tools.x_client import (
    XApiClient, XAuthError, XDuplicateContentError, XPostError, XRateLimitError,
    build_x_client_from_env,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


def _client(request_fn=None, nonce="fixednonce", ts=1700000000):
    return XApiClient(consumer_key="ck", consumer_secret="cs",
                      access_token="at", access_token_secret="ats",
                      request_fn=request_fn, nonce_fn=lambda: nonce, timestamp_fn=lambda: ts)


def test_oauth_header_shape():
    c = _client()
    header = c._oauth_header("POST", "https://api.twitter.com/2/tweets")
    check("header starts with OAuth scheme", header.startswith("OAuth "))
    for field in ("oauth_consumer_key", "oauth_nonce", "oauth_signature_method",
                  "oauth_timestamp", "oauth_token", "oauth_version", "oauth_signature"):
        check(f"header includes {field}", f'{field}="' in header)
    check("signature method is HMAC-SHA1", 'oauth_signature_method="HMAC-SHA1"' in header)
    check("token/consumer key values present", 'oauth_consumer_key="ck"' in header and 'oauth_token="at"' in header)


def test_oauth_signature_deterministic_and_sensitive():
    c1 = _client()
    h1a = c1._oauth_header("POST", "https://api.twitter.com/2/tweets")
    h1b = c1._oauth_header("POST", "https://api.twitter.com/2/tweets")
    check("same inputs -> identical signature (fixed nonce/timestamp)", h1a == h1b)

    c2 = _client(nonce="differentnonce")
    h2 = c2._oauth_header("POST", "https://api.twitter.com/2/tweets")
    check("different nonce -> different signature", h1a != h2)

    c3 = _client()
    h3 = c3._oauth_header("POST", "https://api.twitter.com/2/other-path")
    check("different URL -> different signature", h1a != h3)


def test_oauth_percent_encoding_rfc5849():
    # RFC 5849 3.6: only A-Z a-z 0-9 - . _ ~ are left unescaped
    c = _client(nonce="a b&c=d")  # deliberately includes reserved chars
    header = c._oauth_header("POST", "https://api.twitter.com/2/tweets")
    check("reserved chars in a param are percent-encoded, not raw",
          "a b&c=d" not in header and "a%20b%26c%3Dd" in header)


async def _run_post_thread(fail_on_index=None):
    calls = []

    async def fake_request(method, url, headers, json_body):
        calls.append((method, url, dict(json_body or {})))
        if fail_on_index is not None and len(calls) - 1 == fail_on_index:
            return 429, {"errors": [{"message": "rate limited"}]}
        return 201, {"data": {"id": f"tid{len(calls)}"}}

    client = _client(request_fn=fake_request)
    posts = [{"text": "tweet one"}, {"text": "tweet two"}, {"text": "https://example.com/post", "reply_to_prev": True}]
    result = await client.post_thread(posts)
    return result, calls


def test_thread_chains_sequentially():
    result, calls = asyncio.run(_run_post_thread())
    check("post_thread returns one id per post", result["ids"] == ["tid1", "tid2", "tid3"])
    check("exactly 3 API calls made", len(calls) == 3)

    _, _, body0 = calls[0]
    check("first tweet has NO reply field", "reply" not in body0)
    _, _, body1 = calls[1]
    check("second tweet replies to the FIRST tweet's real id",
          body1.get("reply", {}).get("in_reply_to_tweet_id") == "tid1")
    _, _, body2 = calls[2]
    check("third tweet (the link) replies to the SECOND tweet's real id, not the first",
          body2.get("reply", {}).get("in_reply_to_tweet_id") == "tid2")
    check("tweet text passed through unchanged", body0["text"] == "tweet one" and body2["text"] == "https://example.com/post")


def test_failed_post_raises_and_stops_the_thread():
    raised = False
    try:
        asyncio.run(_run_post_thread(fail_on_index=1))
    except XPostError as exc:
        raised = True
        check("error message includes the API status", "429" in str(exc))
        check("a 429 is classified as XRateLimitError specifically", isinstance(exc, XRateLimitError))
    check("a failed post in the middle of a thread raises (doesn't silently continue)", raised)


def test_build_x_client_from_env_graceful_degrade():
    saved = {k: os.environ.pop(k, None) for k in
             ("X_API_KEY", "X_API_KEY_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")}
    try:
        check("no creds set -> None (graceful degrade, not an error)",
              build_x_client_from_env() is None)
        os.environ["X_API_KEY"] = "a"
        os.environ["X_API_KEY_SECRET"] = "b"
        os.environ["X_ACCESS_TOKEN"] = "c"
        # X_ACCESS_TOKEN_SECRET deliberately left unset
        check("partial creds -> still None", build_x_client_from_env() is None)
        os.environ["X_ACCESS_TOKEN_SECRET"] = "d"
        client = build_x_client_from_env()
        check("all four creds set -> a real client", isinstance(client, XApiClient))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def main() -> int:
    test_oauth_header_shape()
    test_oauth_signature_deterministic_and_sensitive()
    test_oauth_percent_encoding_rfc5849()
    test_thread_chains_sequentially()
    test_failed_post_raises_and_stops_the_thread()
    test_build_x_client_from_env_graceful_degrade()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_x_client():
    """pytest entrypoint (see pyproject.toml python_functions)."""
    assert main() == 0, "x_client suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
