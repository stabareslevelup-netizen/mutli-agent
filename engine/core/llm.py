"""
engine/core/llm.py — thin async wrapper over the Anthropic SDK.

Tier-1 reasoning agents (Research, Timing) call Claude with the server-side
`web_search` tool. This wrapper returns (text, Usage) so the caller can log
token cost through cost_guard, handles the server-tool `pause_turn` resume
loop, and ships a tolerant JSON extractor (models wrap JSON in prose/fences).

The SDK auto-retries 429/5xx with backoff (max_retries) — that's the LLM
resilience path; external non-LLM calls use engine/core/resilience.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# Server-side web search tool (dynamic filtering version; Opus 4.8 supported).
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    def __init__(self, client: Any = None, max_retries: int = 3):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(max_retries=max_retries)
        self._client = client

    async def complete(self, *, model: str, system: str, user: str,
                       tools: Optional[list] = None, max_tokens: int = 2048,
                       effort: str = "low", max_continuations: int = 3) -> tuple[str, Usage]:
        messages: list[dict] = [{"role": "user", "content": user}]
        usage = Usage()
        text = ""
        for _ in range(max_continuations + 1):
            kwargs: dict[str, Any] = dict(
                model=model, max_tokens=max_tokens, system=system,
                messages=messages, output_config={"effort": effort})
            if tools:
                kwargs["tools"] = tools
            resp = await self._client.messages.create(**kwargs)
            usage.input_tokens += resp.usage.input_tokens
            usage.output_tokens += resp.usage.output_tokens
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text")
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            break
        return text, usage


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model output (handles ```json fences + preamble)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = (fence.group(1) if fence else text).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = candidate.find(open_c)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == open_c:
                depth += 1
            elif candidate[i] == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    raise ValueError("no JSON found in model output")
