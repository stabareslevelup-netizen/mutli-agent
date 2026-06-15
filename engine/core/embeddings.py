"""
engine/core/embeddings.py — swappable embedding layer (DI, like velocity_probe).

v1 default = Voyage voyage-3 (1024-dim, matches EMBED_DIM). The provider is an
injectable Protocol; swapping to OpenAI / Cohere / a local model is a config
change, never a memory-logic change.

GRACEFUL DEGRADATION: if api.voyageai.com is unreachable (egress allowlist) or
no key is set, `safe_embed` returns None vectors. Memory writes still record
content (embedding column NULL); semantic search falls back to recency until
the host is reachable. Nothing blocks.
"""
from __future__ import annotations

import os
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable


class EmbeddingUnavailable(Exception):
    """The provider cannot produce vectors right now (no key / host blocked)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dimension: int
    def available(self) -> bool: ...
    async def embed(self, texts: list[str]) -> list[Optional[list[float]]]: ...


class NullEmbeddingProvider:
    """Always-available no-op: returns None vectors. The graceful-degrade floor."""
    name = "null"

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def available(self) -> bool:
        return False

    async def embed(self, texts: list[str]) -> list[Optional[list[float]]]:
        return [None] * len(texts)


# An injectable POST hook: payload dict -> response dict. Default uses httpx.
PostFn = Callable[[str, dict, dict], Awaitable[dict]]


async def _httpx_post(url: str, json_body: dict, headers: dict) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=json_body, headers=headers)
        resp.raise_for_status()
        return resp.json()


class VoyageEmbeddingProvider:
    """Voyage voyage-3. Network/host failures raise EmbeddingUnavailable so the
    caller (safe_embed) can degrade rather than crash."""
    name = "voyage"

    def __init__(self, api_key: Optional[str], model: str = "voyage-3",
                 dimension: int = 1024, base_url: str = "https://api.voyageai.com",
                 post_fn: Optional[PostFn] = None):
        self._api_key = api_key
        self.model = model
        self.dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._post = post_fn or _httpx_post

    def available(self) -> bool:
        return bool(self._api_key)

    async def embed(self, texts: list[str]) -> list[Optional[list[float]]]:
        if not texts:
            return []
        if not self.available():
            raise EmbeddingUnavailable("VOYAGE_API_KEY not set")
        try:
            data = await self._post(
                f"{self._base_url}/v1/embeddings",
                {"input": texts, "model": self.model},
                {"Authorization": f"Bearer {self._api_key}",
                 "Content-Type": "application/json"},
            )
        except Exception as exc:  # connect/timeout/host-blocked/5xx
            raise EmbeddingUnavailable(f"voyage request failed: {exc}") from exc
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [it.get("embedding") for it in items]


def build_embedding_provider(post_fn: Optional[PostFn] = None) -> EmbeddingProvider:
    """Default factory: Voyage if a key is present, else the Null fallback."""
    dim = int(os.getenv("EMBED_DIM", "1024"))
    key = os.getenv("VOYAGE_API_KEY")
    if key:
        return VoyageEmbeddingProvider(api_key=key, dimension=dim, post_fn=post_fn)
    return NullEmbeddingProvider(dimension=dim)


async def safe_embed(provider: EmbeddingProvider, texts: list[str]) -> list[Optional[list[float]]]:
    """Never raises. On any provider failure, returns None vectors (degrade)."""
    if not texts:
        return []
    try:
        return await provider.embed(texts)
    except EmbeddingUnavailable:
        return [None] * len(texts)
    except Exception:
        return [None] * len(texts)
