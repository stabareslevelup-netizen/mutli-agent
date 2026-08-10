"""
engine/memory/episodic.py — vector memory base + episodic memory.

Episodic (what happened) and semantic (what engaged / facts) memory share the
same vector-store shape. Both AUTO-UPDATE — safe per spec (only procedural
writes are guarded).

GRACEFUL DEGRADATION: writes always record content; the embedding is NULL when
the provider is unavailable. Queries use vector similarity when an embedding is
available, else fall back to recency. `degraded` reports which path ran.
"""
from __future__ import annotations

from typing import Optional

from engine.core.embeddings import EmbeddingProvider, safe_embed
from engine.core.models import MemoryItem
from engine.memory.backend import MemoryBackend


class VectorMemory:
    kind = "episodic"

    def __init__(self, backend: MemoryBackend, embeddings: EmbeddingProvider):
        self._backend = backend
        self._emb = embeddings
        self.degraded = False   # True when the last query fell back to recency

    async def record(self, *, brand_id: str, content: str,
                     meta: Optional[dict] = None, job_id: Optional[str] = None) -> int:
        vec = (await safe_embed(self._emb, [content]))[0]
        return await self._backend.add_record(
            kind=self.kind, brand_id=brand_id, content=content,
            embedding=vec, meta=meta or {}, job_id=job_id)

    async def query(self, *, brand_id: str, text: str, k: int = 5) -> list[MemoryItem]:
        qvec = (await safe_embed(self._emb, [text]))[0]
        if qvec is None:
            self.degraded = True
            rows = await self._backend.recent_records(kind=self.kind, brand_id=brand_id, k=k)
        else:
            self.degraded = False
            rows = await self._backend.search_records(
                kind=self.kind, brand_id=brand_id, query_embedding=qvec, k=k)
        return [MemoryItem(kind=self.kind, content=r.content, score=round(r.score, 4),
                           ref_id=str(r.id)) for r in rows]


class EpisodicMemory(VectorMemory):
    kind = "episodic"
