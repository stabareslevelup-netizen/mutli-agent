"""
engine/memory/narrative.py — narrative memory: the brand's STAKED POSITIONS.

These are a hard constraint for the Strategy agent: it must never contradict an
active staked position. Auto-updates are safe (recording a position is not
self-modification of prompts). Embeddings are best-effort (NULL on degrade) and
only used for future similarity surfacing; the constraints read path does not
depend on them.
"""
from __future__ import annotations

from engine.core.embeddings import EmbeddingProvider, safe_embed
from engine.core.models import NarrativeConstraint
from engine.memory.backend import MemoryBackend


class NarrativeMemory:
    def __init__(self, backend: MemoryBackend, embeddings: EmbeddingProvider):
        self._backend = backend
        self._emb = embeddings

    async def stake_position(self, *, brand_id: str, position: str, stance: str) -> int:
        vec = (await safe_embed(self._emb, [f"{position} :: {stance}"]))[0]
        return await self._backend.add_position(
            brand_id=brand_id, position=position, stance=stance, embedding=vec)

    async def constraints(self, *, brand_id: str) -> list[NarrativeConstraint]:
        rows = await self._backend.active_positions(brand_id=brand_id)
        return [NarrativeConstraint(position=r.position, stance=r.stance, ref_id=str(r.id))
                for r in rows]

    async def retire_position(self, *, position_id: int) -> None:
        await self._backend.deactivate_position(position_id=position_id)
