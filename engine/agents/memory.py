"""
engine/agents/memory.py — Memory query agent [PROVEN read].

Composes episodic + semantic + narrative reads into a single MemoryQueryResult.
Pure retrieval in v1 (no LLM call, no token cost) — the model-tiered reasoning
path is reserved for later phases. Semantic search degrades to recency when
embeddings are unavailable (handled in the memory layer).
"""
from __future__ import annotations

from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.semantic import SemanticMemory
from engine.core.models import MemoryQueryResult


class MemoryAgent:
    name = "memory"

    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory,
                 narrative: NarrativeMemory):
        self._epi = episodic
        self._sem = semantic
        self._nar = narrative

    async def query(self, *, brand_id: str, text: str, k: int = 5) -> MemoryQueryResult:
        return MemoryQueryResult(
            episodic=await self._epi.query(brand_id=brand_id, text=text, k=k),
            semantic=await self._sem.query(brand_id=brand_id, text=text, k=k),
            narrative=await self._nar.constraints(brand_id=brand_id),
        )
