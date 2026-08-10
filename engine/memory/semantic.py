"""
engine/memory/semantic.py — semantic memory (facts / what engaged).

Same vector-store behavior as episodic; auto-updates safely. Same graceful
degradation (NULL embeddings, recency fallback on query).
"""
from __future__ import annotations

from engine.memory.episodic import VectorMemory


class SemanticMemory(VectorMemory):
    kind = "semantic"
