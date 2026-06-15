"""
engine/memory/backend.py — storage seam for the memory layer.

Managers hold policy (degradation, the procedural guard); the backend holds
persistence. InMemoryBackend powers offline tests (cosine in Python);
SqlBackend uses Postgres + pgvector. DI keeps the managers testable without a
live DB — the same pattern as the Phase 2 sinks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredRecord:
    id: int
    kind: str
    brand_id: str
    content: str
    embedding: Optional[list[float]]
    meta: dict
    created_at: datetime
    score: float = 0.0


@dataclass
class StoredPosition:
    id: int
    brand_id: str
    position: str
    stance: str
    active: bool = True


@dataclass
class ProceduralRecord:
    id: int
    brand_id: str
    agent: str
    prompt_text: str
    version: int
    active: bool
    rolled_back: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class ProposalRecord:
    id: int
    brand_id: str
    agent: str
    current_version: int
    proposed_prompt: str
    performance_data: dict
    voice_similarity: Optional[float]
    status: str  # pending | approved | rejected
    created_at: datetime = field(default_factory=_utcnow)


@runtime_checkable
class MemoryBackend(Protocol):
    # episodic/semantic (kind discriminates)
    async def add_record(self, *, kind: str, brand_id: str, content: str,
                         embedding: Optional[list[float]], meta: dict,
                         job_id: Optional[str] = None) -> int: ...
    async def search_records(self, *, kind: str, brand_id: str,
                            query_embedding: list[float], k: int) -> list[StoredRecord]: ...
    async def recent_records(self, *, kind: str, brand_id: str, k: int) -> list[StoredRecord]: ...
    # narrative
    async def add_position(self, *, brand_id: str, position: str, stance: str,
                          embedding: Optional[list[float]]) -> int: ...
    async def active_positions(self, *, brand_id: str) -> list[StoredPosition]: ...
    async def deactivate_position(self, *, position_id: int) -> None: ...
    # procedural
    async def active_procedural(self, *, brand_id: str, agent: str) -> Optional[ProceduralRecord]: ...
    async def procedural_versions(self, *, brand_id: str, agent: str) -> list[ProceduralRecord]: ...
    async def insert_procedural(self, *, brand_id: str, agent: str, prompt_text: str,
                               version: int, active: bool,
                               approved_by: Optional[str] = None) -> ProceduralRecord: ...
    async def set_procedural_flags(self, *, rec_id: int, active: bool,
                                  rolled_back: Optional[bool] = None) -> None: ...
    async def create_proposal(self, *, brand_id: str, agent: str, current_version: int,
                             proposed_prompt: str, performance_data: dict,
                             voice_similarity: Optional[float]) -> ProposalRecord: ...
    async def get_proposal(self, *, proposal_id: int) -> Optional[ProposalRecord]: ...
    async def set_proposal_status(self, *, proposal_id: int, status: str) -> None: ...
    async def list_proposals(self, *, status: Optional[str] = None) -> list[ProposalRecord]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryBackend:
    """Offline backend. Vector search via Python cosine similarity."""

    def __init__(self) -> None:
        self._records: list[StoredRecord] = []
        self._positions: list[StoredPosition] = []
        self._procedural: list[ProceduralRecord] = []
        self._proposals: list[ProposalRecord] = []
        self._seq = 0

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    async def add_record(self, *, kind, brand_id, content, embedding, meta, job_id=None) -> int:
        rid = self._next()
        self._records.append(StoredRecord(rid, kind, brand_id, content, embedding,
                                          meta or {}, _utcnow()))
        return rid

    async def search_records(self, *, kind, brand_id, query_embedding, k) -> list[StoredRecord]:
        scored = []
        for r in self._records:
            if r.kind != kind or r.brand_id != brand_id or r.embedding is None:
                continue
            r.score = _cosine(query_embedding, r.embedding)
            scored.append(r)
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    async def recent_records(self, *, kind, brand_id, k) -> list[StoredRecord]:
        rows = [r for r in self._records if r.kind == kind and r.brand_id == brand_id]
        return list(reversed(rows))[:k]

    async def add_position(self, *, brand_id, position, stance, embedding) -> int:
        pid = self._next()
        self._positions.append(StoredPosition(pid, brand_id, position, stance, True))
        return pid

    async def active_positions(self, *, brand_id) -> list[StoredPosition]:
        return [p for p in self._positions if p.brand_id == brand_id and p.active]

    async def deactivate_position(self, *, position_id) -> None:
        for p in self._positions:
            if p.id == position_id:
                p.active = False

    async def active_procedural(self, *, brand_id, agent) -> Optional[ProceduralRecord]:
        actives = [r for r in self._procedural
                   if r.brand_id == brand_id and r.agent == agent and r.active]
        return max(actives, key=lambda r: r.version) if actives else None

    async def procedural_versions(self, *, brand_id, agent) -> list[ProceduralRecord]:
        rows = [r for r in self._procedural if r.brand_id == brand_id and r.agent == agent]
        return sorted(rows, key=lambda r: r.version)

    async def insert_procedural(self, *, brand_id, agent, prompt_text, version,
                               active, approved_by=None) -> ProceduralRecord:
        rec = ProceduralRecord(self._next(), brand_id, agent, prompt_text, version,
                               active, approved_by=approved_by,
                               approved_at=_utcnow() if approved_by else None)
        self._procedural.append(rec)
        return rec

    async def set_procedural_flags(self, *, rec_id, active, rolled_back=None) -> None:
        for r in self._procedural:
            if r.id == rec_id:
                r.active = active
                if rolled_back is not None:
                    r.rolled_back = rolled_back

    async def create_proposal(self, *, brand_id, agent, current_version,
                             proposed_prompt, performance_data, voice_similarity) -> ProposalRecord:
        rec = ProposalRecord(self._next(), brand_id, agent, current_version,
                             proposed_prompt, performance_data or {},
                             voice_similarity, "pending")
        self._proposals.append(rec)
        return rec

    async def get_proposal(self, *, proposal_id) -> Optional[ProposalRecord]:
        return next((p for p in self._proposals if p.id == proposal_id), None)

    async def set_proposal_status(self, *, proposal_id, status) -> None:
        for p in self._proposals:
            if p.id == proposal_id:
                p.status = status

    async def list_proposals(self, *, status=None) -> list[ProposalRecord]:
        return [p for p in self._proposals if status is None or p.status == status]


class SqlBackend:
    """Postgres + pgvector backend. Vector search via cosine distance."""

    def __init__(self, sessionmaker) -> None:
        self._sm = sessionmaker

    async def add_record(self, *, kind, brand_id, content, embedding, meta, job_id=None) -> int:
        from engine.core import database as db
        model = db.EpisodicMemory if kind == "episodic" else db.SemanticMemory
        async with self._sm() as s:
            kwargs = dict(brand_id=brand_id, content=content, embedding=embedding, meta=meta or {})
            if kind == "episodic":
                kwargs["job_id"] = job_id
            row = model(**kwargs)
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def search_records(self, *, kind, brand_id, query_embedding, k) -> list[StoredRecord]:
        from sqlalchemy import select
        from engine.core import database as db
        model = db.EpisodicMemory if kind == "episodic" else db.SemanticMemory
        async with self._sm() as s:
            dist = model.embedding.cosine_distance(query_embedding)
            stmt = (select(model, dist.label("d"))
                    .where(model.brand_id == brand_id, model.embedding.isnot(None))
                    .order_by(dist).limit(k))
            rows = (await s.execute(stmt)).all()
            return [StoredRecord(r[0].id, kind, brand_id, r[0].content, None,
                                 r[0].meta, r[0].created_at, score=1.0 - float(r[1]))
                    for r in rows]

    async def recent_records(self, *, kind, brand_id, k) -> list[StoredRecord]:
        from sqlalchemy import select
        from engine.core import database as db
        model = db.EpisodicMemory if kind == "episodic" else db.SemanticMemory
        async with self._sm() as s:
            stmt = (select(model).where(model.brand_id == brand_id)
                    .order_by(model.created_at.desc()).limit(k))
            rows = (await s.execute(stmt)).scalars().all()
            return [StoredRecord(r.id, kind, brand_id, r.content, None, r.meta,
                                 r.created_at) for r in rows]

    async def add_position(self, *, brand_id, position, stance, embedding) -> int:
        from engine.core import database as db
        async with self._sm() as s:
            row = db.NarrativeMemory(brand_id=brand_id, position=position,
                                     stance=stance, embedding=embedding, active=True)
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def active_positions(self, *, brand_id) -> list[StoredPosition]:
        from sqlalchemy import select
        from engine.core import database as db
        async with self._sm() as s:
            stmt = select(db.NarrativeMemory).where(
                db.NarrativeMemory.brand_id == brand_id, db.NarrativeMemory.active.is_(True))
            rows = (await s.execute(stmt)).scalars().all()
            return [StoredPosition(r.id, r.brand_id, r.position, r.stance, r.active) for r in rows]

    async def deactivate_position(self, *, position_id) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.NarrativeMemory, position_id)
            if row:
                row.active = False
                await s.commit()

    async def active_procedural(self, *, brand_id, agent) -> Optional[ProceduralRecord]:
        from sqlalchemy import select
        from engine.core import database as db
        async with self._sm() as s:
            stmt = (select(db.ProceduralMemory).where(
                        db.ProceduralMemory.brand_id == brand_id,
                        db.ProceduralMemory.agent_name == agent,
                        db.ProceduralMemory.active.is_(True))
                    .order_by(db.ProceduralMemory.version.desc()).limit(1))
            row = (await s.execute(stmt)).scalars().first()
            return self._proc(row) if row else None

    async def procedural_versions(self, *, brand_id, agent) -> list[ProceduralRecord]:
        from sqlalchemy import select
        from engine.core import database as db
        async with self._sm() as s:
            stmt = (select(db.ProceduralMemory).where(
                        db.ProceduralMemory.brand_id == brand_id,
                        db.ProceduralMemory.agent_name == agent)
                    .order_by(db.ProceduralMemory.version))
            rows = (await s.execute(stmt)).scalars().all()
            return [self._proc(r) for r in rows]

    async def insert_procedural(self, *, brand_id, agent, prompt_text, version,
                               active, approved_by=None) -> ProceduralRecord:
        from engine.core import database as db
        async with self._sm() as s:
            row = db.ProceduralMemory(
                brand_id=brand_id, agent_name=agent, prompt_text=prompt_text,
                version=version, active=active, proposed=False,
                approved_by=approved_by, approved_at=_utcnow() if approved_by else None)
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return self._proc(row)

    async def set_procedural_flags(self, *, rec_id, active, rolled_back=None) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.ProceduralMemory, rec_id)
            if row:
                row.active = active
                if rolled_back is not None:
                    row.rolled_back = rolled_back
                await s.commit()

    async def create_proposal(self, *, brand_id, agent, current_version,
                             proposed_prompt, performance_data, voice_similarity) -> ProposalRecord:
        from engine.core import database as db
        async with self._sm() as s:
            row = db.ProceduralProposal(
                brand_id=brand_id, agent_name=agent, current_version=current_version,
                proposed_prompt=proposed_prompt, performance_data=performance_data or {},
                voice_similarity=voice_similarity, status="pending")
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return self._prop(row)

    async def get_proposal(self, *, proposal_id) -> Optional[ProposalRecord]:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.ProceduralProposal, proposal_id)
            return self._prop(row) if row else None

    async def set_proposal_status(self, *, proposal_id, status) -> None:
        from engine.core import database as db
        async with self._sm() as s:
            row = await s.get(db.ProceduralProposal, proposal_id)
            if row:
                row.status = status
                await s.commit()

    async def list_proposals(self, *, status=None) -> list[ProposalRecord]:
        from sqlalchemy import select
        from engine.core import database as db
        async with self._sm() as s:
            stmt = select(db.ProceduralProposal)
            if status is not None:
                stmt = stmt.where(db.ProceduralProposal.status == status)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._prop(r) for r in rows]

    @staticmethod
    def _proc(r) -> ProceduralRecord:
        return ProceduralRecord(r.id, r.brand_id, r.agent_name, r.prompt_text,
                                r.version, r.active, r.rolled_back, r.approved_by,
                                r.approved_at, r.created_at)

    @staticmethod
    def _prop(r) -> ProposalRecord:
        return ProposalRecord(r.id, r.brand_id, r.agent_name, r.current_version,
                              r.proposed_prompt, r.performance_data, r.voice_similarity,
                              r.status, r.created_at)
