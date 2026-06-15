"""
engine/core/database.py — PostgreSQL + pgvector schema and async access.

Brand-agnostic. All tables from SPEC_v3 "DATABASE SCHEMA", including the v3
additions: procedural_memory versioning, cost_log, dead_letter,
procedural_proposals.

No live DB is needed to inspect the schema:
    python -m engine.core.database --print-ddl
emits the full CREATE TABLE DDL (PostgreSQL dialect) so the schema can be
reviewed/validated before a database exists.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Embedding dimension for pgvector columns (Voyage voyage-3 = 1024 by default).
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    current_tier: Mapped[int] = mapped_column(Integer, default=0)
    trigger: Mapped[str] = mapped_column(String(32), default="manual")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# --------------------------------------------------------------------------
# Memory tables
# --------------------------------------------------------------------------
class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SemanticMemory(Base):
    __tablename__ = "semantic_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class NarrativeMemory(Base):
    """Staked positions. Strategy must never contradict these (hard constraint)."""
    __tablename__ = "narrative_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    position: Mapped[str] = mapped_column(Text)
    stance: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ProceduralMemory(Base):
    """Self-modifying agent prompts — GUARDED. v3 adds versioning + approval."""
    __tablename__ = "procedural_memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    prompt_text: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)               # v3
    proposed: Mapped[bool] = mapped_column(Boolean, default=False)         # v3
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)   # v3
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # v3
    rolled_back: Mapped[bool] = mapped_column(Boolean, default=False)      # v3
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
class ContentCalendar(Base):
    __tablename__ = "content_calendar"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pillar_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="planned")
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")


# --------------------------------------------------------------------------
# v3 additions: cost_log, dead_letter, procedural_proposals
# --------------------------------------------------------------------------
class CostLog(Base):
    __tablename__ = "cost_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), index=True, nullable=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    agent: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class DeadLetter(Base):
    __tablename__ = "dead_letter"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), index=True, nullable=True)
    step: Mapped[str] = mapped_column(String(128))
    error: Mapped[str] = mapped_column(Text)
    context: Mapped[dict] = mapped_column(JSONB, default=dict)   # enough to resume/diagnose
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class ProceduralProposal(Base):
    """Pending prompt changes awaiting human approval in the review UI."""
    __tablename__ = "procedural_proposals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    current_version: Mapped[int] = mapped_column(Integer)
    proposed_prompt: Mapped[str] = mapped_column(Text)
    performance_data: Mapped[dict] = mapped_column(JSONB, default=dict)   # what triggered it
    voice_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)  # held-out check
    status: Mapped[str] = mapped_column(String(32), default="pending")   # pending|approved|rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


ALL_TABLES = [
    Job, EpisodicMemory, SemanticMemory, NarrativeMemory, ProceduralMemory,
    ContentCalendar, CostLog, DeadLetter, ProceduralProposal,
]


# --------------------------------------------------------------------------
# Async engine / session
# --------------------------------------------------------------------------
_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/media_engine")
        _engine = create_async_engine(url, pool_pre_ping=True, future=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_db() -> None:
    """Create the pgvector extension and all tables. Requires a live DB."""
    from sqlalchemy import text
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


def render_ddl() -> str:
    """Emit CREATE TABLE DDL without a live DB (PostgreSQL dialect)."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable
    parts = ["CREATE EXTENSION IF NOT EXISTS vector;\n"]
    for table in Base.metadata.sorted_tables:
        parts.append(str(CreateTable(table).compile(dialect=postgresql.dialect())).strip() + ";\n")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    if "--print-ddl" in sys.argv:
        print(render_ddl())
    else:
        print(f"Tables: {[t.name for t in Base.metadata.sorted_tables]}")
        print(f"EMBED_DIM={EMBED_DIM}")
