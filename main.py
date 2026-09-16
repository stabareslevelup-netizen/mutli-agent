"""
main.py — FastAPI orchestration + review API + dashboard.

Builds one shared engine at startup (so the review queue persists across
requests), wires the review dashboard API (queue / job detail / approve /
reject / procedural proposals), and serves the dark-editorial
review_dashboard.html. Brand-agnostic: all brand values come from the
loaded config.

Phase 2 (X-agent migration): the manual single-topic trigger path (POST
/jobs -> ManualTrigger -> Orchestrator.run(topic=...)) is retired entirely,
not rebuilt -- the account is fully proactive-sweep-based now.
Orchestrator.run_sweep() takes zero topic input by design; there is no
manual "fire one topic" entry point anymore. engine/triggers/manual_ui.py
and engine/triggers/calendar_cron.py were deleted, not just orphaned --
both were built entirely around single-topic firing, a premise that no
longer exists. A schedule-based trigger that calls run_sweep() on a cadence
would be new code, not a revival of calendar_cron.py's old shape -- not
built here, out of scope for this change.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # resolve API keys for the server process

from engine.core.brand_loader import BrandConfig, load_brand

_UI = os.path.join(os.path.dirname(__file__), "ui", "review_dashboard.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.brand = load_brand()
    job_store = dead_letter_sink = review_store = None   # InMemory defaults inside build_orchestrator()
    if os.getenv("DATABASE_URL"):
        from engine.core.database import init_db, get_sessionmaker
        await init_db()
        app.state.db_ready = True
        # Sql-backed stores so a separate process (e.g. a scheduled sweep
        # script) and this dashboard server share state through the same
        # database, instead of each holding its own in-memory queue that
        # the other can never see. PostHistoryStore/NarrativeConflictSink/
        # RateLimitQueue deliberately stay in-memory -- none of them are
        # load-bearing for "a human can see and act on a staged item".
        from engine.core.job_manager import SqlJobStore
        from engine.core.dead_letter import SqlDeadLetterSink
        from engine.core.review_store import SqlReviewStore
        sm = get_sessionmaker()
        job_store = SqlJobStore(sm)
        dead_letter_sink = SqlDeadLetterSink(sm)
        review_store = SqlReviewStore(sm)
    else:
        app.state.db_ready = False
    # one shared engine so the review queue persists across requests
    from engine.core.assembly import build_orchestrator
    from engine.core.review_service import ReviewService
    from engine.core.feedback_service import FeedbackService
    from engine.tools.x_client import build_x_client_from_env
    eng = build_orchestrator(app.state.brand, job_store=job_store,
                             dead_letter_sink=dead_letter_sink, review_store=review_store)
    app.state.engine = eng
    # x_http stays None (graceful degrade) until X_API_KEY/X_API_KEY_SECRET/
    # X_ACCESS_TOKEN/X_ACCESS_TOKEN_SECRET are all set in .env — the human
    # confirm gate in ReviewService.approve() is identical either way; this
    # only decides whether an approved post actually reaches X or is marked
    # "approved, publish pending credentials".
    app.state.review = ReviewService(
        review_store=eng.review_store, distribution=eng.distribution,
        procedural=eng.procedural, dead_letter_sink=eng.dead_letter,
        x_http=build_x_client_from_env())
    epi, sem, _nar = eng.memory
    app.state.feedback = FeedbackService(
        review_store=eng.review_store, episodic=epi, semantic=sem,
        procedural=eng.procedural, brand_id=app.state.brand.brand_id)
    yield


app = FastAPI(title="Media Engine", version="0.1.0", lifespan=lifespan)


class RejectRequest(BaseModel):
    reason: str


class ApprovalRequest(BaseModel):
    approved_by: str


class FeedbackRequest(BaseModel):
    # per-platform engagement metrics; manual paste works:
    # {"engagement": {"x": {"views": 12000, "likes": 340, "shares": 22, "saves": 15, "replies": 8}}}
    engagement: dict[str, dict[str, float]]


@app.get("/healthz")
async def healthz():
    brand: BrandConfig = app.state.brand
    return {"status": "ok", "brand_id": brand.brand_id,
            "posting_mode": brand.posting_mode.value, "db_ready": app.state.db_ready}


@app.get("/brand")
async def brand_info():
    brand: BrandConfig = app.state.brand
    return {"brand_id": brand.brand_id, "display_name": brand.display_name,
            "pillars": brand.pillar_ids(), "formats": brand.formats,
            "fusion_weights": brand.fusion_weights, "daily_budget_usd": brand.budget.daily_usd}


# --- review dashboard API ---------------------------------------------------
@app.get("/review/queue")
async def review_queue():
    items = await app.state.review.queue()
    return [i.model_dump(mode="json") for i in items]


@app.get("/review/job/{job_id}")
async def review_job(job_id: str):
    item = await app.state.review.get(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail="job not found")
    return item.model_dump(mode="json")


@app.post("/review/job/{job_id}/approve")
async def review_approve(job_id: str):
    return await app.state.review.approve(job_id=job_id, confirmed=True)


@app.post("/review/job/{job_id}/reject")
async def review_reject(job_id: str, req: RejectRequest):
    return await app.state.review.reject(job_id=job_id, reason=req.reason)


@app.get("/review/pillar_report")
async def review_pillar_report(days: int = 7):
    """Weekly boredom-signal report: drafts generated/approved/rejected per
    content pillar (engine/core/pillar_report.py). Only reflects this
    process's own review store -- see that module's docstring."""
    from engine.core.pillar_report import build_pillar_report
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stats = await build_pillar_report(app.state.engine.review_store, since=since)
    return [asdict(s) for s in stats.values()]


@app.get("/review/proposals")
async def review_proposals():
    return [p.model_dump(mode="json") for p in await app.state.review.proposals()]


@app.post("/review/proposals/{proposal_id}/approve")
async def review_proposal_approve(proposal_id: int, req: ApprovalRequest):
    return await app.state.review.approve_proposal(proposal_id=proposal_id, approved_by=req.approved_by)


@app.post("/review/proposals/{proposal_id}/reject")
async def review_proposal_reject(proposal_id: int):
    return await app.state.review.reject_proposal(proposal_id=proposal_id)


@app.post("/review/rollback/{agent}")
async def review_rollback(agent: str):
    return await app.state.review.rollback(agent=agent, brand_id=app.state.brand.brand_id)


@app.post("/feedback/{job_id}")
async def ingest_feedback(job_id: str, req: FeedbackRequest):
    """Engagement ingest (Phase 9). Accepts manual input — paste the numbers
    after posting by hand. Auto-updates episodic + semantic memory; surfaces a
    procedural proposal when a cross-job pattern emerges (never auto-applies)."""
    return await app.state.feedback.ingest(job_id=job_id, engagement=req.engagement)


@app.get("/")
async def dashboard():
    return FileResponse(_UI)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
