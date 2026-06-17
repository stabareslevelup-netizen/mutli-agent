"""
main.py — FastAPI orchestration + review API + dashboard.

Builds one shared engine at startup (so the review queue persists across
requests), wires the manual trigger (POST /jobs) and the review dashboard API
(queue / job detail / approve / reject / procedural proposals), and serves the
dark-editorial review_dashboard.html. Brand-agnostic: all brand values come
from the loaded config.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.core.brand_loader import BrandConfig, load_brand

_UI = os.path.join(os.path.dirname(__file__), "ui", "review_dashboard.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.brand = load_brand()
    if os.getenv("DATABASE_URL"):
        from engine.core.database import init_db
        await init_db()
        app.state.db_ready = True
    else:
        app.state.db_ready = False
    # one shared engine so the review queue persists across requests
    from engine.core.assembly import build_orchestrator
    from engine.core.review_service import ReviewService
    eng = build_orchestrator(app.state.brand)
    app.state.engine = eng
    app.state.review = ReviewService(
        review_store=eng.review_store, distribution=eng.distribution,
        procedural=eng.procedural, dead_letter_sink=eng.dead_letter, x_http=None)
    yield


app = FastAPI(title="Media Engine", version="0.1.0", lifespan=lifespan)


class JobRequest(BaseModel):
    topic: str
    entity: str | None = None


class RejectRequest(BaseModel):
    reason: str


class ApprovalRequest(BaseModel):
    approved_by: str


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


@app.post("/jobs")
async def create_job(req: JobRequest):
    """Manual trigger: fires the full pipeline via Orchestrator.run. In
    posting_mode=confirm the job is staged for review — nothing publishes."""
    from engine.triggers.manual_ui import ManualTrigger
    res = await ManualTrigger(app.state.engine.orchestrator).fire(topic=req.topic, entity=req.entity)
    return {"job_id": res.job_id, "status": res.status, "reason": res.reason,
            "cost_usd": res.cost_usd}


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


@app.get("/")
async def dashboard():
    return FileResponse(_UI)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
