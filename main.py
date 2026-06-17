"""
main.py — FastAPI orchestration skeleton (Phase 1).

Loads + validates the brand config at startup and exposes health/inspection
routes. Pipeline tiers are wired in later phases. Brand-agnostic: every
brand-specific value comes from the loaded config, never a literal here.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from engine.core.brand_loader import BrandConfig, load_brand


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Brand config is required to run; fail loud if it's invalid.
    app.state.brand = load_brand()
    # DB init is opt-in (Phase 1 can boot without a live Postgres).
    if os.getenv("DATABASE_URL"):
        from engine.core.database import init_db
        await init_db()
        app.state.db_ready = True
    else:
        app.state.db_ready = False
    yield


app = FastAPI(title="Media Engine", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    brand: BrandConfig = app.state.brand
    return {
        "status": "ok",
        "brand_id": brand.brand_id,
        "posting_mode": brand.posting_mode.value,
        "db_ready": app.state.db_ready,
    }


@app.get("/brand")
async def brand_info():
    brand: BrandConfig = app.state.brand
    return {
        "brand_id": brand.brand_id,
        "display_name": brand.display_name,
        "pillars": brand.pillar_ids(),
        "formats": brand.formats,
        "fusion_weights": brand.fusion_weights,
        "daily_budget_usd": brand.budget.daily_usd,
    }


class JobRequest(BaseModel):
    topic: str
    entity: str | None = None


@app.post("/jobs")
async def create_job(req: JobRequest):
    """Manual trigger (Phase 7): fires the full pipeline via Orchestrator.run.
    In posting_mode=confirm the result is staged for review — nothing publishes."""
    from engine.core.assembly import build_orchestrator
    from engine.triggers.manual_ui import ManualTrigger

    engine = build_orchestrator(app.state.brand)
    res = await ManualTrigger(engine.orchestrator).fire(topic=req.topic, entity=req.entity)
    return {
        "job_id": res.job_id,
        "status": res.status,
        "reason": res.reason,
        "cost_usd": res.cost_usd,
        "chosen_angle": getattr(res.artifacts.get("packet"), "chosen_angle", None),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
