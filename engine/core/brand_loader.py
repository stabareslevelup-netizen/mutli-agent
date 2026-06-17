"""
engine/core/brand_loader.py — loads + validates a brand config at runtime.

This is the seam that keeps the engine brand-free: every brand-specific value
(voice, pillars, character, weights) lives in a YAML file and is validated into
a typed BrandConfig here. The engine code reads BrandConfig fields, never
literals. No brand string is hardcoded in this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from engine.core.models import ContentFormat, PostingMode


class Pillar(BaseModel):
    id: str
    desc: str = ""
    format: Optional[ContentFormat] = None   # preferred content format for this pillar


class Character(BaseModel):
    name: str
    higgsfield_element_id: str
    placeholder: str
    palette: list[str] = Field(default_factory=list)
    typography: list[str] = Field(default_factory=list)
    style: str = ""
    visual_encoding: str = ""

    @model_validator(mode="after")
    def _placeholder_wraps_element_id(self) -> "Character":
        if self.higgsfield_element_id not in self.placeholder:
            raise ValueError("character.placeholder must contain higgsfield_element_id")
        return self


class Budget(BaseModel):
    daily_usd: float = Field(25.0, gt=0)


class BrandConfig(BaseModel):
    brand_id: str
    display_name: str
    voice: str
    pillars: list[Pillar] = Field(..., min_length=1)
    character: Character
    formats: list[str] = Field(..., min_length=1)
    fusion_weights: dict[str, float]
    budget: Budget = Field(default_factory=Budget)
    posting_mode: PostingMode = PostingMode.confirm
    credentials_ref: str = "env"

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "BrandConfig":
        total = sum(self.fusion_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"fusion_weights must sum to 1.0 (got {total:.3f})")
        required = {"research", "memory", "timing"}
        missing = required - set(self.fusion_weights)
        if missing:
            raise ValueError(f"fusion_weights missing keys: {sorted(missing)}")
        return self

    def pillar_ids(self) -> list[str]:
        return [p.id for p in self.pillars]


def default_brand_path() -> Path:
    # No brand-specific default may live in engine code. The path is supplied
    # at runtime via env (set BRAND_CONFIG_PATH; see .env.example).
    raw = os.getenv("BRAND_CONFIG_PATH")
    if not raw:
        raise RuntimeError("BRAND_CONFIG_PATH is not set (point it at a brand config)")
    return Path(raw)


def load_brand(path: Optional[str | Path] = None) -> BrandConfig:
    p = Path(path) if path else default_brand_path()
    if not p.exists():
        raise FileNotFoundError(f"brand config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"brand config must be a mapping, got {type(raw).__name__}")
    return BrandConfig.model_validate(raw)


if __name__ == "__main__":
    cfg = load_brand()
    print(f"loaded brand_id={cfg.brand_id} pillars={cfg.pillar_ids()} "
          f"posting_mode={cfg.posting_mode.value} weights_ok=True")
