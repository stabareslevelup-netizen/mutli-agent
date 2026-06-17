"""
engine/core/models.py — Pydantic v2 schemas for EVERY inter-agent handoff.

Brand-agnostic by construction: no brand/character/voice/pillar string appears
here. All brand-specific values arrive at runtime from the brand config.

Three Phase-0 constraints are encoded as schema-level invariants (validators),
not left to agent discipline:

  RULE 1 (velocity): ordinal verdict + confidence, NEVER fake-precise numbers
          until a real time-series source is wired in. -> VelocitySignal
  RULE 2 (narrative gap): entity-specific white space vs topic-generic
          saturation are distinct and modelled. -> NarrativeGap.gap_type
  RULE 3 (citation): presence detection is disambiguation-aware. "present"
          is impossible unless the matched context was verified. -> CitationSignal
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# Shared primitives
# ===========================================================================
class Source(BaseModel):
    url: str
    title: str = ""
    published: Optional[date] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class HandoffMeta(BaseModel):
    """Wraps every tier->tier handoff so the validation gate can inspect it."""
    job_id: str
    tier: int
    agent: str
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_utcnow)


# ===========================================================================
# Tier 1 — Research
# ===========================================================================
class ResearchAngle(BaseModel):
    angle: str
    rationale: str = ""
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ResearchOutput(BaseModel):
    angles: list[ResearchAngle] = Field(..., min_length=1, max_length=3)
    generated_at: datetime = Field(default_factory=_utcnow)


# ===========================================================================
# Tier 1 — Timing  (the three [UNPROVEN] signals, de-risked in Phase 0)
# ===========================================================================
class VelocityVerdict(str, Enum):
    surging = "surging"
    steady = "steady"
    declining = "declining"
    dead = "dead"
    unknown = "unknown"      # graceful-degrade state when no source available


class VelocitySource(str, Enum):
    web_search_ordinal = "web_search_ordinal"      # ordinal only
    gdelt_timeseries = "gdelt_timeseries"          # numeric
    wikipedia_pageviews = "wikipedia_pageviews"    # numeric
    unavailable = "unavailable"                    # nothing reachable


_NUMERIC_SOURCES = {VelocitySource.gdelt_timeseries, VelocitySource.wikipedia_pageviews}
_ORDINAL_SOURCES = {VelocitySource.web_search_ordinal, VelocitySource.unavailable}


class VelocitySignal(BaseModel):
    """RULE 1: ordinal verdict + confidence by default. `numeric_rate` is only
    permitted — and required — when a real time-series source produced it."""
    topic: str
    verdict: VelocityVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: VelocitySource
    numeric_rate: Optional[float] = None    # e.g. week-over-week % change; numeric sources only
    window_days: int = 30
    rationale: str = ""
    needs_human: bool = False

    @model_validator(mode="after")
    def _no_fake_precision(self) -> "VelocitySignal":
        if self.source in _ORDINAL_SOURCES and self.numeric_rate is not None:
            raise ValueError(
                "RULE 1: ordinal/unavailable source must not carry a numeric_rate "
                "(no fake-precise velocity until a real time-series is wired in)"
            )
        if self.source in _NUMERIC_SOURCES and self.numeric_rate is None:
            raise ValueError("a time-series source must provide a numeric_rate")
        return self


class GapType(str, Enum):
    """RULE 2: the distinction that prevents false 'covered' verdicts."""
    entity_specific = "entity_specific"   # white space around a named entity
    topic_generic = "topic_generic"       # the broad topic angle


class NarrativeGap(BaseModel):
    angle: str
    gap_type: GapType
    entity: Optional[str] = None
    coverage_density: float = Field(..., ge=0.0, le=1.0)   # 0 = open white space
    is_open: bool
    evidence: str = ""

    @model_validator(mode="after")
    def _entity_required_for_specific(self) -> "NarrativeGap":
        if self.gap_type == GapType.entity_specific and not self.entity:
            raise ValueError("RULE 2: an entity_specific gap must name its `entity`")
        return self


class NarrativeGapSignal(BaseModel):
    topic: str
    entity: Optional[str] = None
    gaps: list[NarrativeGap] = Field(default_factory=list)


class CitationPresence(str, Enum):
    present = "present"
    absent = "absent"
    ambiguous = "ambiguous"   # collision unresolved -> cannot claim present/absent


class DisambiguationGuard(BaseModel):
    """RULE 3: built into the schema, not bolted on. A naive substring match of
    the brand name inside an unrelated context (e.g. a same-named MTG card)
    must NOT be reported as a citation."""
    collision_terms: list[str] = Field(default_factory=list)   # known same-name entities
    matched_context_verified: bool = False                     # did we confirm it's really the brand?
    risk: Optional[str] = None


class CitationSignal(BaseModel):
    query: str
    brand_aliases: list[str] = Field(default_factory=list)
    presence: CitationPresence
    rank: Optional[int] = None
    share_of_voice: float = Field(0.0, ge=0.0, le=1.0)
    incumbents: list[str] = Field(default_factory=list)        # competitive set cited instead
    disambiguation: DisambiguationGuard = Field(default_factory=DisambiguationGuard)

    @model_validator(mode="after")
    def _guard(self) -> "CitationSignal":
        if self.presence == CitationPresence.present and not self.disambiguation.matched_context_verified:
            raise ValueError(
                "RULE 3: presence='present' requires disambiguation.matched_context_verified=True "
                "(guards against the false-positive same-name-context match)"
            )
        if self.disambiguation.collision_terms and not self.disambiguation.matched_context_verified \
                and self.presence != CitationPresence.ambiguous:
            raise ValueError(
                "RULE 3: unresolved name collision must be reported as presence='ambiguous'"
            )
        return self


class PrimarySourceSignal(BaseModel):
    """Primary-source signals: official posts/filings/releases over commentary."""
    sources: list[Source] = Field(default_factory=list)
    has_primary: bool = False


class TimingSignal(BaseModel):
    """The Timing agent's Tier-1 handoff into fusion."""
    velocity: VelocitySignal
    gaps: NarrativeGapSignal
    citation: CitationSignal
    primary_sources: PrimarySourceSignal = Field(default_factory=PrimarySourceSignal)
    confidence_overall: float = Field(0.5, ge=0.0, le=1.0)


# ===========================================================================
# Tier 1 — Memory query result
# ===========================================================================
class MemoryItem(BaseModel):
    kind: str                 # episodic | semantic | narrative
    content: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    ref_id: Optional[str] = None


class NarrativeConstraint(BaseModel):
    """Staked positions the Strategy agent must never contradict."""
    position: str
    stance: str
    ref_id: Optional[str] = None


class MemoryQueryResult(BaseModel):
    episodic: list[MemoryItem] = Field(default_factory=list)
    semantic: list[MemoryItem] = Field(default_factory=list)
    narrative: list[NarrativeConstraint] = Field(default_factory=list)


# ===========================================================================
# Tier 2 — Strategy (fusion output)
# ===========================================================================
class ContentFormat(str, Enum):
    video = "video"          # cinematic character render (default for deep angles)
    image = "image"          # static render — faster/cheaper, speed-sensitive angles
    text_only = "text_only"  # copy stands alone, no visual


class StrategyPacket(BaseModel):
    chosen_angle: str
    rationale: str = ""
    content_format: ContentFormat = ContentFormat.video
    pillar_id: str = ""                  # matched pillar (drives Prompt Engineer mood)
    citation_status: str = ""            # timing citation presence (for the verification gate)
    velocity_confidence: float = 0.0
    requires_hedging: bool = False       # Fix 1: hedge all claims when set
    skeptic_summary: str = ""            # Fix 2: filled by the Skeptic Agent
    confidence_adjustment: float = 0.0   # -0.1..0.0 from the Skeptic
    formats: list[str] = Field(default_factory=list)
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    hard_constraints: list[NarrativeConstraint] = Field(default_factory=list)
    inputs_digest: dict[str, float] = Field(default_factory=dict)   # per-signal contribution

    @field_validator("fusion_weights")
    @classmethod
    def _weights_sane(cls, v: dict[str, float]) -> dict[str, float]:
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("fusion_weights must sum to 1.0")
        return v


class SkepticReview(BaseModel):
    """Adversarial review of the chosen angle (Tier 2; informs, never blocks)."""
    disputes: str = ""                   # what an insider would dispute
    hidden_assumptions: str = ""         # unstated assumptions
    alternative_explanations: str = ""   # other readings of the same facts
    overstatement: str = ""              # is scale/impact overstated
    skeptic_summary: str                 # 2-3 sentences
    confidence_adjustment: float = Field(0.0, ge=-0.1, le=0.0)  # never increases confidence


# ===========================================================================
# Tier 3 — Production-input agents
# ===========================================================================
class CopyOutput(BaseModel):
    x_thread: list[str] = Field(default_factory=list)
    ig_caption: str = ""
    youtube_script: str = ""


class PromptEngineerOutput(BaseModel):
    higgsfield_prompt: str
    character_placeholder: str

    @model_validator(mode="after")
    def _placeholder_present(self) -> "PromptEngineerOutput":
        if self.character_placeholder and self.character_placeholder not in self.higgsfield_prompt:
            raise ValueError("higgsfield_prompt must embed the character placeholder")
        return self


# ===========================================================================
# Tier 4 — Production + Quality
# ===========================================================================
class ProductionResult(BaseModel):
    asset_id: Optional[str] = None
    asset_url: Optional[str] = None
    status: str = "pending"   # pending | ready | failed


class QualityRoute(str, Enum):
    publish_queue = "publish_queue"
    revise = "revise"
    reject = "reject"


class QualityScore(BaseModel):
    voice: float = Field(..., ge=0.0, le=1.0)
    narrative: float = Field(..., ge=0.0, le=1.0)
    format: float = Field(..., ge=0.0, le=1.0)
    hook: float = Field(..., ge=0.0, le=1.0)
    coherence: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)
    route: QualityRoute
    reasons: list[str] = Field(default_factory=list)


# ===========================================================================
# Tier 5 — Distribution (confirm mode in v1)
# ===========================================================================
class PostingMode(str, Enum):
    confirm = "confirm"
    scheduled = "scheduled"
    auto = "auto"


class DistributionPlan(BaseModel):
    posting_mode: PostingMode
    platforms: list[str] = Field(default_factory=list)
    staged: bool = True
    published: bool = False

    @model_validator(mode="after")
    def _confirm_never_auto_publishes(self) -> "DistributionPlan":
        if self.posting_mode == PostingMode.confirm and self.published:
            raise ValueError("confirm mode must stage only; nothing publishes without human confirm")
        return self


# ===========================================================================
# Review UI (Phase 8) — the screenshot-able "my 10 agents built this" record
# ===========================================================================
class AgentAttribution(BaseModel):
    """The agent story behind a piece — designed to be displayed/screenshotted."""
    research_chosen: str = ""
    research_alternatives: list[str] = Field(default_factory=list)
    timing_velocity: str = ""
    timing_velocity_confidence: float = 0.0
    timing_gaps: list[str] = Field(default_factory=list)
    timing_citation: str = ""
    memory_context: list[str] = Field(default_factory=list)
    copy_output_seconds: float = 0.0
    quality: dict[str, float] = Field(default_factory=dict)   # 5 dims + overall
    skeptic_summary: str = ""                                 # Skeptic Agent (11th)


class ReviewItem(BaseModel):
    job_id: str
    brand_id: str
    status: str
    content_format: ContentFormat
    pillar_id: str = ""
    requires_hedging: bool = False
    created_at: datetime = Field(default_factory=_utcnow)
    chosen_angle: str = ""
    x_thread: list[str] = Field(default_factory=list)
    ig_caption: str = ""
    youtube_script: str = ""
    asset_url: Optional[str] = None
    quality_overall: float = 0.0
    quality_route: str = ""
    auto_eligible: bool = False
    platforms: list[str] = Field(default_factory=list)
    attribution: AgentAttribution = Field(default_factory=AgentAttribution)


class ProposalView(BaseModel):
    """A pending procedural prompt-change surfaced for human approval."""
    proposal_id: int
    agent_name: str
    current_version: int
    proposed_prompt: str
    performance_data: dict = Field(default_factory=dict)
    voice_similarity: Optional[float] = None
    voice_threshold: float = 0.6
    status: str = "pending"


__all__ = [
    "SCHEMA_VERSION", "Source", "HandoffMeta",
    "ResearchAngle", "ResearchOutput",
    "VelocityVerdict", "VelocitySource", "VelocitySignal",
    "GapType", "NarrativeGap", "NarrativeGapSignal",
    "CitationPresence", "DisambiguationGuard", "CitationSignal",
    "PrimarySourceSignal", "TimingSignal",
    "MemoryItem", "NarrativeConstraint", "MemoryQueryResult",
    "ContentFormat", "StrategyPacket", "SkepticReview", "CopyOutput", "PromptEngineerOutput",
    "ProductionResult", "QualityRoute", "QualityScore",
    "PostingMode", "DistributionPlan",
    "AgentAttribution", "ReviewItem", "ProposalView",
]
