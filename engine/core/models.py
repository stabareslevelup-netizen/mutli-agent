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
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

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
# Tier 1 — Research (Phase 2, X-agent migration: sweep across primary-source
# government/patent/paper feeds, not a single-topic lookup — see
# engine/tools/sam_gov.py etc. and Research.sweep())
# ===========================================================================
class SourceKind(str, Enum):
    sam_gov = "SAM.gov"
    darpa = "DARPA"
    patent = "Patent"
    arxiv = "arXiv"
    congress = "Congress"
    fed_register = "FedRegister"
    job_signal = "JobSignal"


class ResearchItem(BaseModel):
    item_id: str                          # Research-generated; Timing references
                                           # it as a pass-through (not in the
                                           # original spec text — needed so
                                           # TimingDecision has something to key on)
    source: SourceKind
    date_found: date
    headline: str
    raw_detail: str
    novelty_score: int = Field(..., ge=1, le=10)
    post_angle: str
    source_url: HttpUrl

    @model_validator(mode="after")
    def _novelty_floor(self) -> "ResearchItem":
        # spec: "Only output items with novelty_score 6 or higher" — enforced
        # structurally so a borderline LLM call can't sneak a 5 through.
        if self.novelty_score < 6:
            raise ValueError("novelty_score below 6 — spec requires discarding, not outputting")
        return self


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
    """Superseded by TimingDecision (Phase 2) as Timing's public return type.
    Timing uses PostHistoryStore for duplicate_check and narrative_gap_check.
    Velocity, urgency, and citation_hedge_required fold into the batch LLM
    call in timing.py. velocity_probe.py, citation_monitor.py, AND
    narrative_gap.py are all orphaned by this migration (verified: Strategy's
    conflict check uses the separate checker/NarrativeConstraint mechanism,
    not find_gaps() — narrative_gap.py has no live caller left). Kept here
    (unused) rather than deleted, per the orphan-don't-delete-until-Phase-5
    convention applied to all three tool modules."""
    velocity: VelocitySignal
    gaps: NarrativeGapSignal
    citation: CitationSignal
    primary_sources: PrimarySourceSignal = Field(default_factory=PrimarySourceSignal)
    confidence_overall: float = Field(0.5, ge=0.0, le=1.0)


class PostingSlot(str, Enum):
    slot_7am = "7AM"
    slot_9am = "9AM"
    slot_12pm = "12PM"
    slot_3pm = "3PM"
    slot_6pm = "6PM"
    reject = "reject"


class TimingDecision(BaseModel):
    """Timing's Phase 2 return type: one per ResearchItem, assigning a
    posting slot (or rejecting) across the whole day's batch at once —
    "at most ONE item per slot per day" is a cross-item constraint Timing
    enforces over the full list, not per-item in isolation."""
    item_id: str
    recommended_slot: PostingSlot
    rejection_reason: Optional[str] = None
    urgency_note: str
    citation_hedge_required: bool          # PRESERVED — maps to the existing
                                            # CitationSignal/requires_hedging logic

    @model_validator(mode="after")
    def _reject_needs_reason(self) -> "TimingDecision":
        if self.recommended_slot == PostingSlot.reject and not self.rejection_reason:
            raise ValueError("recommended_slot='reject' requires rejection_reason")
        return self


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
# Tier 2 — Strategy (Phase 2: operates on a single ResearchItem, not a
# fused multi-angle blend — engine/core/fusion.py is no longer called by
# the new Orchestrator flow, left in place but orphaned pending cleanup)
# ===========================================================================
class PostFormat(str, Enum):
    short_hook = "short_hook"
    pov_post = "pov_post"
    thread = "thread"
    data_drop = "data_drop"


APPROVED_HASHTAGS = {"#AI", "#PhysicalAI", "#Robotics", "#DefenseTech",
                     "#AutonomousSystems", "#AIAgents", "#FutureOfWar", "#DARPA"}


def _validate_hashtags(v: list[str]) -> list[str]:
    bad = [h for h in v if h not in APPROVED_HASHTAGS]
    if bad:
        raise ValueError(f"hashtags not on the approved list: {bad}")
    return v


class StrategyOutput(BaseModel):
    chosen_angle: str
    format: PostFormat
    must_include: list[str] = Field(..., min_length=1, max_length=3)
    must_avoid: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list, max_length=2)
    thread_spine: Optional[list[str]] = None
    narrative_conflict_flag: bool = False             # PRESERVED — maps to the
    narrative_conflict_note: Optional[str] = None      # existing StrategyBlocked /
                                                        # memory-conflict check

    @field_validator("hashtags")
    @classmethod
    def _hashtags_allowed(cls, v: list[str]) -> list[str]:
        return _validate_hashtags(v)

    @model_validator(mode="after")
    def _conflict_needs_note(self) -> "StrategyOutput":
        if self.narrative_conflict_flag and not self.narrative_conflict_note:
            raise ValueError("narrative_conflict_flag=True requires narrative_conflict_note")
        return self

    @model_validator(mode="after")
    def _thread_spine_shape(self) -> "StrategyOutput":
        if self.format == PostFormat.thread and not self.thread_spine:
            raise ValueError("format='thread' requires a thread_spine")
        if self.thread_spine is not None and not (4 <= len(self.thread_spine) <= 7):
            raise ValueError("thread_spine must have 4-7 entries")
        return self


class SkepticVerdict(str, Enum):
    approved = "approved"
    revise = "revise"
    reject = "reject"


class SkepticOutput(BaseModel):
    """Phase 2: Skeptic can now reject/request revision (Orchestrator-enforced
    retry bound), not just inform — see orchestrator.py's Skeptic loop."""
    verdict: SkepticVerdict
    critique: str
    revised_angle: Optional[str] = None
    revised_must_include: Optional[list[str]] = None

    @model_validator(mode="after")
    def _revise_needs_revision(self) -> "SkepticOutput":
        if self.verdict == SkepticVerdict.revise and (
                not self.revised_angle or not self.revised_must_include):
            raise ValueError("verdict='revise' requires revised_angle and revised_must_include")
        return self


# ===========================================================================
# Tier 3 — Copy (Phase 2: X-only — main_post + reply_link replace the old
# x_thread/ig_caption/youtube_script shape entirely, per the clean-break
# identity pivot)
# ===========================================================================
class CopyOutput(BaseModel):
    main_post: str = Field(..., max_length=280)
    reply_link: str
    thread_tweets: Optional[list[str]] = None
    format_used: PostFormat
    char_count: int = 0                  # recomputed below, not LLM-trusted
    hashtags_used: list[str] = Field(default_factory=list, max_length=2)
    citation_hedged: bool = False        # PRESERVED — set True when Timing
                                          # flagged citation_hedge_required

    @field_validator("hashtags_used")
    @classmethod
    def _hashtags_allowed(cls, v: list[str]) -> list[str]:
        return _validate_hashtags(v)

    @model_validator(mode="after")
    def _no_url_in_body(self) -> "CopyOutput":
        # spec: "NON-NEGOTIABLE... never put a URL in main_post"
        if "http://" in self.main_post or "https://" in self.main_post:
            raise ValueError("main_post must never contain a URL — use reply_link")
        return self

    @model_validator(mode="after")
    def _char_count_is_real(self) -> "CopyOutput":
        object.__setattr__(self, "char_count", len(self.main_post))
        return self


# ===========================================================================
# Tier 4 — Quality (Phase 2: 5 new dimensions, gate logic computed
# structurally from scores rather than trusted from the LLM's self-report —
# same reasoning as VelocitySignal's no-fake-precision rule)
# ===========================================================================
class QualityDimensions(BaseModel):
    source_specificity: int = Field(..., ge=0, le=10)
    differentiation: int = Field(..., ge=0, le=10)
    hook_strength: int = Field(..., ge=0, le=10)
    format_compliance: int = Field(..., ge=0, le=10)
    thesis_alignment: int = Field(..., ge=0, le=10)


class QualityVerdict(str, Enum):
    pass_ = "pass"
    revise = "revise"
    reject = "reject"


class QualityOutput(BaseModel):
    scores: QualityDimensions
    total: int = 0                                    # recomputed below
    verdict: QualityVerdict = QualityVerdict.pass_     # recomputed below
    notes: list[str] = Field(default_factory=list)
    blocking_dimension: Optional[str] = None           # recomputed below
    approval_note: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _compute_gate(self) -> "QualityOutput":
        d = self.scores.model_dump()
        object.__setattr__(self, "total", sum(d.values()))
        lowest_name = min(d, key=d.get)
        lowest_val = d[lowest_name]
        if lowest_val < 5:
            verdict, blocking = QualityVerdict.reject, lowest_name
        elif lowest_val < 7:
            verdict, blocking = QualityVerdict.revise, lowest_name
        else:
            verdict, blocking = QualityVerdict.pass_, None
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "blocking_dimension", blocking)
        return self


# ===========================================================================
# Tier 5 — Distribution staging record (Phase 2)
# ===========================================================================
class StagedPost(BaseModel):
    main_post: str
    reply_link: str
    thread_tweets: Optional[list[str]] = None
    format_used: PostFormat
    scheduled_slot: PostingSlot
    source_url: HttpUrl
    novelty_score: int = Field(..., ge=1, le=10)
    quality_scores: QualityDimensions
    approval_note: str
    requires_human_approval: Literal[True] = True   # structurally can't be False —
                                                     # mirrors DistributionPlan's
                                                     # confirm-mode invariant below
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None

    @model_validator(mode="after")
    def _approval_fields_paired(self) -> "StagedPost":
        if (self.approved_at is None) != (self.approved_by is None):
            raise ValueError("approved_at and approved_by must be set together")
        return self


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
# Review UI (Phase 8) — the screenshot-able "my agents built this" record
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
    skeptic_summary: str = ""                                 # Skeptic Agent


class ReviewItem(BaseModel):
    job_id: str
    brand_id: str
    status: str
    pillar_id: str = ""
    requires_hedging: bool = False
    created_at: datetime = Field(default_factory=_utcnow)
    chosen_angle: str = ""
    x_thread: list[str] = Field(default_factory=list)
    ig_caption: str = ""
    youtube_script: str = ""
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
    "SourceKind", "ResearchItem",
    "VelocityVerdict", "VelocitySource", "VelocitySignal",
    "GapType", "NarrativeGap", "NarrativeGapSignal",
    "CitationPresence", "DisambiguationGuard", "CitationSignal",
    "PrimarySourceSignal", "TimingSignal",
    "PostingSlot", "TimingDecision",
    "MemoryItem", "NarrativeConstraint", "MemoryQueryResult",
    "PostFormat", "APPROVED_HASHTAGS", "StrategyOutput",
    "SkepticVerdict", "SkepticOutput", "CopyOutput",
    "QualityDimensions", "QualityVerdict", "QualityOutput", "StagedPost",
    "PostingMode", "DistributionPlan",
    "AgentAttribution", "ReviewItem", "ProposalView",
]
