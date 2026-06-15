# PHASE 0 FINDINGS — De-risking the [UNPROVEN] Timing tools

**Date:** 2026-06-15
**Scope:** SPEC_v3 Phase 0 only. Throwaway spikes of `velocity_probe`,
`narrative_gap`, `citation_monitor`. **No production code.** Goal per spec:
prove each can "return something useful and reliable," else mark the Timing
signal "manual input for now."

**Verdict in one line:** all three are **viable** — none needs to drop to
manual-for-now — but each ships with a hard constraint that must shape Phase 1.

---

## 0. Environment finding that affects all three (read first)

This execution environment enforces a **network egress allowlist**. Every
external data host a production probe would want is **blocked**:

| Host | Purpose | Status |
|---|---|---|
| api.gdeltproject.org | news-volume time-series (ideal velocity source) | BLOCKED |
| en.wikipedia.org (pageviews API) | attention time-series | BLOCKED |
| trends.google.com | search-interest velocity | BLOCKED |
| reddit.com / newsapi.org | volume signals | BLOCKED |

The **only** web access available is the harness `WebSearch`/`WebFetch` tools.
This mirrors how the real Timing agent will run anyway: through the Anthropic
**server-side `web_search` tool**. So the spikes test the *logic and
reliability* against live WebSearch data, and the verdicts below are valid for
a web_search-backed implementation.

**Phase 1 decision needed:** either (a) allowlist GDELT + Wikipedia-pageviews
(both free, keyless, reliable) to unlock *numeric* velocity, or (b) ship the
search-only ordinal version below. Recommend (a) — it's cheap and turns
velocity from ordinal into quantitative.

---

## 1. velocity_probe — VERDICT: WORKS (ordinal), numeric upgrade available

**What it returns reliably:** an ordinal verdict
`{SURGING, STEADY, DECLINING, DEAD}` + confidence + rationale, derived from
recency-of-dates and momentum-vs-decline language in search results.

**Live test (4 topics, real WebSearch data):**

| Topic | Verdict | Why |
|---|---|---|
| Figure 03 production | SURGING | all 2026, momentum=9 ("ramps","24x","milestone") |
| Hangzhou robot police | SURGING | all 2026, "first","begins operations" |
| Roomba | STEADY | fresh 2026 news but maintenance language, momentum=1 |
| Honda ASIMO | DECLINING | newest 2025, decline=5 ("retired","discontinued") |

**Key finding — volume ≠ velocity.** Roomba has *fresh, plentiful* 2026 news
yet is not accelerating. A naive "count recent results" probe would have
mislabeled it. The signal that actually works is **date-recency + momentum
lexicon**, not volume.

**What does NOT work from search alone:** a *precise numeric* rate-of-change
("attention up 340% week-over-week"). That needs a time-series API (GDELT /
Wikipedia pageviews) — currently blocked. Ordinal is enough for the Timing
agent to gate on; numeric is a nice-to-have unlocked by allowlisting.

**Spike caveats to carry into Phase 1:**
- DEAD vs DECLINING boundary is sensitive to stray recent name-mentions
  (ASIMO → DECLINING because "Asimo OS" appeared in 2025). Acceptable/honest,
  but the threshold is tunable.
- Thin-evidence gate (`<4 results → needs_human`) is independent of
  separation confidence, so a high-confidence call can still flag for human
  review on small samples. Correct behavior; just be aware it fires often on
  narrow queries.

---

## 2. narrative_gap — VERDICT: WORKS, and gaps are verifiable

**What it returns reliably:** per-candidate-angle coverage **density** + a
gap flag, ranked. Candidate angles = the brand pillars.

**Live test (topic: Figure 03 ramp):** coverage is ~75% throughput/production.
Probe flagged as gaps: `incident_file` (0.0), `deployment_reality` (0.0),
entity-specific `labor impact` (0.0), `company_intel` (0.25).

**Verification (the important part — are the gaps real or hallucinated?):**
- Searched the top proposed gap — *"Figure 03 malfunction/injury/incident."*
  Confirmed **genuinely under-covered**: the viral factory-malfunction footage
  is **Unitree H1, not Figure**; Figure 02→03 had only "minor forearm issues."
  So "Figure incident file" is real white space, not a hallucination. ✓
- Searched *"BotQ factory labor displacement."* Found **tons of generic**
  automation-and-jobs research but **nothing BotQ-specific**.

**Key finding — entity-specific vs topic-generic.** The labor angle looks
"covered" at the topic level yet is wide open for the *specific entity*. The
probe must (and does) distinguish these, or it will wrongly report saturated
white space. This is the single most important reliability rule for this tool.

---

## 3. citation_monitor — VERDICT: WORKS, but REQUIRES a disambiguation guard

**What it returns reliably:** for a target query, the set of sources an answer
engine cites, our brand's presence/rank/share-of-voice, the incumbent set to
displace, and a zero-state baseline.

**Live test:**
- *"best physical-AI sources"* → engine names specific domains: **The Robot
  Report, Robotics & Automation News, CNBC, AI Magazine, NVIDIA**. Madre de
  Máquinas **absent** → correct zero-state baseline + a concrete competitive
  set. ✓
- *"Madre de Maquinas physical AI"* → brand absent organically; engine even
  says it "doesn't appear to be a specific project," and the name **collides
  with an MTG card** ("Elesh Norn, Mother of Machines").

**Key finding — false-positive risk is real and was reproduced.** The spike's
naive substring presence check returned `brand_present=True` for case 2 by
matching the brand name *inside the MTG-card context* — the wrong entity. The
disambiguation flag fired alongside it. **Takeaway:** presence detection must
be disambiguation-aware (scope queries with brand-specific qualifiers, verify
the matched context is actually the brand) or the monitor will track the wrong
"Madre de Máquinas." True "does ChatGPT/Perplexity/Google-AI Overview cite us"
needs those answer-engine endpoints; the web_search proxy is reliable for
**share-of-voice and incumbent mapping**, which is the actionable part.

---

## Recommendation for Phase 1

| Tool | Ship as | Hard requirement carried forward |
|---|---|---|
| velocity_probe | ordinal (SURGING/STEADY/DECLINING/DEAD) + confidence | date-recency + momentum lexicon; optional GDELT/Wiki numeric if allowlisted |
| narrative_gap | density-per-pillar gap ranker | MUST separate entity-specific vs topic-generic white space |
| citation_monitor | presence + share-of-voice + incumbents + baseline | MUST be disambiguation-aware before trusting "present/absent" |

No Timing signal needs to be downgraded to "manual input for now." Two
infra asks before Phase 1: **(1)** decide on allowlisting GDELT +
Wikipedia-pageviews for numeric velocity; **(2)** confirm the Timing agent
will call the Anthropic server-side `web_search` tool as the data layer for
all three.

*These files are throwaway spikes. The scoring logic is real and runs offline
against captured 2026-06-15 WebSearch data; the network layer is stubbed.*
