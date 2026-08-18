# X Agent Pipeline — Complete Project Spec

> **RECONSTRUCTION NOTICE**: This file is **not** a copy of an original source
> document. No such file ever existed in this repository or anywhere on the
> filesystem this project was built in — the spec was pasted directly into a
> chat conversation with Claude Code across several messages, and lived only
> in that conversation's history until now. This file was assembled from that
> chat history, plus corroborating references already committed to the
> codebase (docstrings, commit messages, and comments in `orchestrator.py`,
> `models.py`, and elsewhere that quote or paraphrase spec requirements).
>
> Treat this as a best-effort reconstruction, not an authoritative original.
> Sections are flagged below with a confidence note where anything is
> uncertain, paraphrased rather than verbatim, or inferred rather than
> directly recalled. If anything here conflicts with a real original
> document should one surface later, the original wins.
>
> Reconstructed: 2026-08-18.

---

## DECISION RECORD

*(High confidence — this table was quoted and worked from directly and repeatedly throughout the migration.)*

These decisions were marked final in the original spec ("Do not re-litigate them").

| Decision | Choice | Reason |
|---|---|---|
| X account | Keep "Madre de Máquinas" | 4 followers = no equity to lose, credentials already wired |
| Identity | Full pivot to defense-tech/DoD intelligence | Clean break, no mixed feed |
| Codebase | Keep existing foundation, migrate | Orchestrator pattern is correct, no reason to rebuild |
| Auto-post | Never — human approval required at Distribution | Legal risk, credibility risk, accuracy requirements |
| Safety mechanisms | Preserve both (citation/hedging + narrative-conflict) | More important for this identity than the old one |
| Reply blitz | Phase 5, not bundled with migration | Too much new surface area, defer |
| First posts | Write manually before running pipeline | Need to feel the voice before automating it |

---

## ACCOUNT IDENTITY

*(High confidence.)*

**Thesis:**
> "Every post answers the question a defense tech investor, an AI researcher, or a national security analyst would ask at 7am: what did the US government just commit money to, and what does it mean?"

**One-line identity:**
> "The account that finds what's being built before it's announced."

**X bio (to set manually):**
> "Tracking what the US government is paying for in AI and autonomous systems — before it's news. Primary sources only: SAM.gov, DARPA, patents, Congress."

**Manual account changes (Phase 0, before any code work):**
- Update X bio to the text above
- Update display name to reflect the new identity
- Delete or archive prior posts that don't fit
- Do NOT announce the pivot — just start posting new content
- Write first 2–3 posts manually before running the pipeline

---

## PHASE TRACKER

*(High confidence — this table's exact wording drove the whole migration's build order.)*

| Phase | What | Status |
|---|---|---|
| Phase 0 | Account identity changes (manual) | Do first |
| Phase 1 | API integrations — SAM.gov, Federal Register, Congress.gov | Do before any prompt/schema changes |
| Phase 2 | Schema migration — new Pydantic models, Orchestrator rewiring, test rebuilds | Do after Phase 1 |
| Phase 3 | Safety preservation — port citation/hedging + narrative-conflict into new agents | Do during Phase 2 |
| Phase 4 | New features — human approval gate, two-call posting sequence | Do after Phase 2 |
| Phase 5 | Reply blitz — read endpoints, suggestion queue, cron trigger | Separate build, after Phase 4 |

**Status as of this reconstruction:**
- Phase 0: not tracked in this codebase (manual, human-side action)
- Phase 1: done, merged (SAM.gov, Federal Register, Congress.gov, arXiv, DARPA-fallback integrations)
- Phase 2: done, merged (schema migration, Orchestrator rewiring, all 7 agent files, test suite rebuild)
- Phase 3: done — folded into Phase 2's execution per its own "Do during Phase 2" status (citation_hedge_required, narrative_conflict_flag/note, hard-halt routing all shipped as part of Phase 2)
- Phase 4: **not started** — human-approval-gate *mechanism* exists (`StagedPost.requires_human_approval: Literal[True]`), but the dashboard rebuild and the two-call posting sequence's specific error-handling table are not built
- Phase 5: not started

---

## PHASE 1 — API INTEGRATIONS

*(High confidence.)*

Build these as new tools available to the Research agent alongside existing `web_search`. Each is a separate integration, built and tested independently before moving to Phase 2.

### 1a. SAM.gov API
- Endpoint: `https://api.sam.gov/opportunities/v2/search`
- Auth: free API key from sam.gov/profile
- Key parameters: `postedFrom`, `postedTo`, `keywords`, `deptname` (filter to DoD)
- What to pull: contract awards posted in the last 24 hours matching keywords: "artificial intelligence", "autonomous systems", "machine learning", "robotics", "unmanned"
- Extract per result: `title`, `awardee.name`, `award.amount`, `description`, `uiLink` (source URL), `postedDate`
- Highest-priority integration — the whole angle depends on it

### 1b. Federal Register API
- Endpoint: `https://www.federalregister.gov/api/v1/articles`
- Auth: none required
- Key parameters: `conditions[term]`, `conditions[publication_date][gte]`, `conditions[agencies][]`
- What to pull: rules and proposed rules mentioning "artificial intelligence" in the last 7 days
- Extract per result: `title`, `agency_names`, `abstract`, `html_url`, `publication_date`

### 1c. Congress.gov API
- Endpoint: `https://api.congress.gov/v3/bill`
- Auth: free API key from api.congress.gov
- Key parameters: `q` (search), `fromDateTime`, `sort`
- What to pull: bills introduced or amended in the last 7 days mentioning "artificial intelligence" or "autonomous weapons"
- Extract per result: `title`, `introducedDate`, `sponsors`, `latestAction.text`, `url`

### 1d. DARPA solicitations
- No official API — assess scrapeability of darpa.mil/news-events/solicitations
- If scraping is unreliable, fall back to `web_search` with `site:darpa.mil`
- Lower priority than 1a–1c

### 1e. arXiv
- Endpoint: `https://export.arxiv.org/api/query`
- Auth: none required
- Key parameters: `search_query` (`cat:cs.RO OR cat:cs.AI`), `start`, `max_results`, `sortBy=submittedDate`
- Filter: submitted in the last 48 hours
- Extract: `title`, `authors`, `summary`, `id` (source URL), `published`

### Research agent tool registration
After each API is built, register it as a named tool the Research agent can call explicitly rather than relying on `web_search` to find the same data:
`search_sam_gov(keywords, days_back)`, `search_federal_register(keywords, days_back)`,
`search_congress(keywords, days_back)`, `search_arxiv(categories, days_back)`,
`search_darpa_solicitations()` (no params, just latest).

*Note: the actual implementation resolved 1d differently than a literal reading of "assess scrapeability" — `darpa_solicitations.py` ships only the `web_search` fallback query, on the grounds that this dev environment couldn't fetch a live page sample to verify a scraper against. See that module's own docstring.*

---

## PHASE 2 — SCHEMA MIGRATION

*(High confidence for the models and order-of-operations; the actual implementation added structural validators — no-URL-in-body, thread_tweets 280-char cap, computed verdict/total, etc. — beyond what's shown here as bare field lists. See `models.py` itself for the as-built, validator-complete versions.)*

**Order of operations — do not skip steps:**
1. Define new Pydantic models (below)
2. Update Orchestrator wiring to match new field names
3. Update all test files against new models
4. Only then replace system prompt text with the Phase 2 prompts
5. Run tests — do not proceed until all pass

### New Pydantic models (as originally specified)

**ResearchItem**
```python
class ResearchItem(BaseModel):
    source: Literal["SAM.gov", "DARPA", "Patent", "arXiv", "Congress", "FedRegister", "JobSignal"]
    date_found: date
    headline: str
    raw_detail: str
    novelty_score: int  # 1-10
    post_angle: str
    source_url: HttpUrl
```

**TimingDecision**
```python
class TimingDecision(BaseModel):
    item_id: str
    recommended_slot: Literal["7AM", "9AM", "12PM", "3PM", "6PM", "reject"]
    rejection_reason: str | None
    urgency_note: str
    citation_hedge_required: bool  # preserved from existing Timing agent
```

**StrategyOutput**
```python
class StrategyOutput(BaseModel):
    chosen_angle: str
    format: Literal["short_hook", "pov_post", "thread", "data_drop"]
    must_include: list[str]  # 1-3 items
    must_avoid: list[str]
    hashtags: list[str]  # 0-2 items
    thread_spine: list[str] | None
    narrative_conflict_flag: bool  # preserved from existing Strategy agent
    narrative_conflict_note: str | None
```

**SkepticOutput**
```python
class SkepticOutput(BaseModel):
    verdict: Literal["approved", "revise", "reject"]
    critique: str
    revised_angle: str | None
    revised_must_include: list[str] | None
```

**CopyOutput**
```python
class CopyOutput(BaseModel):
    main_post: str          # <=280 chars, NO URL
    reply_link: str         # format: "Source: [URL]"
    thread_tweets: list[str] | None
    format_used: Literal["short_hook", "pov_post", "thread", "data_drop"]
    char_count: int
    hashtags_used: list[str]
    citation_hedged: bool   # True if Timing flagged citation_hedge_required
```

**QualityOutput**
```python
class QualityOutput(BaseModel):
    scores: dict[str, int]  # keys: source_specificity, differentiation, hook_strength, format_compliance, thesis_alignment
    total: int              # sum, max 50
    verdict: Literal["pass", "revise", "reject"]
    notes: list[str]        # one per dimension scored below 8
    blocking_dimension: str | None
    approval_note: str      # what human reviewer should check, or "Looks clean."
```

**StagedPost**
```python
class StagedPost(BaseModel):
    main_post: str
    reply_link: str
    thread_tweets: list[str] | None
    format_used: str
    scheduled_slot: str
    source_url: HttpUrl
    novelty_score: int
    quality_scores: dict[str, int]
    approval_note: str
    requires_human_approval: bool = True   # never False — no auto-post path
    approved_at: datetime | None = None
    approved_by: str | None = None
```

---

## PHASE 3 — SAFETY PRESERVATION

*(High confidence.)*

Both mechanisms must survive the migration. Port them explicitly — do not assume they carry over.

### Citation/hedging check (was in the Timing agent)
**What it does:** detects when a research item's claims need hedging before posting.
**Where it lives after migration:** Timing agent, as `citation_hedge_required: bool` on `TimingDecision`. Copy agent reads this field and applies hedging language if True.
**How Copy handles it:** if `citation_hedge_required` is True, Copy must add a source qualifier to the post ("according to SAM.gov", "per the solicitation", "documents suggest"). Set `citation_hedged: True` in `CopyOutput`.

### Narrative-conflict check (was in the Strategy agent)
**What it does:** checks the item against memory/vector store to block posts that contradict the account's prior stated positions.
**Where it lives after migration:** Strategy agent. `narrative_conflict_flag: bool` + `narrative_conflict_note: str | None` on `StrategyOutput`. If flagged, the Orchestrator halts the job and logs the conflict — does not proceed to Skeptic.
**Why this matters more now:** a defense-intel account that contradicts its own prior analysis of a DoD program destroys credibility permanently. This check is not optional.

---

## PHASE 4 — NEW FEATURES

*(High confidence for scope and the error-handling table; medium confidence on exact wording in a couple of sub-bullets, noted inline.)*

Build after Phase 2 + Phase 3 are complete and tests pass.

### 4a. Human approval gate in Distribution
The pipeline always halts at Distribution. No auto-post fallback exists.

Review dashboard must surface, **in this order**:
1. `approval_note` from Quality agent — read this first
2. `main_post` text
3. `reply_link`
4. `quality_scores` breakdown
5. `source_url` — click to verify the claim directly before approving

Only after manual Approve does Distribution proceed to posting.

### 4b. Two-call posting sequence
Reuses existing thread-chaining code.

For standard posts:
```
Step 1: POST main_post -> capture tweet_id
Step 2: wait 5-10 seconds
Step 3: POST reply_link as reply (in_reply_to_tweet_id = tweet_id)
```

For threads:
```
Step 1: POST thread_tweets[0] -> capture tweet_id_1
Step 2: POST thread_tweets[1] as reply to tweet_id_1 -> capture tweet_id_2
... chain each tweet as reply to previous ...
Final step: POST reply_link as reply to last tweet_id
wait 5-10 seconds between each call
```

Error handling:
- Rate limit -> queue and retry after 15 minutes
- 403/401 -> halt all posting, log alert, do not retry (auth issue)
- Duplicate content error -> skip, log, do not retry
- Never post the same `main_post` content within 72 hours

*Note: the as-built `distribution.stage_v2()` gets the sequential chaining behavior "for free" by reusing the existing `XAdapter`/`x_client.py::post_thread()`, which already chains every post to the real previous tweet ID regardless of any flag in the payload. The specific error-handling table above (15-minute rate-limit queue, hard-stop-no-retry on 401/403, skip-no-retry on duplicate content, 72-hour same-**content** dedup at publish time) is NOT yet built — the current `confirm_publish()` path uses a pre-existing generic retry/circuit-breaker (3 retries, exponential backoff) that does not distinguish these error types. `PostHistoryStore`'s duplicate check is also not the same mechanism: it dedupes by `source_url` at **stage** time (for Timing's benefit), not by post content at **publish** time.*

---

## PHASE 5 — REPLY BLITZ (SEPARATE BUILD)

*(High confidence.)*

Do not bundle with Phase 4. Standalone feature build.

### What needs to be built
1. Read/timeline endpoints added to `x_client.py` — currently write-only
2. Reply suggestion queue — surfaces to the review dashboard, human approves each one (not auto-post)
3. State tracking — store replied-to tweet IDs, prevent duplicate replies
4. Daily cron trigger at 8PM ET — separate from the main pipeline

### Reply rules (for suggestion-queue logic)
- Only surface posts from the last 4 hours
- Reply must add a specific fact or counterpoint — flag and exclude agreement-only suggestions
- Suggested reply length: 100–200 characters
- No links in replies
- Max 10 suggestions per session
- Do not suggest the same target account twice in one day

### Target accounts (priority order)
@sama, @karpathy, @ylecun, @GaryMarcus, @benedictevans, @PalmerLuckey, @ChrisJBakke, @pmarca

### Why this matters
The X algorithm weights replies roughly 15x higher than likes. This is the primary growth mechanism for a new account. It is the human replying with their own voice, surfaced by the agent — not the agent auto-replying.

---

## CONTENT RULES (enforced across all agents)

*(High confidence.)*

### What this account covers
- DoD contract awards for AI, robotics, autonomous systems
- DARPA solicitations and program announcements
- Defense tech startup signals (funding, hiring velocity, patents)
- Congressional and regulatory moves affecting defense AI
- Physical AI deployment in real industries (manufacturing, logistics, agriculture)
- Technical research with real-world defense/autonomy implications

### What this account never covers
- Consumer AI products (ChatGPT features, Gemini updates, Claude releases)
- Generic AI news already on TechCrunch, Wired, The Verge
- Opinion with no factual primary-source anchor
- Hype or promotional framing for any company

### Banned words (all agents reject on sight)
revolutionary, groundbreaking, game-changing, transformative, unprecedented, exciting, powerful, amazing, disruptive (used as a positive descriptor)

### Character rules
- Standard post: 280 characters max
- URLs count as 23 characters regardless of length
- Emojis count as 2 characters each
- Ideal hook: 71–100 characters
- Thread tweets: 200–270 characters each
- Hashtags: 1–2 max per post, from the approved list only
- Links: never in the main post body — always in the first reply

### Approved hashtag list
`#AI #PhysicalAI #Robotics #DefenseTech #AutonomousSystems #AIAgents #FutureOfWar #DARPA`

---

## POSTING SCHEDULE

*(High confidence for the table; the spec does not specify what actually calls `run_sweep()` on this cadence — see the open gap noted below.)*

| Slot | Time ET | Format | Notes |
|---|---|---|---|
| Morning hook | 7:00 AM | short_hook | Most punchy item of the day |
| Morning POV | 9:00 AM | pov_post | React to overnight developments |
| Midday thread | 12:00 PM | thread | Deepest item of the day |
| Afternoon drop | 3:00 PM | data_drop | Contract award, funding, job signal |
| Evening POV | 6:00 PM | pov_post | End-of-day analysis |
| Reply blitz | 8:00 PM | — | Phase 5 feature, manual for now |

**Known gap (not addressed anywhere in the spec as reconstructed):** nothing specifies what triggers `run_sweep()` on this cadence, or how often it should run to keep these slots populated. The old `calendar_cron.py` handled scheduling for the now-retired single-topic model and was deleted outright during the Phase 2 migration rather than adapted, since its entire shape (a `topic_provider` returning one topic) doesn't apply to a sweep with zero input. A schedule-based trigger that calls `run_sweep()` on a cadence would be new code — not specified here, not built yet.

---

## PHASE 2 AGENT SYSTEM PROMPTS (target state, as originally specified)

*(High confidence on content; note that the as-shipped prompts in `research.py`/`timing.py`/`strategy.py`/`skeptic.py`/`copy_agent.py`/`quality.py` diverge from these in real ways discovered during implementation — most notably: Research's prompt was adapted for a pre-fetch + single-batch-scoring shape rather than one call per source; Timing's prompt only needs to emit the fields DuplicateCheck/topic-recency don't already answer at the code level; Strategy's prompt no longer needs to self-report `narrative_conflict_flag` at all, since that's computed in code from the existing checker; Quality's prompt doesn't request `verdict`/`total`/`blocking_dimension` at all, since those are computed by `QualityOutput`'s own validator; Copy's prompt never asks for `char_count`. The as-shipped prompts are the source of truth for what's actually running — see each agent file directly.)*

### engine/agents/research.py (original)
```
You are the Research Agent for an X (Twitter) account covering physical AI, defense tech,
and autonomous systems.
Your only job: find stories that haven't been reported yet — or that mainstream tech press
will cover in 24-72 hours but hasn't touched yet. You are looking for primary source
signals, not news summaries.
TOOLS AVAILABLE (use in this priority order):
1. search_sam_gov — DoD contract awards from last 24 hours
2. search_federal_register — AI-related rules and proposed rules from last 7 days
3. search_congress — Bills mentioning AI or autonomous weapons from last 7 days
4. search_arxiv — cs.RO and cs.AI papers from last 48 hours
5. search_darpa_solicitations — latest DARPA BAAs and solicitations
6. web_search — fallback only, for job signals and patent searches
SEARCH TERMS TO USE:
SAM.gov: "artificial intelligence", "autonomous systems", "machine learning", "robotics", "unmanned"
Federal Register: "artificial intelligence", "autonomous weapons", "unmanned systems"
Congress.gov: "artificial intelligence", "autonomous weapons"
arXiv: categories cs.RO and cs.AI, sorted by submission date
Patents (via web_search): "autonomous weapon", "drone swarm", "physical AI", "humanoid robot", "AI targeting"
Job signals (via web_search): defense/robotics companies posting 5+ technical roles simultaneously
FOR EACH ITEM FOUND, output a ResearchItem with exactly these fields:
- source, date_found, headline, raw_detail, novelty_score, post_angle, source_url
NOVELTY SCORING GUIDE:
- 9-10: Contract award or patent from a company with no press coverage
- 7-8: DARPA solicitation or Congressional move not yet in tech press
- 5-6: arXiv paper with real-world implications most people missed
- 3-4: Story lightly covered but angle is fresh
- 1-2: Already reported by TechCrunch, Wired, or The Verge
Only output items with novelty_score 6 or higher. Discard the rest.
Do not editorialize. Do not summarize existing news articles. Do not pull from OpenAI,
Anthropic, or Google product announcements — that is not this account's lane.
```

### engine/agents/timing.py (original)
```
You are the Timing Agent for an X (Twitter) posting pipeline.
You receive a list of ResearchItems that passed novelty scoring. Your job: decide WHEN
each item should post and WHETHER narrative conditions are right to post it now.
POSTING SLOTS (ET): 7AM short_hook, 9AM pov_post, 12PM thread, 3PM data_drop, 6PM pov_post.
FOR EACH ITEM, run these checks:
1. velocity_check: accelerating or decelerating?
2. narrative_gap_check: posted on this topic in the last 48h? Skip unless the new item
   directly contradicts or advances the prior post.
3. duplicate_check: same source_url in a post in the last 72h? Reject entirely.
4. citation_hedge_required: preliminary/inferred/single-source claim? Flag True.
5. slot_assignment: which slot fits this item's format and urgency?
OUTPUT a TimingDecision per item. Assign at most ONE item per slot per day; on collision,
pick the higher novelty_score.
```

### engine/agents/strategy.py (original)
```
You are the Strategy Agent for an X (Twitter) posting pipeline.
You receive a single ResearchItem that passed timing checks. Choose the exact angle and
set constraints Copy will use to write the post.
ACCOUNT THESIS — never deviate: primary-source intelligence on physical AI, defense tech,
and autonomous systems, before it becomes mainstream news.
ANGLE PRIORITY: (1) unreported build, (2) contract/funding signal, (3) failure/setback
contradicting hype, (4) policy/regulatory move, (5) technical paper with real-world stakes.
FORMAT SELECTION: short_hook / pov_post / thread / data_drop, per character targets.
NARRATIVE CONFLICT CHECK — run before outputting: compare against prior staked positions
in memory; if this would contradict a prior assertion, set narrative_conflict_flag = True
and describe the conflict. The Orchestrator halts the job if this flag is True.
OUTPUT a StrategyOutput.
```

### engine/agents/skeptic.py (original)
```
You are the Skeptic Agent for an X (Twitter) posting pipeline. You adversarially review
Strategy's output before Copy writes anything. You are looking for reasons to reject or
revise — not reasons to approve.
HARD REJECTION (any one = reject): angle claims something the source doesn't say; post is
indistinguishable from generic AI news; story already covered by major tech press in the
last 6 hours; must_include has no verifiable identifier; topic is a consumer AI product;
angle is opinion with no factual anchor.
REVISION (send back with notes): format undersells the story; the angle buries the most
surprising fact; must_include is missing the most specific detail; hashtags too generic.
BANNED WORDS: flag for revision if present.
OUTPUT a SkepticOutput.
```

### engine/agents/copy_agent.py (original)
```
You are the Copy Agent for an X (Twitter) posting pipeline. You receive Strategy's
approved constraints and write the actual post.
HARD CHARACTER RULES, LINK RULE (never a URL in main_post — non-negotiable), HASHTAG RULE,
EMOJI RULE, CITATION HEDGING (hedge when citation_hedge_required is True), VOICE (direct,
specific, slightly skeptical, analyst-texting-a-friend, never hype).
FORMAT TEMPLATES for short_hook / pov_post / thread / data_drop.
OUTPUT a CopyOutput.
```

### engine/agents/quality.py (original)
```
You are the Quality Agent for an X (Twitter) posting pipeline. Final gate before
Distribution. Score the post on 5 dimensions (0-10 each): SOURCE_SPECIFICITY,
DIFFERENTIATION, HOOK_STRENGTH, FORMAT_COMPLIANCE (character limits, URL-in-body,
hashtag count, banned words), THESIS_ALIGNMENT.
GATE LOGIC: all scores >=7 -> pass. Any 5-6, none below 5 -> revise. Any below 5 -> reject.
OUTPUT a QualityOutput including an approval_note flagging anything a human reviewer
should double-check before approving.
```

### engine/agents/distribution.py (no LLM — original implementation notes)

*Note on this section's confidence: this account's content came from an earlier, less complete pasted message than the Phase 4a bullet list above, and the two disagree on ordering — flagging rather than silently picking one. The earlier message's distribution.py notes listed the dashboard fields as main_post, reply_link, quality_scores, approval_note, source_url (approval_note fourth). The later, more complete spec's Phase 4a section (reproduced above) explicitly numbers approval_note first. The Phase 4a version is treated as authoritative in this reconstruction since it's the more complete, more recently stated version — but the discrepancy itself is real, not invented.*

```
HUMAN APPROVAL — DEFAULT BEHAVIOR: the pipeline ALWAYS halts here. No auto-post path.
StagedPost.requires_human_approval: bool = True (never False).
Review dashboard should surface: main_post text, reply_link, quality_scores from Quality
agent, approval_note from Quality agent (read this first before approving), source_url to
verify the claim directly. [See ordering discrepancy note above.]
TWO-CALL POSTING SEQUENCE — runs only after human approval (see Phase 4b above).
REPLY BLITZ — Phase 5, not built yet. Placeholder only.
```
