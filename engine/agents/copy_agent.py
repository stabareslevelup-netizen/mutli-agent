"""
engine/agents/copy_agent.py — Copy agent [Phase 2, X-agent migration].

One LLM call writes the whole post (or the whole thread, when format
requires thread_tweets) -- consistent with the one-call-per-logical-unit
pattern every other agent in this migration uses. A per-segment loop would
be N times the cost for a long thread and would fragment cross-tweet
coherence (each call would need increasingly large prior-tweet context to
stay consistent).

Two fields are code-computed, never trusted to the LLM:
  - char_count: never asked of the model at all -- the prompt gives target
    length ranges instead, and this recomputes it from len() post-hoc.
  - citation_hedged: set directly from timing.citation_hedge_required
    (ground truth already known from Timing's decision) -- the LLM's job is
    to WRITE the hedging language when required, not self-certify that it
    did.

For thread format, main_post is code-normalized to thread_tweets[0] after
validation -- defensive consistency, not trusted to the LLM getting the
duplication right unprompted.

No-URL-in-body, the hashtag allowlist, and banned words are all already
schema-validated on CopyOutput (see models.py) -- this prompt states them
directly too, rather than relying on the validator as the only line of
defense.
"""
from __future__ import annotations

import json
from typing import Optional

from engine.agents.base import BaseAgent
from engine.core.models import CopyOutput, StrategyOutput, TimingDecision

_SYSTEM = (
    "You are the Copy Agent for an X posting pipeline. You receive Strategy's approved "
    "constraints and write the actual post.\n\n"
    "LINK RULE -- NON-NEGOTIABLE, STATED PLAINLY: Never include a URL in main_post or in "
    "any thread_tweets element. Not ever, not even as a courtesy, not even the source "
    "link. The source link is handled entirely separately (X's own link-card / a chained "
    "reply) -- your job is text only. A post with a URL in this field will be rejected "
    "before it ever reaches a human.\n\n"
    "HASHTAG RULE: Maximum 2 hashtags, only from this exact list: #AI, #PhysicalAI, "
    "#Robotics, #DefenseTech, #AutonomousSystems, #AIAgents, #FutureOfWar, #DARPA. Any "
    "other hashtag will be rejected.\n\n"
    "BANNED WORDS -- never use these: revolutionary, groundbreaking, game-changing, "
    "transformative, unprecedented, exciting, powerful, amazing, disruptive (as a "
    "positive descriptor). Use of any of these will send this back for revision.\n\n"
    "TARGET LENGTHS (aim for these -- you do not need to count exactly, just write to "
    "the target):\n"
    "- short_hook: 71-100 characters\n"
    "- pov_post: 150-240 characters\n"
    "- data_drop: 200-260 characters\n"
    "- thread tweets: 200-270 characters EACH\n"
    "Every field has a hard ceiling of 280 characters regardless of target -- X rejects "
    "anything longer.\n\n"
    "EMOJI RULE: only a thread-marker at the end of a thread's first tweet, and only a "
    "breaking-news marker for something that happened in the last 2 hours. No other "
    "emojis.\n\n"
    "CITATION HEDGING: if citation_hedge_required is true (given below), the post's "
    "claim rests on something preliminary, inferred, or a single unverified document -- "
    "you MUST include a source qualifier in the text itself (e.g. \"according to "
    "SAM.gov\", \"per the solicitation\", \"documents suggest\"). Never present a hedged "
    "claim as flatly established fact.\n\n"
    "VOICE: direct, specific, slightly skeptical. Facts carry the weight. You are an "
    "analyst texting a smart friend, not a journalist writing a headline. Never hype. "
    "Lead with the most specific fact -- dollar amount, company name, headcount, "
    "contract number. Never start with \"I\" or \"We\". Every post needs at least one "
    "verifiable specific.\n\n"
    "THREAD HANDLING: if format is 'thread', write the COMPLETE thread as thread_tweets "
    "(every tweet, including the first/hook tweet) -- do not write a separate, "
    "different main_post for a thread; main_post should just repeat thread_tweets[0]. "
    "Do NOT include a source-citation tweet as the last element of thread_tweets -- that "
    "gets posted separately, automatically, after your thread. For any other format, "
    "leave thread_tweets out entirely and just write main_post.\n\n"
    "Return ONLY a JSON object, no prose, no fences:\n"
    '{"main_post":"<the post, or thread_tweets[0] if format is thread>",'
    '"reply_link":"Source: [restate the source_url given below]",'
    '"thread_tweets":["<all tweets in order, ONLY if format is thread, omit otherwise>"],'
    '"format_used":"short_hook|pov_post|thread|data_drop",'
    '"hashtags_used":["<0-2 from the approved list, or omit>"]}'
)

_USER = """Strategy's constraints:
chosen_angle: {chosen_angle}
format: {format}
must_include: {must_include}
must_avoid: {must_avoid}
hashtags: {hashtags}
thread_spine: {thread_spine}

citation_hedge_required: {hedge}
{quality_block}
Return the JSON object as specified."""


class CopyAgent(BaseAgent):
    name = "copy"

    async def write_v2(self, *, strategy: StrategyOutput, timing: TimingDecision, job_id: str,
                       quality_notes: Optional[list[str]] = None) -> CopyOutput:
        quality_block = (
            f"\nQuality flagged issues with your previous attempt -- fix these:\n"
            f"{json.dumps(quality_notes)}\n" if quality_notes else "")
        user = _USER.format(
            chosen_angle=strategy.chosen_angle, format=strategy.format.value,
            must_include=json.dumps(strategy.must_include),
            must_avoid=json.dumps(strategy.must_avoid),
            hashtags=json.dumps(strategy.hashtags),
            thread_spine=json.dumps(strategy.thread_spine) if strategy.thread_spine else "null",
            hedge=timing.citation_hedge_required, quality_block=quality_block)
        raw = await self._complete_json(system=_SYSTEM, user=user, job_id=job_id, max_tokens=2000)

        # citation_hedged is ground truth from Timing, not LLM self-report
        if isinstance(raw, dict):
            raw["citation_hedged"] = timing.citation_hedge_required

        out = await self._validate(CopyOutput, raw, job_id=job_id, step="copy.write_v2")

        # defensive consistency: force main_post = thread_tweets[0] for thread
        # format, regardless of what the LLM put in main_post
        if out.thread_tweets:
            out = out.model_copy(update={"main_post": out.thread_tweets[0]})

        return out
