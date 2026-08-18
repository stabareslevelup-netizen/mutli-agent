"""
engine/agents/quality.py — Quality agent [Phase 2, X-agent migration].

verdict/total/blocking_dimension are computed by QualityOutput's own
_compute_gate model_validator (models.py) from the 5 scores -- never asked
of the LLM, same "hard invariant, not self-report" pattern as
CopyOutput.char_count. The requested JSON shape below only asks for scores
+ notes + approval_note.

Pure evaluator: no side effects, no retry awareness, no knowledge that a
'reject' triggers a fresh Strategy attempt vs a 'revise' triggers a
Copy-only retry -- that routing lives entirely in orchestrator.py's
existing loop, which reads .verdict off what this returns.

format_compliance's only remaining real job is a banned-word check --
char-limit, URL-in-body, and hashtag-count are already structurally
guaranteed by CopyOutput's own validators before Quality ever sees it.
"""
from __future__ import annotations

import json

from engine.agents.base import BaseAgent
from engine.core.models import CopyOutput, QualityOutput

_SYSTEM = (
    "You are the Quality Agent for an X posting pipeline. Score the post Copy wrote "
    "on 5 dimensions, 0-10 each.\n\n"
    "1. SOURCE_SPECIFICITY: does the post contain a verifiable primary-source detail -- "
    "a dollar amount, contract number, company name + headcount, a named program, a "
    "date? 10 = specific and verifiable. 0 = pure opinion, nothing to look up.\n\n"
    "2. DIFFERENTIATION: would a generic AI news account post this exact thing today? "
    "10 = nobody else is covering this angle. 0 = identical to what any AI news account "
    "would write.\n\n"
    "3. HOOK_STRENGTH: does the first sentence make a smart person stop scrolling? "
    "10 = counterintuitive, specific, immediate curiosity. 0 = generic statement anyone "
    "could have written.\n\n"
    "4. FORMAT_COMPLIANCE: character limits, URL-in-body, and hashtag-count are already "
    "guaranteed correct by the time you see this -- don't re-check them. Score this "
    "dimension ONLY on whether any of these banned words appear anywhere in the text: "
    "revolutionary, groundbreaking, game-changing, transformative, unprecedented, "
    "exciting, powerful, amazing, disruptive (as a positive descriptor). 10 = none "
    "present. Below 5 if any are present.\n\n"
    "5. THESIS_ALIGNMENT: does this fit the account's thesis -- \"primary-source "
    "intelligence on physical AI, defense tech, and autonomous systems, before it "
    "becomes mainstream news\"? 10 = perfectly on-thesis. 0 = consumer AI product or "
    "opinion with no source anchor.\n\n"
    "Return ONLY a JSON object, no prose, no fences:\n"
    '{"scores":{"source_specificity":<0-10>,"differentiation":<0-10>,'
    '"hook_strength":<0-10>,"format_compliance":<0-10>,"thesis_alignment":<0-10>},'
    '"notes":["<one sentence per dimension you scored below 8, explaining what is weak>"],'
    '"approval_note":"<one sentence flagging anything a human reviewer should double-'
    'check before approving -- a dollar amount that seems off, an ambiguous company '
    'name, a claim that goes slightly beyond the source. If nothing to flag, say so '
    'plainly.>"}'
)

_USER = """Post to score:
main_post: {main_post}
thread_tweets: {thread_tweets}
hashtags_used: {hashtags_used}

Return the JSON object as specified."""


class QualityAgent(BaseAgent):
    name = "quality"

    async def evaluate_v2(self, *, copy: CopyOutput, job_id: str) -> QualityOutput:
        user = _USER.format(
            main_post=copy.main_post,
            thread_tweets=json.dumps(copy.thread_tweets) if copy.thread_tweets else "null",
            hashtags_used=json.dumps(copy.hashtags_used))
        raw = await self._complete_json(system=_SYSTEM, user=user, job_id=job_id, max_tokens=1200)

        # verdict/total/blocking_dimension are computed by the model's own
        # validator from scores -- strip any self-reported values so the LLM's
        # own guess (if it produced one anyway) can't leak through as input
        if isinstance(raw, dict):
            raw.pop("verdict", None)
            raw.pop("total", None)
            raw.pop("blocking_dimension", None)

        return await self._validate(QualityOutput, raw, job_id=job_id, step="quality.evaluate_v2")
