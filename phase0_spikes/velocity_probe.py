"""
PHASE 0 SPIKE — velocity_probe  [THROWAWAY, not production code]

Question: can we reliably return a "is this topic accelerating in cultural
attention?" signal good enough for the Timing agent to act on?

What this spike proves:
  The SCORING LOGIC is real and runnable offline. It consumes a list of
  search results (title + snippet + optional date) and emits an ORDINAL
  velocity verdict {SURGING, STEADY, DECLINING, DEAD} + confidence + why.

What is stubbed (the network part):
  In production, `fetch` would call the Timing agent's web_search tool (or a
  numeric time-series API like GDELT/Wikipedia-pageviews/Trends). In THIS
  environment every external host is blocked by the egress allowlist, so the
  fixtures below are the *actual* WebSearch output captured on 2026-06-15.

Honest finding (see PHASE0_FINDINGS.md): search-derived velocity is a reliable
ORDINAL signal, NOT a precise numeric rate-of-change. Volume alone is
misleading (Roomba has fresh news but is not accelerating). The discriminator
that works is recency-of-dates + momentum-vs-decline language.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import date

# --- momentum / decline lexicons (the part that actually carries signal) ----
MOMENTUM = [
    "ramp", "ramps", "surge", "surges", "soar", "record", "milestone",
    "first", "breakthrough", "accelerat", "doubles", "24x", "unprecedented",
    "begins operations", "rolls out", "launch", "unveil", "scaling", "scale-up",
]
DECLINE = [
    "retire", "retired", "discontinu", "shut down", "shutting", "halt",
    "halted", "legacy", "what happened to", "defunct", "abandoned", "sunset",
]
THIS_YEAR = 2026


@dataclass
class VelocityResult:
    topic: str
    verdict: str            # SURGING | STEADY | DECLINING | DEAD
    confidence: float       # 0..1
    fresh_ratio: float      # share of dated items from THIS_YEAR
    momentum_hits: int
    decline_hits: int
    rationale: str
    needs_human: bool = False


def _years_in(text: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]


def score_velocity(topic: str, results: list[dict]) -> VelocityResult:
    blob = " ".join(f"{r.get('title','')} {r.get('snippet','')}" for r in results).lower()
    years = []
    for r in results:
        years += _years_in(f"{r.get('title','')} {r.get('snippet','')} {r.get('date','')}")
    fresh_ratio = (sum(1 for y in years if y >= THIS_YEAR) / len(years)) if years else 0.0
    newest = max(years) if years else 0

    mom = sum(blob.count(k) for k in MOMENTUM)
    dec = sum(blob.count(k) for k in DECLINE)

    # decision: recency first (dead = stale), then momentum vs decline language
    if newest and newest <= THIS_YEAR - 3:
        verdict = "DEAD"
    elif dec > mom and dec >= 2:
        verdict = "DECLINING"
    elif mom >= 3 and fresh_ratio >= 0.6:
        verdict = "SURGING"
    else:
        verdict = "STEADY"

    # confidence = signal separation, dampened when evidence is thin
    sep = abs(mom - dec) / max(mom + dec, 1)
    confidence = round(min(1.0, 0.35 + 0.4 * sep + 0.25 * fresh_ratio), 2)
    needs_human = confidence < 0.55 or len(results) < 4

    rationale = (
        f"newest_year={newest} fresh_ratio={fresh_ratio:.2f} "
        f"momentum={mom} decline={dec} -> {verdict}"
    )
    return VelocityResult(topic, verdict, confidence, round(fresh_ratio, 2),
                          mom, dec, rationale, needs_human)


# --- FIXTURES: real WebSearch output captured 2026-06-15 --------------------
FIXTURES = {
    "Figure 03 production (expect SURGING)": [
        {"title": "Figure ramps humanoid robot production from one per day to one per hour",
         "snippet": "24x throughput improvement in under 120 days; delivered over 350 robots", "date": "2026-05-27"},
        {"title": "Figure’s Humanoid Robot Factory Just Hit a Major Production Milestone", "snippet": "BotQ first line up to 12,000 robots/yr", "date": "2026"},
        {"title": "Figure AI Ramps Up Production to One Humanoid Robot Per Hour", "snippet": "unprecedented speed scale-up", "date": "2026-05-01"},
    ],
    "Hangzhou robot police (expect SURGING)": [
        {"title": "China’s first robot traffic police squad begins operations", "snippet": "15 robots deployed May 1 2026, issued 11,897 warnings in three days", "date": "2026-05"},
        {"title": "Hangzhou's AI Robot Police Officer Directs City Traffic", "snippet": "rolls out first organized squad", "date": "2026-05-04"},
    ],
    "Roomba (expect STEADY)": [
        {"title": "Roomba robot vacuums are quietly changing in 2026", "snippet": "software-driven improvements, no flashy redesign", "date": "2026"},
        {"title": "iRobot just unveiled 8 new Roombas at once", "snippet": "2026 lineup revealed, slimmer designs", "date": "2026-03"},
        {"title": "iRobot acquired by Picea", "snippet": "ownership change Feb 2026", "date": "2026-02-12"},
    ],
    "Honda ASIMO (expect DEAD)": [
        {"title": "Even as It Retires, ASIMO Still Manages to Impress", "snippet": "Honda retired Asimo in 2022 after four decades", "date": "2022"},
        {"title": "Asimo does bottles, lovey-dovey hand gestures via CNET", "snippet": "legacy demo", "date": "2011"},
        {"title": "Honda Asimo OS: A Robot Legend Is Reborn To Power New EVs", "snippet": "name homage at CES 2025; robot discontinued", "date": "2025"},
    ],
}

if __name__ == "__main__":
    for label, rows in FIXTURES.items():
        topic = label.split(" (")[0]
        r = score_velocity(topic, rows)
        flag = "  <needs_human>" if r.needs_human else ""
        print(f"{r.verdict:10} conf={r.confidence:<4} {topic}{flag}")
        print(f"            {r.rationale}")
