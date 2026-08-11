"""
engine/tools/darpa_solicitations.py — DARPA solicitations (Phase 1, lowest
priority per spec: "no official API — assess scrapeability, fall back to
web_search if unreliable").

Scoping note: darpa.mil has no documented API and no stable, published HTML
structure to scrape against — building a scraper here would mean guessing at
markup with no way to verify it in this environment (same network
restriction that blocks live testing of the other 4 sources also blocks
fetching a real page sample to inspect). Rather than ship a scraper that
looks complete but silently returns nothing forever, this implements only
the fallback the spec explicitly approves: a ready-to-run web_search query,
executed via the Research agent's existing WEB_SEARCH_TOOL (Phase 2 wiring).
Revisit with a real scraper once someone can supply a live page sample.
"""
from __future__ import annotations

_QUERY = "site:darpa.mil news-events/solicitations OR broad-agency-announcement"


def search_darpa_solicitations() -> dict:
    """Returns a fallback web_search directive (no live client-side fetch —
    see module docstring). {"query": ..., "note": ...} — the Research agent
    runs this through web_search rather than a dedicated API call."""
    return {
        "query": _QUERY,
        "note": ("no DARPA API; scraping unverified in this environment — "
                 "falls back to web_search per spec Phase 1 §1d"),
    }
