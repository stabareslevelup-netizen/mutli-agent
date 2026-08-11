"""
engine/tools/arxiv_search.py — arXiv paper search (Phase 1).

No auth required. Returns an Atom XML feed (not JSON) — parsed with stdlib
xml.etree.ElementTree, no new dependency. Filters to papers submitted within
a lookback window (the API itself has no date-range param, so filtering
happens client-side against each entry's `published` date).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

FetchFn = Callable[[str, dict], Awaitable[str]]  # (url, params) -> raw XML text

_ENDPOINT = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


async def _httpx_get_text(url: str, params: dict) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.text


def _parse_feed(xml_text: str, *, since: datetime) -> list[dict]:
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall(f"{_ATOM}entry"):
        published_raw = (entry.findtext(f"{_ATOM}published") or "").strip()
        try:
            published = datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if published < since:
            continue
        authors = [a.findtext(f"{_ATOM}name") or "" for a in entry.findall(f"{_ATOM}author")]
        results.append({
            "title": (entry.findtext(f"{_ATOM}title") or "").strip(),
            "authors": authors,
            "summary": (entry.findtext(f"{_ATOM}summary") or "").strip(),
            "source_url": (entry.findtext(f"{_ATOM}id") or "").strip(),
            "published": published_raw,
        })
    return results


async def search_arxiv(*, categories: list[str], days_back: int = 2,
                       max_results: int = 20,
                       fetch_fn: Optional[FetchFn] = None) -> list[dict]:
    """Returns raw paper dicts: title, authors, summary, source_url,
    published. Empty list on no matches."""
    fetch = fetch_fn or _httpx_get_text
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    xml_text = await fetch(_ENDPOINT, params)
    return _parse_feed(xml_text, since=since)
