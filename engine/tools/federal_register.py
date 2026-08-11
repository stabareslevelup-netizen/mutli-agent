"""
engine/tools/federal_register.py — Federal Register rule search (Phase 1).

No auth required. Rules/proposed rules mentioning a search term, filtered to
publication date >= N days back.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Awaitable, Callable, Optional

FetchFn = Callable[[str, dict], Awaitable[dict]]

_ENDPOINT = "https://www.federalregister.gov/api/v1/articles"


async def _httpx_get(url: str, params: dict) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def search_federal_register(*, term: str, days_back: int = 7,
                                   agencies: Optional[list[str]] = None,
                                   fetch_fn: Optional[FetchFn] = None) -> list[dict]:
    """Returns raw article dicts: title, agency_names, abstract, source_url,
    publication_date. Empty list on no matches."""
    fetch = fetch_fn or _httpx_get
    since = date.today() - timedelta(days=days_back)
    params = {
        "conditions[term]": term,
        "conditions[publication_date][gte]": since.isoformat(),
    }
    if agencies:
        params["conditions[agencies][]"] = agencies
    body = await fetch(_ENDPOINT, params)
    results = []
    for item in body.get("results", []) or []:
        results.append({
            "title": item.get("title", ""),
            "agency_names": item.get("agency_names", []),
            "abstract": item.get("abstract", ""),
            "source_url": item.get("html_url", ""),
            "publication_date": item.get("publication_date", ""),
        })
    return results
