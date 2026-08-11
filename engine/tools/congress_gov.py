"""
engine/tools/congress_gov.py — Congress.gov bill search (Phase 1).

Free API key from api.congress.gov. Bills introduced/amended mentioning a
search term, sorted newest-first, filtered to a lookback window.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Awaitable, Callable, Optional

FetchFn = Callable[[str, dict], Awaitable[dict]]

_ENDPOINT = "https://api.congress.gov/v3/bill"


async def _httpx_get(url: str, params: dict) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


class CongressGovClient:
    def __init__(self, api_key: str, *, fetch_fn: Optional[FetchFn] = None):
        self._api_key = api_key
        self._fetch = fetch_fn or _httpx_get

    async def search(self, *, query: str, days_back: int = 7) -> list[dict]:
        """Returns raw bill dicts: title, introduced_date, sponsors,
        latest_action, source_url. Empty list on no matches."""
        since = date.today() - timedelta(days=days_back)
        params = {
            "api_key": self._api_key,
            "q": query,
            "fromDateTime": f"{since.isoformat()}T00:00:00Z",
            "sort": "updateDate+desc",
            "format": "json",
        }
        body = await self._fetch(_ENDPOINT, params)
        results = []
        for item in body.get("bills", []) or []:
            latest_action = item.get("latestAction") or {}
            results.append({
                "title": item.get("title", ""),
                "introduced_date": item.get("introducedDate", ""),
                "sponsors": item.get("sponsors", []),
                "latest_action": latest_action.get("text", ""),
                "source_url": item.get("url", ""),
            })
        return results


def build_congress_gov_client_from_env() -> Optional[CongressGovClient]:
    """Graceful degrade: None when CONGRESS_GOV_API_KEY isn't set."""
    key = os.getenv("CONGRESS_GOV_API_KEY")
    if not key:
        return None
    return CongressGovClient(key)
