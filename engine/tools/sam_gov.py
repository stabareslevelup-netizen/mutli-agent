"""
engine/tools/sam_gov.py — SAM.gov contract-award search (Phase 1, X-agent migration).

Highest-priority Research source: DoD contract awards, posted in the last N
days, matching AI/autonomy keywords. Free API key from sam.gov/profile.

Stdlib + httpx only (matches x_client.py's DI style). fetch_fn is injectable
so tests never hit the network; the real implementation is the default.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Awaitable, Callable, Optional

FetchFn = Callable[[str, dict], Awaitable[dict]]  # (url, params) -> parsed JSON body

_ENDPOINT = "https://api.sam.gov/opportunities/v2/search"


async def _httpx_get(url: str, params: dict) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


class SamGovClient:
    def __init__(self, api_key: str, *, fetch_fn: Optional[FetchFn] = None):
        self._api_key = api_key
        self._fetch = fetch_fn or _httpx_get

    async def search(self, *, keywords: list[str], days_back: int = 1,
                     dept_name: Optional[str] = None) -> list[dict]:
        """Returns raw opportunity dicts: title, awardee_name, award_amount,
        description, source_url, posted_date. Empty list on no matches."""
        today = date.today()
        since = today - timedelta(days=days_back)
        params = {
            "api_key": self._api_key,
            "postedFrom": since.strftime("%m/%d/%Y"),
            "postedTo": today.strftime("%m/%d/%Y"),
            "keywords": " OR ".join(keywords),
        }
        if dept_name:
            params["deptname"] = dept_name
        body = await self._fetch(_ENDPOINT, params)
        results = []
        for item in body.get("opportunitiesData", []) or []:
            award = item.get("award") or {}
            awardee = award.get("awardee") or {}
            results.append({
                "title": item.get("title", ""),
                "awardee_name": awardee.get("name", ""),
                "award_amount": award.get("amount"),
                "description": item.get("description", ""),
                "source_url": item.get("uiLink", ""),
                "posted_date": item.get("postedDate", ""),
            })
        return results


def build_sam_gov_client_from_env() -> Optional[SamGovClient]:
    """Graceful degrade: None when SAM_GOV_API_KEY isn't set."""
    key = os.getenv("SAM_GOV_API_KEY")
    if not key:
        return None
    return SamGovClient(key)
