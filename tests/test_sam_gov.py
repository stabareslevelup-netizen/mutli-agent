"""
SAM.gov search tests — FREE (injected fake transport, no network).
"""
from __future__ import annotations

import asyncio
import os

from engine.tools.sam_gov import SamGovClient, build_sam_gov_client_from_env

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


_SAMPLE_BODY = {
    "opportunitiesData": [
        {
            "title": "AI-Enabled Autonomous Perimeter Defense",
            "description": "Contract for AI perception software.",
            "uiLink": "https://sam.gov/opp/abc123",
            "postedDate": "2026-08-10",
            "award": {"amount": 66000000, "awardee": {"name": "Epirus"}},
        }
    ]
}


def test_search_maps_fields_and_sends_expected_params():
    captured = {}

    async def fake_fetch(url, params):
        captured["url"] = url
        captured["params"] = params
        return _SAMPLE_BODY

    client = SamGovClient("test-key", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(keywords=["artificial intelligence", "robotics"],
                                     days_back=1, dept_name="DoD"))
    check("hits the opportunities/v2/search endpoint",
          captured["url"] == "https://api.sam.gov/opportunities/v2/search")
    check("api_key passed through", captured["params"]["api_key"] == "test-key")
    check("keywords joined with OR", captured["params"]["keywords"] == "artificial intelligence OR robotics")
    check("deptname passed when provided", captured["params"]["deptname"] == "DoD")
    check("postedFrom/postedTo present", "postedFrom" in captured["params"] and "postedTo" in captured["params"])
    check("one result returned", len(out) == 1, str(out))
    r = out[0]
    check("title mapped", r["title"] == "AI-Enabled Autonomous Perimeter Defense")
    check("awardee_name mapped from nested award.awardee.name", r["awardee_name"] == "Epirus")
    check("award_amount mapped from nested award.amount", r["award_amount"] == 66000000)
    check("source_url mapped from uiLink", r["source_url"] == "https://sam.gov/opp/abc123")
    check("posted_date mapped", r["posted_date"] == "2026-08-10")


def test_no_dept_name_omits_param():
    captured = {}

    async def fake_fetch(url, params):
        captured["params"] = params
        return {"opportunitiesData": []}

    client = SamGovClient("k", fetch_fn=fake_fetch)
    asyncio.run(client.search(keywords=["ai"]))
    check("deptname omitted when not provided", "deptname" not in captured["params"])


def test_empty_results_returns_empty_list():
    async def fake_fetch(url, params):
        return {"opportunitiesData": []}

    client = SamGovClient("k", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(keywords=["ai"]))
    check("no matches -> empty list, not an error", out == [])


def test_missing_award_data_does_not_crash():
    async def fake_fetch(url, params):
        return {"opportunitiesData": [{"title": "no award info"}]}

    client = SamGovClient("k", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(keywords=["ai"]))
    check("missing nested award/awardee doesn't crash", len(out) == 1)
    check("awardee_name defaults to empty string", out[0]["awardee_name"] == "")
    check("award_amount defaults to None", out[0]["award_amount"] is None)


def test_build_from_env_graceful_degrade():
    saved = os.environ.pop("SAM_GOV_API_KEY", None)
    try:
        check("no SAM_GOV_API_KEY -> None (not an error)", build_sam_gov_client_from_env() is None)
        os.environ["SAM_GOV_API_KEY"] = "real-key"
        c = build_sam_gov_client_from_env()
        check("SAM_GOV_API_KEY set -> a real client", isinstance(c, SamGovClient))
    finally:
        if saved is not None:
            os.environ["SAM_GOV_API_KEY"] = saved
        else:
            os.environ.pop("SAM_GOV_API_KEY", None)


def main() -> int:
    test_search_maps_fields_and_sends_expected_params()
    test_no_dept_name_omits_param()
    test_empty_results_returns_empty_list()
    test_missing_award_data_does_not_crash()
    test_build_from_env_graceful_degrade()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_sam_gov():
    assert main() == 0, "sam_gov suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
