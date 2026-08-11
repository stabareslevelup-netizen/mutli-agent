"""
Congress.gov search tests — FREE (injected fake transport, no network).
"""
from __future__ import annotations

import asyncio
import os

from engine.tools.congress_gov import CongressGovClient, build_congress_gov_client_from_env

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


_SAMPLE_BODY = {
    "bills": [
        {
            "title": "Autonomous Weapons Accountability Act",
            "introducedDate": "2026-08-01",
            "sponsors": [{"fullName": "Sen. Jane Doe"}],
            "latestAction": {"text": "Referred to the Committee on Armed Services."},
            "url": "https://api.congress.gov/v3/bill/119/s1234",
        }
    ]
}


def test_search_maps_fields_and_sends_expected_params():
    captured = {}

    async def fake_fetch(url, params):
        captured["url"] = url
        captured["params"] = params
        return _SAMPLE_BODY

    client = CongressGovClient("test-key", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(query="autonomous weapons", days_back=7))
    check("hits the v3/bill endpoint", captured["url"] == "https://api.congress.gov/v3/bill")
    check("api_key passed through", captured["params"]["api_key"] == "test-key")
    check("q passed through", captured["params"]["q"] == "autonomous weapons")
    check("fromDateTime present", "fromDateTime" in captured["params"])
    check("one result returned", len(out) == 1, str(out))
    r = out[0]
    check("title mapped", r["title"] == "Autonomous Weapons Accountability Act")
    check("introduced_date mapped", r["introduced_date"] == "2026-08-01")
    check("sponsors mapped", r["sponsors"] == [{"fullName": "Sen. Jane Doe"}])
    check("latest_action mapped from nested latestAction.text",
          r["latest_action"] == "Referred to the Committee on Armed Services.")
    check("source_url mapped", r["source_url"] == "https://api.congress.gov/v3/bill/119/s1234")


def test_empty_results_returns_empty_list():
    async def fake_fetch(url, params):
        return {"bills": []}

    client = CongressGovClient("k", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(query="ai"))
    check("no matches -> empty list, not an error", out == [])


def test_missing_latest_action_does_not_crash():
    async def fake_fetch(url, params):
        return {"bills": [{"title": "no action info"}]}

    client = CongressGovClient("k", fetch_fn=fake_fetch)
    out = asyncio.run(client.search(query="ai"))
    check("missing nested latestAction doesn't crash", len(out) == 1)
    check("latest_action defaults to empty string", out[0]["latest_action"] == "")


def test_build_from_env_graceful_degrade():
    saved = os.environ.pop("CONGRESS_GOV_API_KEY", None)
    try:
        check("no CONGRESS_GOV_API_KEY -> None (not an error)", build_congress_gov_client_from_env() is None)
        os.environ["CONGRESS_GOV_API_KEY"] = "real-key"
        c = build_congress_gov_client_from_env()
        check("CONGRESS_GOV_API_KEY set -> a real client", isinstance(c, CongressGovClient))
    finally:
        if saved is not None:
            os.environ["CONGRESS_GOV_API_KEY"] = saved
        else:
            os.environ.pop("CONGRESS_GOV_API_KEY", None)


def main() -> int:
    test_search_maps_fields_and_sends_expected_params()
    test_empty_results_returns_empty_list()
    test_missing_latest_action_does_not_crash()
    test_build_from_env_graceful_degrade()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_congress_gov():
    assert main() == 0, "congress_gov suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
