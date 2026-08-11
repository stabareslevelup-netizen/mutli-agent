"""
Federal Register search tests — FREE (injected fake transport, no network).
"""
from __future__ import annotations

import asyncio

from engine.tools.federal_register import search_federal_register

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


_SAMPLE_BODY = {
    "results": [
        {
            "title": "Artificial Intelligence in Government Procurement",
            "agency_names": ["Department of Defense"],
            "abstract": "A proposed rule on AI procurement standards.",
            "html_url": "https://federalregister.gov/d/2026-12345",
            "publication_date": "2026-08-05",
        }
    ]
}


def test_search_maps_fields_and_sends_expected_params():
    captured = {}

    async def fake_fetch(url, params):
        captured["url"] = url
        captured["params"] = params
        return _SAMPLE_BODY

    out = asyncio.run(search_federal_register(term="artificial intelligence", days_back=7,
                                               agencies=["defense-department"], fetch_fn=fake_fetch))
    check("hits the articles endpoint", captured["url"] == "https://www.federalregister.gov/api/v1/articles")
    check("term passed through", captured["params"]["conditions[term]"] == "artificial intelligence")
    check("agencies passed through when provided",
          captured["params"]["conditions[agencies][]"] == ["defense-department"])
    check("one result returned", len(out) == 1, str(out))
    r = out[0]
    check("title mapped", r["title"] == "Artificial Intelligence in Government Procurement")
    check("agency_names mapped", r["agency_names"] == ["Department of Defense"])
    check("abstract mapped", "AI procurement" in r["abstract"])
    check("source_url mapped from html_url", r["source_url"] == "https://federalregister.gov/d/2026-12345")
    check("publication_date mapped", r["publication_date"] == "2026-08-05")


def test_no_agencies_omits_param():
    captured = {}

    async def fake_fetch(url, params):
        captured["params"] = params
        return {"results": []}

    asyncio.run(search_federal_register(term="ai", fetch_fn=fake_fetch))
    check("agencies filter omitted when not provided", "conditions[agencies][]" not in captured["params"])


def test_empty_results_returns_empty_list():
    async def fake_fetch(url, params):
        return {"results": []}

    out = asyncio.run(search_federal_register(term="ai", fetch_fn=fake_fetch))
    check("no matches -> empty list, not an error", out == [])


def test_no_auth_required():
    # sanity: the function signature has no api_key param at all
    import inspect
    sig = inspect.signature(search_federal_register)
    check("no api_key parameter (Federal Register requires no auth)", "api_key" not in sig.parameters)


def main() -> int:
    test_search_maps_fields_and_sends_expected_params()
    test_no_agencies_omits_param()
    test_empty_results_returns_empty_list()
    test_no_auth_required()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_federal_register():
    assert main() == 0, "federal_register suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
