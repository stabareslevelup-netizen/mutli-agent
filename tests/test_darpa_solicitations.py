"""
DARPA solicitations tests — FREE, no network (there's nothing to fetch:
this module returns a web_search fallback directive, see its docstring for
why a live scraper isn't implemented yet).
"""
from __future__ import annotations

from engine.tools.darpa_solicitations import search_darpa_solicitations

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


def test_returns_a_web_search_fallback_directive():
    out = search_darpa_solicitations()
    check("returns a dict", isinstance(out, dict))
    check("has a query field", "query" in out and isinstance(out["query"], str) and out["query"])
    check("query is scoped to darpa.mil", "site:darpa.mil" in out["query"])
    check("has a note explaining the fallback", "note" in out and out["note"])


def test_no_params_required():
    import inspect
    sig = inspect.signature(search_darpa_solicitations)
    check("takes no parameters, matching the spec's 'just latest'", len(sig.parameters) == 0)


def main() -> int:
    test_returns_a_web_search_fallback_directive()
    test_no_params_required()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_darpa_solicitations():
    assert main() == 0, "darpa_solicitations suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
