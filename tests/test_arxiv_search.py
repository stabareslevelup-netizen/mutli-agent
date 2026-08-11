"""
arXiv search tests — FREE (injected fake transport, no network).

Covers Atom XML parsing (arXiv's API returns XML, not JSON) and the
client-side days_back filter (the API itself has no date-range param).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from engine.tools.arxiv_search import search_arxiv

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


def _entry(published_dt: datetime, title: str, arxiv_id: str) -> str:
    ts = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""
    <entry>
      <id>http://arxiv.org/abs/{arxiv_id}</id>
      <published>{ts}</published>
      <title>{title}</title>
      <summary>A summary of {title}.</summary>
      <author><name>A. Researcher</name></author>
      <author><name>B. Researcher</name></author>
    </entry>"""


def _feed(entries: list[str]) -> str:
    body = "".join(entries)
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'


def test_parses_recent_entry_within_window():
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=6)
    xml = _feed([_entry(recent, "Humanoid Locomotion via RL", "2608.00001")])
    captured = {}

    async def fake_fetch(url, params):
        captured["url"] = url
        captured["params"] = params
        return xml

    out = asyncio.run(search_arxiv(categories=["cs.RO", "cs.AI"], days_back=2, fetch_fn=fake_fetch))
    check("hits the export.arxiv.org query endpoint", captured["url"] == "https://export.arxiv.org/api/query")
    check("categories joined into cat: OR query",
          captured["params"]["search_query"] == "cat:cs.RO OR cat:cs.AI")
    check("one result parsed from XML", len(out) == 1, str(out))
    r = out[0]
    check("title parsed", r["title"] == "Humanoid Locomotion via RL")
    check("summary parsed", "A summary of" in r["summary"])
    check("both authors parsed", r["authors"] == ["A. Researcher", "B. Researcher"])
    check("source_url parsed from id", r["source_url"] == "http://arxiv.org/abs/2608.00001")


def test_old_entry_outside_window_is_filtered_out():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    xml = _feed([_entry(old, "An Old Paper", "2601.00001")])

    async def fake_fetch(url, params):
        return xml

    out = asyncio.run(search_arxiv(categories=["cs.RO"], days_back=2, fetch_fn=fake_fetch))
    check("entry older than days_back is excluded", out == [])


def test_mixed_recent_and_old_only_returns_recent():
    now = datetime.now(timezone.utc)
    xml = _feed([
        _entry(now - timedelta(hours=1), "Fresh Paper", "2608.00002"),
        _entry(now - timedelta(days=10), "Stale Paper", "2607.00003"),
    ])

    async def fake_fetch(url, params):
        return xml

    out = asyncio.run(search_arxiv(categories=["cs.AI"], days_back=2, fetch_fn=fake_fetch))
    check("only the fresh entry survives the filter", len(out) == 1, str(out))
    check("the surviving entry is the fresh one", out[0]["title"] == "Fresh Paper" if out else False)


def test_empty_feed_returns_empty_list():
    async def fake_fetch(url, params):
        return _feed([])

    out = asyncio.run(search_arxiv(categories=["cs.RO"], fetch_fn=fake_fetch))
    check("empty feed -> empty list, not an error", out == [])


def main() -> int:
    test_parses_recent_entry_within_window()
    test_old_entry_outside_window_is_filtered_out()
    test_mixed_recent_and_old_only_returns_recent()
    test_empty_feed_returns_empty_list()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def pytest_arxiv_search():
    assert main() == 0, "arxiv_search suite reported failures — see printed PASS/FAIL above"


if __name__ == "__main__":
    raise SystemExit(main())
