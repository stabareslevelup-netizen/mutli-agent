"""
engine/tools/velocity_probe.py — cultural-velocity signal for the Timing agent.

# ORPHANED - Phase 2 migration. Candidate for cleanup. Do not delete until
# after Phase 5 is complete. The new TimingDecision schema has no
# velocity-verdict field; velocity/urgency judgment now folds into
# timing.py's single batch LLM call instead of this structured probe.

Design resolves the Phase 0 infra asks:

  * PRIMARY data layer = the Anthropic server-side `web_search` tool, injected
    as `search_fn`. Works today, with NO external-host egress. Produces an
    ORDINAL verdict (RULE 1: never fake-precise numbers).

  * NUMERIC upgrade = time-series sources (GDELT, Wikipedia pageviews). They
    are dependency-injected and self-describe `available()`. When their hosts
    are allowlisted (see INFRA_REQUIREMENTS.md) they take priority and emit a
    real `numeric_rate`. When blocked/unreachable they report unavailable and
    the probe DEGRADES GRACEFULLY to the ordinal path — no rewrite, no redeploy.

Source priority: first available numeric source -> ordinal -> unknown.

Brand-agnostic: no brand/character/voice/pillar strings here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional, Protocol, Sequence

from engine.core.models import VelocitySignal, VelocitySource, VelocityVerdict

# --- ordinal lexicons (the signal that actually separates topics) -----------
_MOMENTUM = [
    "ramp", "surge", "soar", "record", "milestone", "first", "breakthrough",
    "accelerat", "doubles", "unprecedented", "begins operations", "rolls out",
    "launch", "unveil", "scaling", "scale-up", "24x",
]
_DECLINE = [
    "retire", "discontinu", "shut down", "shutting", "halt", "legacy",
    "what happened to", "defunct", "abandoned", "sunset",
]


# A search result the ordinal path consumes (shape returned by web_search).
@dataclass
class SearchResult:
    title: str = ""
    snippet: str = ""
    published: Optional[str] = None     # ISO-ish date string if known


SearchFn = Callable[[str], Sequence[dict]]   # query -> list of result dicts


class SourceUnavailable(Exception):
    pass


# ===========================================================================
# Data-source protocol (dependency injection seam)
# ===========================================================================
@dataclass
class VelocityMeasurement:
    source: VelocitySource
    verdict: VelocityVerdict
    confidence: float
    numeric_rate: Optional[float]
    rationale: str
    needs_human: bool = False


class VelocityDataSource(Protocol):
    name: str
    def available(self) -> bool: ...
    def measure(self, topic: str, window_days: int) -> VelocityMeasurement: ...


# ===========================================================================
# Ordinal source (PRIMARY, available today via web_search)
# ===========================================================================
class WebSearchOrdinalSource:
    name = "web_search_ordinal"

    def __init__(self, search_fn: SearchFn, this_year: Optional[int] = None):
        self._search = search_fn
        self._this_year = this_year or datetime.now(timezone.utc).year

    def available(self) -> bool:
        return self._search is not None

    @staticmethod
    def _years(text: str) -> list[int]:
        return [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]

    def measure(self, topic: str, window_days: int) -> VelocityMeasurement:
        results = list(self._search(topic) or [])
        if not results:
            return VelocityMeasurement(
                VelocitySource.web_search_ordinal, VelocityVerdict.unknown,
                0.2, None, "no search results", needs_human=True,
            )
        blob = " ".join(f"{r.get('title','')} {r.get('snippet','')}" for r in results).lower()
        years: list[int] = []
        for r in results:
            years += self._years(f"{r.get('title','')} {r.get('snippet','')} {r.get('published','')}")
        fresh_ratio = (sum(1 for y in years if y >= self._this_year) / len(years)) if years else 0.0
        newest = max(years) if years else 0
        mom = sum(blob.count(k) for k in _MOMENTUM)
        dec = sum(blob.count(k) for k in _DECLINE)

        if newest and newest <= self._this_year - 3:
            verdict = VelocityVerdict.dead
        elif dec > mom and dec >= 2:
            verdict = VelocityVerdict.declining
        elif mom >= 3 and fresh_ratio >= 0.6:
            verdict = VelocityVerdict.surging
        else:
            verdict = VelocityVerdict.steady

        sep = abs(mom - dec) / max(mom + dec, 1)
        confidence = round(min(1.0, 0.35 + 0.4 * sep + 0.25 * fresh_ratio), 2)
        needs_human = confidence < 0.55 or len(results) < 4
        rationale = (f"ordinal: newest={newest} fresh={fresh_ratio:.2f} "
                     f"momentum={mom} decline={dec}")
        return VelocityMeasurement(
            VelocitySource.web_search_ordinal, verdict, confidence, None,
            rationale, needs_human,
        )


# ===========================================================================
# Numeric sources (UPGRADE, active once hosts are allowlisted)
# ===========================================================================
class _NumericTimeSeriesSource:
    """Shared logic: pull a daily series, convert to a verdict + numeric rate."""
    name = "numeric"
    source_enum = VelocitySource.gdelt_timeseries

    def _series(self, topic: str, window_days: int) -> list[float]:
        raise NotImplementedError

    def available(self) -> bool:
        try:
            return self._reachable()
        except Exception:
            return False

    def _reachable(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def _rate_and_verdict(series: list[float]) -> tuple[float, VelocityVerdict, float]:
        if len(series) < 4:
            return 0.0, VelocityVerdict.unknown, 0.2
        half = len(series) // 2
        prior = sum(series[:half]) / max(half, 1)
        recent = sum(series[half:]) / max(len(series) - half, 1)
        rate = ((recent - prior) / prior * 100.0) if prior > 0 else 0.0
        if rate >= 50:
            verdict = VelocityVerdict.surging
        elif rate <= -50:
            verdict = VelocityVerdict.declining
        elif recent < 1e-6:
            verdict = VelocityVerdict.dead
        else:
            verdict = VelocityVerdict.steady
        confidence = round(min(1.0, 0.6 + min(abs(rate), 100) / 250), 2)
        return round(rate, 1), verdict, confidence

    def measure(self, topic: str, window_days: int) -> VelocityMeasurement:
        series = self._series(topic, window_days)
        rate, verdict, conf = self._rate_and_verdict(series)
        return VelocityMeasurement(
            self.source_enum, verdict, conf, rate,
            f"numeric: {len(series)} pts, {rate:+.1f}% recent-vs-prior",
        )


class GdeltTimeSeriesSource(_NumericTimeSeriesSource):
    name = "gdelt"
    source_enum = VelocitySource.gdelt_timeseries
    HOST = "https://api.gdeltproject.org"

    def __init__(self, http_get: Optional[Callable[[str], dict]] = None):
        self._http_get = http_get   # injected; None until host allowlisted

    def _reachable(self) -> bool:
        return self._http_get is not None

    def _series(self, topic: str, window_days: int) -> list[float]:
        if self._http_get is None:
            raise SourceUnavailable("gdelt host not wired/allowlisted")
        url = (f"{self.HOST}/api/v2/doc/doc?query={topic}"
               f"&mode=timelinevol&format=json&timespan={max(window_days,7)}d")
        data = self._http_get(url)
        points = (data.get("timeline", [{}])[0].get("data", [])) if data else []
        return [float(p.get("value", 0.0)) for p in points]


class WikipediaPageviewsSource(_NumericTimeSeriesSource):
    name = "wikipedia"
    source_enum = VelocitySource.wikipedia_pageviews
    HOST = "https://wikimedia.org"

    def __init__(self, http_get: Optional[Callable[[str], dict]] = None,
                 resolve_article: Optional[Callable[[str], str]] = None):
        self._http_get = http_get
        self._resolve = resolve_article or (lambda t: t.strip().replace(" ", "_"))

    def _reachable(self) -> bool:
        return self._http_get is not None

    def _series(self, topic: str, window_days: int) -> list[float]:
        if self._http_get is None:
            raise SourceUnavailable("wikimedia host not wired/allowlisted")
        article = self._resolve(topic)
        end = date.today().strftime("%Y%m%d")
        start = end  # caller-side range computation kept minimal here
        url = (f"{self.HOST}/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}")
        data = self._http_get(url)
        items = data.get("items", []) if data else []
        return [float(i.get("views", 0.0)) for i in items]


# ===========================================================================
# The probe — orchestrates sources, returns a validated VelocitySignal
# ===========================================================================
@dataclass
class VelocityProbe:
    sources: list[VelocityDataSource] = field(default_factory=list)

    def probe(self, topic: str, window_days: int = 30) -> VelocitySignal:
        # numeric sources first (priority), then ordinal, then unknown
        for src in self.sources:
            try:
                if not src.available():
                    continue
                m = src.measure(topic, window_days)
                return VelocitySignal(
                    topic=topic, verdict=m.verdict, confidence=m.confidence,
                    source=m.source, numeric_rate=m.numeric_rate,
                    window_days=window_days, rationale=m.rationale,
                    needs_human=m.needs_human,
                )
            except SourceUnavailable:
                continue
            except Exception as exc:  # a source failing must not crash the probe
                continue
        # graceful degrade: nothing available
        return VelocitySignal(
            topic=topic, verdict=VelocityVerdict.unknown, confidence=0.1,
            source=VelocitySource.unavailable, numeric_rate=None,
            window_days=window_days, rationale="no velocity source available",
            needs_human=True,
        )


def build_default_probe(search_fn: SearchFn,
                        gdelt_http_get: Optional[Callable[[str], dict]] = None,
                        wiki_http_get: Optional[Callable[[str], dict]] = None) -> VelocityProbe:
    """Numeric sources lead; ordinal web_search is the always-on fallback.
    Pass gdelt_http_get / wiki_http_get only once their hosts are allowlisted."""
    return VelocityProbe(sources=[
        GdeltTimeSeriesSource(http_get=gdelt_http_get),
        WikipediaPageviewsSource(http_get=wiki_http_get),
        WebSearchOrdinalSource(search_fn=search_fn),
    ])


if __name__ == "__main__":
    # Offline self-test: numeric sources unavailable (hosts blocked today) ->
    # probe degrades to the ordinal web_search path automatically.
    _FIXTURE = {
        "humanoid robot factory production": [
            {"title": "Maker ramps production from one per day to one per hour",
             "snippet": "24x throughput, unprecedented scale-up", "published": "2026-05-27"},
            {"title": "Factory hits a major production milestone", "snippet": "scaling fast", "published": "2026"},
            {"title": "Production ramps to one unit per hour", "snippet": "launch milestone", "published": "2026-05-01"},
        ],
        "discontinued legacy robot": [
            {"title": "Even as it retires, the robot still impresses", "snippet": "retired after decades", "published": "2022"},
            {"title": "Legacy demo from years ago", "snippet": "discontinued line", "published": "2011"},
        ],
    }
    probe = build_default_probe(search_fn=lambda q: _FIXTURE.get(q, []))
    for topic in _FIXTURE:
        sig = probe.probe(topic)
        print(f"{sig.verdict.value:10} conf={sig.confidence:<4} "
              f"src={sig.source.value:18} numeric={sig.numeric_rate}  {topic}")
        print(f"            {sig.rationale}")
