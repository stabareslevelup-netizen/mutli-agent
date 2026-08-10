# INFRA REQUIREMENTS — hosts/endpoints to allowlist

This engine runs in an environment with a **network egress allowlist**. Below
is exactly what must be reachable, split by priority. Phase 0 confirmed the
Timing data layer works on the Anthropic server-side `web_search` tool alone;
GDELT + Wikipedia are the *upgrade* that turns velocity from ordinal into
numeric rate-of-change.

---

## 1. REQUIRED for v1 (engine cannot run without these)

| Host | Purpose | Notes |
|---|---|---|
| `api.anthropic.com` | Claude Agent SDK + **server-side `web_search` tool** (Timing/Research data layer) | web_search executes on Anthropic's side, so our container only needs to reach the API host. This is the *primary* data layer per the resolved infra decision. |
| `pypi.org`, `files.pythonhosted.org` | Python dependency install | already reachable |
| PostgreSQL host | `pgvector` memory store | set via `DATABASE_URL`; if self-hosted in-cluster, no public egress needed |
| Higgsfield MCP endpoint | video production | Phase 5 |
| `api.x.com` / `graph.facebook.com` / `www.googleapis.com` | distribution (X, Instagram, YouTube) | Phase 5; staged-only in `confirm` mode |

> The Timing Agent is built against the **`web_search` tool**, NOT direct
> external hosts. Do not design around the blocked hosts below — they are an
> optional numeric upgrade with graceful fallback.

---

## 2. OPTIONAL — allowlist to unlock NUMERIC velocity (recommended, free, keyless)

Both are free and require no API key. When reachable, `velocity_probe`
auto-upgrades from ordinal verdict to numeric rate-of-change. When blocked, it
degrades gracefully to the ordinal `web_search` path — no code change, no
redeploy (dependency-injected data sources; see `engine/tools/velocity_probe.py`).

### 2a. GDELT DOC 2.0 API — news-volume time-series
- **Host to allowlist:** `api.gdeltproject.org`
- **Endpoint (GET):**
  `https://api.gdeltproject.org/api/v2/doc/doc`
- **Example call (1-month volume timeline, JSON):**
  `https://api.gdeltproject.org/api/v2/doc/doc?query=<topic>&mode=timelinevol&format=json&timespan=1m`
- **Returns:** daily normalized coverage volume → we compute slope / % change.
- **Method/auth:** GET, no key, no POST.

### 2b. Wikimedia Pageviews REST API — attention time-series
- **Host to allowlist:** `wikimedia.org` (and `*.wikimedia.org`)
- **Endpoint (GET):**
  `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}`
- **Example:**
  `.../per-article/en.wikipedia/all-access/all-agents/Humanoid_robot/daily/20260501/20260601`
- **Returns:** daily pageviews for an article → we compute week-over-week rate.
- **Method/auth:** GET, no key. Send a descriptive `User-Agent` (Wikimedia policy).
- **Optional companion:** `en.wikipedia.org` (article/title resolution via the
  MediaWiki `action=query` search API) if we want topic→article matching.

---

## 3. Verification after allowlisting

Once 2a/2b are configured, confirm with:

```bash
curl -s -m 15 -A "media-engine/1.0 (ops@septem.systems)" \
  "https://api.gdeltproject.org/api/v2/doc/doc?query=humanoid%20robot&mode=timelinevol&format=json&timespan=1m" | head -c 200
curl -s -m 15 -A "media-engine/1.0 (ops@septem.systems)" \
  "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Humanoid_robot/daily/20260501/20260601" | head -c 200
```

A non-empty JSON body (not `Host not in allowlist`) means numeric velocity is
live. The probe's `available()` check flips to `True` automatically and numeric
sources take priority over the ordinal fallback.
