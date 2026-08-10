"""
Phase 3 memory-layer tests. Runnable with `python3 -m tests.test_phase3`
(no pytest, no live DB — InMemoryBackend + injected embedding providers).

Covers:
  - embedding graceful degradation (Voyage host blocked -> NULL vectors)
  - episodic/semantic: vector search when embeddings present; recency fallback
    when absent; writes still succeed with NULL embeddings
  - narrative: staked positions as active constraints
  - procedural guard: propose NEVER applies; approve requires human + voice
    check; voice regression blocks; rollback reverts; auto-apply impossible
"""
from __future__ import annotations

import asyncio

from engine.core.embeddings import (
    EmbeddingUnavailable, NullEmbeddingProvider, VoyageEmbeddingProvider, safe_embed,
)
from engine.memory.backend import InMemoryBackend
from engine.memory.episodic import EpisodicMemory
from engine.memory.narrative import NarrativeMemory
from engine.memory.procedural import (
    ApprovalRequired, LexicalVoiceChecker, NoVersionToRollback, ProceduralMemory,
    VoiceRegressionBlocked,
)
from engine.memory.semantic import SemanticMemory

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))


# A fake embedding provider that returns deterministic vectors (no network).
class FakeEmbeddings:
    name = "fake"
    dimension = 3

    def available(self) -> bool:
        return True

    async def embed(self, texts):
        # crude bag-of-chars vector so similar strings score higher
        out = []
        for t in texts:
            tl = t.lower()
            out.append([float(tl.count("a")), float(tl.count("o")), float(len(tl))])
        return out


# --- 1. embedding degradation ----------------------------------------------
async def test_embedding_degradation():
    # blocked host: post_fn raises -> EmbeddingUnavailable -> safe_embed None
    async def blocked_post(url, body, headers):
        raise RuntimeError("Host not in allowlist: api.voyageai.com")

    voyage = VoyageEmbeddingProvider(api_key="pa-test", post_fn=blocked_post)
    raised = False
    try:
        await voyage.embed(["hello"])
    except EmbeddingUnavailable:
        raised = True
    check("voyage raises EmbeddingUnavailable when host blocked", raised)
    vecs = await safe_embed(voyage, ["hello", "world"])
    check("safe_embed degrades to NULL vectors", vecs == [None, None])

    nullp = NullEmbeddingProvider()
    check("null provider not available", nullp.available() is False)
    check("null provider returns Nones", (await nullp.embed(["x"])) == [None])

    # live-ish path via injected post_fn
    async def ok_post(url, body, headers):
        return {"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(len(body["input"]))]}

    v2 = VoyageEmbeddingProvider(api_key="pa-test", post_fn=ok_post)
    out = await v2.embed(["a", "b"])
    check("voyage returns vectors when reachable", out == [[0.1, 0.2], [0.1, 0.2]])


# --- 2. episodic/semantic: vector vs recency fallback -----------------------
async def test_vector_memory():
    # with embeddings: semantic search ranks by similarity
    be = InMemoryBackend()
    epi = EpisodicMemory(be, FakeEmbeddings())
    await epi.record(brand_id="b", content="robots arrive at the factory")
    await epi.record(brand_id="b", content="aaa ooo")
    hits = await epi.query(brand_id="b", text="robots arrive at the factory", k=2)
    check("vector query returns ranked items", len(hits) == 2)
    check("vector query not degraded", epi.degraded is False)
    check("best match is the identical content", hits[0].content == "robots arrive at the factory",
          hits[0].content)

    # without embeddings: writes still work (NULL), query degrades to recency
    be2 = InMemoryBackend()
    sem = SemanticMemory(be2, NullEmbeddingProvider())
    await sem.record(brand_id="b", content="first")
    await sem.record(brand_id="b", content="second")
    rid = await sem.record(brand_id="b", content="third")
    check("write succeeds with NULL embedding", isinstance(rid, int))
    recent = await sem.query(brand_id="b", text="anything", k=2)
    check("query degrades to recency", sem.degraded is True)
    check("recency returns most-recent-first", [m.content for m in recent] == ["third", "second"],
          str([m.content for m in recent]))


# --- 3. narrative constraints ----------------------------------------------
async def test_narrative():
    be = InMemoryBackend()
    nar = NarrativeMemory(be, NullEmbeddingProvider())
    pid = await nar.stake_position(brand_id="b", position="autonomy is overstated",
                                   stance="we argue physical AI is far from autonomous")
    await nar.stake_position(brand_id="b", position="hype cycle", stance="skeptical")
    cons = await nar.constraints(brand_id="b")
    check("two active constraints", len(cons) == 2)
    check("constraint carries stance", any("autonomous" in c.stance for c in cons))
    await nar.retire_position(position_id=pid)
    cons2 = await nar.constraints(brand_id="b")
    check("retired position drops out", len(cons2) == 1)


# --- 4. procedural guard ----------------------------------------------------
async def test_procedural_guard():
    be = InMemoryBackend()
    refs = ["dark editorial authoritative cinematic physical ai never corporate always specific"]
    proc = ProceduralMemory(be, voice_checker=LexicalVoiceChecker(),
                            reference_set=refs, voice_threshold=0.25)
    await proc.seed(brand_id="b", agent="copy",
                    prompt_text="Write in dark editorial authoritative cinematic voice, always specific.")
    base = await proc.active_prompt(brand_id="b", agent="copy")

    # propose does NOT change the live prompt
    on_voice = ("Write in dark editorial authoritative cinematic physical ai voice, "
                "never corporate, always specific.")
    p1 = await proc.propose(brand_id="b", agent="copy", proposed_prompt=on_voice,
                            performance_data={"ctr": 0.9})
    still = await proc.active_prompt(brand_id="b", agent="copy")
    check("propose does NOT auto-apply", still == base)
    check("proposal is pending", len(await proc.pending_proposals()) == 1)

    # approve without a human approver is refused
    refused = False
    try:
        await proc.approve(proposal_id=p1.id, approved_by="")
    except ApprovalRequired:
        refused = True
    check("approve requires a human approver", refused)

    # voice regression is blocked + proposal rejected
    off_voice = "buy now!! click here for the top 10 robot hacks you won't believe number 7"
    p2 = await proc.propose(brand_id="b", agent="copy", proposed_prompt=off_voice,
                            performance_data={"ctr": 2.0})
    blocked = False
    try:
        await proc.approve(proposal_id=p2.id, approved_by="stephanie")
    except VoiceRegressionBlocked:
        blocked = True
    check("voice regression blocks approval", blocked)
    check("blocked proposal is rejected", (await be.get_proposal(proposal_id=p2.id)).status == "rejected")
    check("live prompt unchanged after blocked change",
          (await proc.active_prompt(brand_id="b", agent="copy")) == base)

    # a human-approved, on-voice change activates a new version
    rec = await proc.approve(proposal_id=p1.id, approved_by="stephanie")
    check("approved change activates new version", rec.version == 2 and rec.active)
    check("live prompt now the approved one",
          (await proc.active_prompt(brand_id="b", agent="copy")) == on_voice)
    check("approver recorded", rec.approved_by == "stephanie")

    # rollback reverts to v1 in one action
    target = await proc.rollback(brand_id="b", agent="copy")
    check("rollback returns prior version", target.version == 1 and target.active)
    check("live prompt restored to base",
          (await proc.active_prompt(brand_id="b", agent="copy")) == base)

    # rolling back again with no prior version errors cleanly
    err = False
    try:
        await proc.rollback(brand_id="b", agent="copy")
    except NoVersionToRollback:
        err = True
    check("rollback with no prior version errors cleanly", err)


async def main() -> int:
    await test_embedding_degradation()
    await test_vector_memory()
    await test_narrative()
    await test_procedural_guard()
    for status, name, detail in results:
        line = f"  [{status}] {name}"
        if detail and status == FAIL:
            line += f"  -- {detail}"
        print(line)
    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
