# Change 001 — Two-Layer Semantic Cache (Redis) for the Query Pipeline

> **Where this lives:** `docs/changes/` — every future change gets a numbered file here.
> **Status of the code described:** written but **not yet committed** (working tree only).
> **Base commit:** `b1591ca Add EngineConfig class and update Postgres/Gemini circuit breakers to use config values`
> **Tests at time of writing:** `153 passed` (was ~115 before this work).

---

## 0. TL;DR — What changed in one paragraph

Until now, every single question asked to the system went through the full expensive
pipeline: classify the intent with an LLM call → embed the question with an LLM call →
search Postgres with pgvector → rerank → call the LLM again to write the answer.

That is a lot of work (and a lot of API cost) to answer **the same question twice**.

This change adds a **cache** in front of the expensive part of the pipeline. Now, when
someone asks a question the system has already answered, the system recognises it and
replays the stored answer instantly — no embedding call, no database search, no
generation call.

The interesting part is *how* it recognises "the same question". A naive cache that
matches on meaning alone is **dangerous** in this domain, because two SOP questions can
sound almost identical and still have different correct answers
(*"Who approves a Team Lead's leave?"* vs *"Who approves an employee's leave?"*). So the
cache here is two-layered:

| Layer | What it matches on | Cost to check | Risk of a wrong answer |
|---|---|---|---|
| **Layer 0** | Exact same words (case/whitespace insensitive) **+** the exact same versions of every document | ~1 ms (one Redis GET) | Essentially zero |
| **Layer 1** | Similar *meaning* (embedding similarity ≥ 0.88) **+** the same **entities** (people / roles / topics) must match exactly | ~1 embedding call + a small in-process scan | Low — but only *because* of the entity guard |

The design rule that governs the whole thing: **the cache is an optimisation, never a
dependency.** If Redis dies, if Gemini dies, if Postgres dies — the user still gets an
answer. The cache just quietly stops being used.

---

## 1. Why was this change needed?

### 1.1 The business problem

In a company SOP assistant, the same handful of questions get asked over and over:
*"Who approves WFH?"*, *"What's the leave process?"*, *"How do I raise a defect?"*
Every repeat of those questions was paying full price — LLM calls, database round trips,
latency — to produce an answer that already existed.

### 1.2 The technical problem

You cannot just add a naive cache. Research on semantic caches found a repeatable failure
mode:

> Two questions that share sentence structure but differ in one critical detail (a name,
> a role, a time period) can score **above a 0.95 similarity threshold** and still return
> the **wrong** cached answer.

In this project's real SOP data that looks like this:

| Question A | Question B | Structurally identical? | Same answer? |
|---|---|---|---|
| Who approves a **Team Lead's** leave? | Who approves an **employee's** leave? | Yes — same words, one word changed | **No — different approvers** |

A pure vector-similarity cache would confidently serve A's answer to B. That is the
"wrong answer delivered with total confidence" failure — the worst kind of bug for an
SOP assistant, because nobody notices it.

**The fix:** similarity finds *candidates*; a **second, independent check** (the entity
guard) decides whether a candidate is trustworthy.

### 1.3 Why not just reuse the retrieval machinery?

The system already has an entity tagger (people / roles / topics) built during ingestion.
The cache reuses that exact vocabulary instead of inventing a second tagging system, and
it uses the *same* tagger on the query text so that query-side tags and chunk-side tags
are guaranteed to be comparable. Zero extra model, zero extra cost, one consistent
vocabulary.

---

## 2. The concepts you need to know (plain English first, technical after)

### 2.1 Cache — non-technical

> A cache is a **scratch pad**. Instead of redoing an expensive calculation, you keep
> the answer on your desk and hand it over the next time someone asks the same thing.

Think of a receptionist: the first time someone asks "where do I park?", the receptionist
asks around and finds out. The next ten times, the receptionist answers immediately from
memory — but if the parking rules change, the receptionist updates their memory.

**The two rules of a well-built cache:**

1. **Correctness first.** A wrong cached answer is worse than no cache at all.
2. **The cache must never be required.** If the scratch pad burns down, you still do the
   work the slow way.

### 2.2 Cache — technical

A cache is a key/value store with a **time to live (TTL)**. Here the key/value store is
**Redis** (an in-memory database). TTL = "delete this entry automatically after N seconds"
(the TTL here is 86 400 s = 24 h).

* **Key** — a deterministic string that identifies "this question, in this document state".
* **Value** — the JSON blob of the answer (`answer`, `confidence`, `sources`).
* **Write-through** — the answer is written into the cache *immediately after* it is
  produced, so the next request can find it.

### 2.3 Embedding & cosine similarity — technical

An **embedding** turns text into a list of 768 floats ("a point in meaning-space").
Texts that mean similar things land near each other; different things land far apart.

**Cosine similarity** is the score of how close two points are:

```
similarity = (A · B) / (|A| × |B|)     # dot product ÷ magnitudes
```

It returns a number from **-1** (opposite) to **+1** (identical direction). Here,
`0.88` is the bar: below it, a candidate is discarded outright.

### 2.4 The entity guard — the heart of the design

Similarity says *"these two sentences are shaped alike"*. The entity guard says
*"…and they talk about the same actors"*.

For each of `people`, `roles`, `topics` the two entity dicts must match **exactly as
sets**. Partial overlap is **not** a match:

```python
if set(a["roles"]) != set(b["roles"]):   # {"Team Lead"} vs {"Employee"} -> NOT a match
    return False
```

So *"Team Lead"* vs *"Employee"* → different roles → different answer → **cache miss**.
Correct behaviour, achieved with a 6-line function.

### 2.5 Version-tied invalidation — how staleness is prevented

Every cache key embeds the **versions of every currently published document**:

```
{ "WFH Policy": "1.0", "Leave Policy": "2.1" }
```

If someone edits the WFH policy, its version bumps `1.0 → 1.1`, the dict changes, and
**every cache key changes with it**. The old entries become unreachable — they are not
looked up again. No delete logic, no event bus, no risk of serving a stale answer.

> This is deliberately **coarse**: editing one document invalidates *all* cached
> answers, not just the ones about that document. Correct and simple first; per-document
> granularity can come later if the hit rate suffers.

The TTL (24 h) is only a **safety net** for the case where an invalidation event is
somehow missed — it is *not* the primary mechanism.

### 2.6 Circuit breaker — technical

The system already had this pattern for Gemini and Postgres. A circuit breaker is a
3-state guard around a flaky dependency:

| State | Behaviour |
|---|---|
| `CLOSED` | Normal — calls go through. |
| `OPEN` | Too many failures — calls **fail fast without even trying**, until the cooldown expires. |
| `HALF_OPEN` | Cooldown elapsed — **one** probe call is allowed; success closes the breaker, failure re-opens it. |

**This change adds a third, independent breaker for Redis** (`app/cache/redis_client.py`,
threshold 3 failures / 15 s cooldown — deliberately more aggressive than the others,
because caching is purely optional: we want to give up on Redis *fast* and get back to
normal "cache-skipped" operation).

Three independent dependencies → three independent failure domains:

```
Gemini breaker   |  Postgres breaker   |  Redis breaker
(app/llm/gemini) |  (app/database)     |  (app/cache/redis_client)
```

One dying does not drag the others down. (And per the existing design rule, the breaker's
state lives **in process memory**, never in Redis — otherwise a cache outage would also
destroy the mechanism protecting against other outages.)

### 2.7 Fail-open vs fail-closed

**Fail-open** = when the optional thing breaks, carry on without it.
**Fail-closed** = when the optional thing breaks, refuse to serve.

This cache is **fail-open** everywhere:

* Redis down → `get_layer0` / `get_layer1` return `None` (= "miss"), never raise.
* Gemini down during query tagging → Layer 1 is skipped, Layer 0 still works.
* Postgres down → caching is skipped for this turn entirely, the real pipeline runs and
  handles its own database problems.

---

## 3. File-by-file map of the change

### New files

| File | What it is |
|---|---|
| `app/cache/__init__.py` | Package marker. |
| `app/cache/redis_client.py` | Owns the Redis connection **and the Redis circuit breaker**. Single shared client, 2 s socket timeouts. |
| `app/cache/layer0_exact.py` | Layer 0: exact-match key building, get, set. |
| `app/cache/layer1_semantic.py` | Layer 1: entity-guarded semantic matching, cosine similarity, get, set. |
| `app/graph/nodes/cache/cache_lookup.py` | The graph node that *reads* the cache. |
| `app/graph/nodes/cache/cache_write.py` | The graph node that *writes* the cache. |
| `app/repositories/document_versions.py` | Read-only helper: `{title: version}` for all published + active docs. |
| `tests/cache/test_layer0_exact.py` | 7 tests for Layer 0. |
| `tests/cache/test_layer1_semantic.py` | 7 tests for Layer 1 (the entity-guard ones matter most). |
| `tests/cache/test_cache_lookup_node.py` | 6 tests for the lookup node. |
| `tests/cache/test_cache_write_node.py` | 9 tests for the write node (what must **never** be cached). |
| `tests/graph/test_builder_cache.py` | 7 end-to-end tests through the **real compiled graph**, real cache nodes, `fakeredis` as the store. |
| `tests/graph/conftest.py` | Auto-fixture that stubs the cache nodes to pass-throughs, unless a test opts in with `@pytest.mark.real_cache`. |
| `tests/database/test_document_versions.py` | 2 tests for the version helper. |

### Modified files

| File | Change |
|---|---|
| `app/config/engine_config.py` | Added 5 new settings: `redis_host`, `redis_port`, `redis_db`, `cache_ttl_seconds`, `cache_semantic_similarity_threshold`. |
| `app/graph/builder.py` | Registered the two new nodes, added the conditional edge `cache_lookup → (hit: END) / (miss: retrieve)`, rewired `informational` and `follow_up` to go through `cache_lookup`, and moved `synthesize → END` to `synthesize → cache_write → END`. |
| `app/graph/state.py` | Added 4 cache fields to `GraphState`: `cache_hit`, `_cache_query_embedding`, `_cache_query_entities`, `_cache_document_versions`. |
| `requirements.txt` | Added `fakeredis>=2.20.0` (test-only fake Redis). `redis>=5.0.0` was already listed. |

---

## 4. The new pipeline, step by step

### 4.1 Before vs after (graph shape)

**BEFORE**

```
START → classify_intention ─┬→ answer_chitchat → END
                            ├→ answer_meta → END
                            ├→ answer_actionable → END
                            ├→ route_to_escalation → END
                            ├→ (informational) → retrieve → rerank → synthesize → END
                            └→ (follow_up) → resolve_follow_up → retrieve → rerank → synthesize → END
```

**AFTER**

```
START → classify_intention ─┬→ answer_chitchat → END
                            ├→ answer_meta → END
                            ├→ answer_actionable → END
                            ├→ route_to_escalation → END
                            ├→ (informational) ─┐
                            └→ (follow_up) → resolve_follow_up ─┘
                                                        │
                                                   cache_lookup
                                                     │  ├─ hit  → END          ← NEW: skips everything
                                                     │  └─ miss → retrieve → rerank → synthesize
                                                     │                                          │
                                                     │                                    cache_write ← NEW
                                                     │                                          │
                                                     └──────────────────────────────────────→ END
```

Two things to notice:

1. **Only `informational` and `follow_up` touch the cache.** Chitchat, meta, actionable
   and escalation are either instant canned responses or human handoffs — nothing to
   cache, and caching them would be noise.
2. **`cache_write` runs after `synthesize`, before `END`.** It is deliberately *not*
   folded into synthesize — one node, one job, and it can be tested in isolation.

### 4.2 Step 1 — `cache_lookup` (the reader)

`app/graph/nodes/cache/cache_lookup.py`

```
query = resolved_query or user_message          # follow-ups use the resolved form
   │
   ├─ get_all_document_versions()                # Postgres read (through the DB breaker)
   │     └─ FAILS → return {cache_hit: False}    # skip caching this turn, don't raise
   │
   ├─ get_layer0(query, versions)                # cheap Redis GET
   │     └─ HIT → return {cache_hit: True, answer, confidence, sources}   ← DONE
   │
   ├─ embed_query(query)                         # 1 LLM-ish call (Gemini embedding)
   │  _extract_query_entities(query)             # 1 Gemini call (same tagger as ingestion)
   │     └─ EITHER FAILS → return {cache_hit: False, versions}   # Layer 0 still writable later
   │
   ├─ get_layer1(embedding, entities, versions)  # scan the small local index
   │     └─ HIT → return {cache_hit: True, answer, confidence, sources}   ← DONE
   │
   └─ FULL MISS → return {cache_hit: False,
                          versions,
                          embedding,               # stashed so cache_write doesn't redo them
                          entities}
```

**Why stash the embedding and entities in state?** Because `cache_write` needs them, and
recomputing them would mean a second embedding call and a second tagging call. The
lookup node computes once and passes them forward through `GraphState`. That is exactly
what the three `_cache_*` fields in `app/graph/state.py` are for.

**Why is Layer 0 checked first?** Because Layer 0 costs one Redis GET, while Layer 1
costs an embedding call + a tagging call. If we checked Layer 1 first, every exact-repeat
question would still pay for embedding + tagging before it could hit Layer 0. Order
matters for latency and cost.

### 4.3 Step 2 — `cache_write` (the writer)

`app/graph/nodes/cache/cache_write.py`

```
if cache_hit              → write nothing (it came FROM the cache)
if db_unavailable         → write nothing (the answer says "database is down" - not a real answer)
if no answer              → write nothing
if confidence is "Low"    → write nothing    ← THE IMPORTANT RULE
if versions missing       → write nothing (no key can be built)

otherwise:
    response = {answer, confidence, sources}
    set_layer0(query, versions, response)                      # always
    set_layer1(embedding, entities, versions, response)        # only if embedding+entities exist
```

**Why "Low confidence never gets cached" is the most important rule in the whole change:**

A `Low` confidence answer is, by construction, one of:

* the Gemini-outage fallback — *"I'm having trouble generating a written answer…"*
* the *"I don't have information on this in the SOPs"* answer
* an answer built from weak retrieval matches

Caching any of those would mean serving a broken answer **confidently** to the next
person, for up to 24 hours, long after the cause is gone. There is a dedicated
regression test for exactly this: `test_gemini_outage_fallback_is_not_cached_and_recovers`
in `tests/graph/test_builder_cache.py` — it asserts that an outage message is never
cached and that the very next request (after recovery) returns the real answer.

### 4.4 Layer 0 internals — `app/cache/layer0_exact.py`

**Key construction:**

```python
normalized = " ".join(query.lower().strip().split())        # case + whitespace insensitive
versions_str = json.dumps(dict(sorted(versions.items())))  # sorted -> key order never matters
digest = sha256(f"{normalized}|{versions_str}")
key = f"cache:l0:{digest}"
```

* Lowercasing and whitespace collapsing means *"Who approves WFH?"* and
  *"who approves wfh"* hit the same entry.
* Sorting the versions dict means `{A, B}` and `{B, A}` produce the same key.
* SHA-256 keeps the key a fixed, safe length no matter how long the question is.

**Read:** `get()` → `None` on miss **or** on any Redis problem (both `CircuitOpenError`
and generic `Exception` are swallowed).
**Write:** `set(..., ex=ttl)` — TTL applied automatically. Write failures are swallowed:
*a failed cache WRITE must never break the response the user is about to receive.*

### 4.5 Layer 1 internals — `app/cache/layer1_semantic.py`

**Storage:** Redis doesn't do vector search natively, so Layer 1 keeps a **single Redis
list** `cache:l1:index`. Each element is a JSON entry:

```json
{
  "embedding": [ ...768 floats... ],
  "entities":  { "people": [...], "roles": [...], "topics": [...] },
  "document_versions": { "WFH Policy": "1.0" },
  "response":  { "answer": "...", "confidence": "High", "sources": [...] },
  "cached_at": 1760000000.0
}
```

Written with `LPUSH` (newest first) and trimmed with `LTRIM 0 499` — the index is
capped at **500 entries** so the per-query linear scan stays fast. This is explicitly
sized for personal-project scale (hundreds to low thousands of cached queries), **not**
millions — at that scale you would move to a proper vector index (Redis Stack / pgvector).

**Read:** scan the list; for each entry, in order:

```
1. entry.document_versions != current versions   → skip (stale)
2. cosine_similarity < 0.88                      → skip (not similar enough)
3. entity sets don't match exactly               → skip (THE GUARD)
4. otherwise → candidate; keep the highest score
```

Return the best candidate's `response`, or `None`.

**The one thing to internalise:** steps 1–3 are all **AND**-ed. Similarity alone can
never produce a hit. The entity guard is what turns a dangerous cache into a safe one.

### 4.6 Document versions — `app/repositories/document_versions.py`

```sql
SELECT title, version FROM sop_documents
WHERE status = 'Published' AND is_active = 'Active'
```

Only **published + active** documents count — exactly the set of documents the retrieval
layer is allowed to serve. If a document is unpublished or deactivated, it disappears
from the dict, the keys change, and its cached answers become unreachable. Same
mechanism, same guarantee.

The read goes through the **existing Postgres circuit breaker** (`with_db_breaker`), so a
DB outage surfaces as a `CircuitOpenError`, which `cache_lookup` catches → caching is
skipped, the real pipeline runs.

This module is deliberately **separate** from `app/repositories/document_repository.py`:
that file is the single *writer* of `sop_documents`/`sop_chunks` and is also extended by
the web app. The cache only needs *reads* and should not entangle with the writer.

---

## 5. Non-technical summary (for teammates / stakeholders)

**What did we build?** A memory for the assistant.

**Why?** Because the same SOP questions were being re-answered from scratch every time,
paying for AI calls and database searches each time.

**How does it decide something is the same question?**

1. **First, exact match.** If the words are the same (ignoring capitalisation and
   extra spaces) *and* no SOP document has changed since, it replays the stored answer.
2. **If not, meaning + facts.** It checks "does this mean the same thing?" using the same
   AI embedding used for document search. But — and this is the important part — it also
   checks "is this about the same people, the same job roles and the same topics?".
   *If not, the cache is ignored.* Two questions can be worded almost identically and
   still have different answers ("who approves a **Team Lead's** leave?" vs "who approves
   an **employee's** leave?"); that check is what stops the assistant from confidently
   giving the wrong answer.

**What if the memory breaks?** Nothing breaks for the user. If Redis (the memory store)
is down, the assistant quietly behaves like it did before — it just recomputes answers
using the normal pipeline. Same if the AI or the database is briefly unavailable.

**What is deliberately never remembered?**

* Answers produced while the database was down.
* Any answer the system itself rated as low confidence — including "I couldn't find that"
  and "I'm having trouble right now". Those must never be replayed to the next person.

**How do we know it's not stale?** Every stored answer is stamped with the version
number of every SOP document. Edit a document → its version number changes → every
saved answer is automatically considered out of date. There's no manual "clear the
cache" step to forget. On top of that, everything expires automatically after 24 hours
as a safety net.

**What's the cost/risk trade-off?**

| | Benefit | Cost |
|---|---|---|
| Layer 0 | Instant answer, zero AI cost | One Redis lookup |
| Layer 1 | Instant answer for rephrased questions | One AI embedding + one AI tagging call, plus a small scan |
| Both | Fewer AI calls, lower latency, consistent answers | A little complexity; a small risk of a wrong cached answer, mitigated by the entity check |

---

## 6. Failure matrix — "what happens if X breaks?"

| What breaks | Behaviour | User impact |
|---|---|---|
| **Redis down / slow** | Breaker opens after 3 failures → calls fail fast → cache returns "miss" | None — normal pipeline answers, just without the speed-up |
| **Gemini down** (during query tagging) | Layer 1 skipped, Layer 0 still works | None |
| **Gemini down** (during synthesis) | `synthesize` returns sources instead of a written answer, confidence `Low` | User sees a graceful "here are the relevant SOP sections" message; **it is never cached** |
| **Postgres down** | Version read fails → caching skipped for that turn; `retrieve` sets `db_unavailable` → `synthesize` returns a distinct honest message | User sees "the system can't search right now", clearly distinguished from "we have no info on this" |
| **A document is edited** | Version bumps → keys change → old entries unreachable | Automatic, no manual cache purge |
| **A document is unpublished** | It leaves the version dict → keys change | Automatic |
| **Bad answer would be produced** (low confidence) | Never written to the cache | No stale bad answers |

---

## 7. Configuration

All settings live in `EngineConfig` (`app/config/engine_config.py`) and are overridable
via environment variables with the `ENGINE_` prefix (from `.env`).

| Setting | Default | Env var | What it controls |
|---|---|---|---|
| `redis_host` | `localhost` | `ENGINE_REDIS_HOST` | Redis server address |
| `redis_port` | `6379` | `ENGINE_REDIS_PORT` | Redis port |
| `redis_db` | `0` | `ENGINE_REDIS_DB` | Redis logical database number |
| `cache_ttl_seconds` | `86400` (24 h) | `ENGINE_CACHE_TTL_SECONDS` | Entry lifetime; safety-net backstop only |
| `cache_semantic_similarity_threshold` | `0.88` | `ENGINE_CACHE_SEMANTIC_SIMILARITY_THRESHOLD` | Minimum cosine similarity for a Layer 1 candidate. Recommended range 0.85–0.90 |

> **Tuning guidance:** raising the threshold makes Layer 1 more conservative (fewer hits,
> safer). Lowering it increases hits but relies more heavily on the entity guard. The
> entity guard — not this number — is what screens out the dangerous false positives, so
> this number is a tuning knob, not a safety mechanism.

---

## 8. Tests

**Total: 153 passing** (`python -m pytest tests -q`). ~38 of those are new.

| Test file | What it proves |
|---|---|
| `tests/cache/test_layer0_exact.py` | Exact query + same versions → hit; case/whitespace differences still hit; different query → miss; **version change → miss**; key is order-independent; Redis unavailable → `None`, not an error; write failure doesn't raise |
| `tests/cache/test_layer1_semantic.py` | High similarity + same entities → hit; **high similarity + different entities → miss (the core regression test)**; low similarity → miss even with matching entities; version mismatch → miss; **partial entity overlap → miss**; Redis unavailable → `None`; write failure doesn't raise |
| `tests/cache/test_cache_lookup_node.py` | Layer 0 hit returns the answer and skips everything else; Layer 1 hit returns the answer; full miss stashes everything `cache_write` needs; resolved query used instead of raw follow-up; Postgres failure skips caching without raising; embedder/Gemini failure degrades to Layer 0 only |
| `tests/cache/test_cache_write_node.py` | High confidence → both layers; Medium → cached; **Low → never cached**; **Gemini-outage fallback → never cached**; `db_unavailable` → never cached; cache hit → not rewritten; missing versions → nothing written; missing embedding → Layer 0 only; empty answer → nothing written |
| `tests/graph/test_builder_cache.py` | End-to-end through the **real compiled graph** with real cache nodes + `fakeredis`: second identical question served from cache (1 synth call, 1 search call); document version change invalidates; low confidence not cached; Gemini outage not cached and then recovers; similar question + same entities → Layer 1 hit; similar question + different entities → miss; **Redis fully down → pipeline still answers** |
| `tests/database/test_document_versions.py` | Only published + active documents are returned; a version bump is reflected |
| `tests/graph/conftest.py` | Auto-stub of both cache nodes for all other graph tests (so routing tests don't need Redis/Postgres), with an opt-in `@pytest.mark.real_cache` marker for the real thing |

---

## 9. Known gaps, risks and deliberate simplifications

Read this section before changing the cache code — it prevents re-introducing the bugs
the design was written to avoid.

| # | Item | Severity | Notes |
|---|---|---|---|
| 1 | **`cache/invalidation.py` does not exist.** Both `engine_config.py` and `layer0_exact.py` docstrings reference it. | Docs | Invalidation is currently handled implicitly by the version-tie (see below). Either write the module or fix the docstrings. |
| 2 | **Layer 1 invalidation is implicit, not explicit.** `get_layer1` skips any entry whose `document_versions` dict differs. | By design | The docstring says Layer 1 "DOES need explicit invalidation since they're not keyed by exact version" — in practice the equality check makes it behave like Layer 0. The docstring is now out of date. |
| 3 | **Coarse invalidation.** Editing one document invalidates *every* cached answer. | Accepted | Explicitly called out as a deliberate simplification. Per-document scoping is a possible later upgrade. |
| 4 | **The Layer 1 index TTL is reset on every write.** `expire` is re-applied on each `set_layer1`, so a busy cache never expires and individual old entries can live far beyond 24 h. | Low | Version-tie still protects correctness; only memory/disk growth is affected. |
| 5 | **Stale Layer 0 keys are never actively deleted.** They just expire via TTL. | Low | Memory grows until expiry. Fine at this scale. |
| 6 | **Query-side entity tagging costs a Gemini call.** So Layer 1 hits are not free — they cost an embedding + a tagging call, and only save a DB search + a generation call. | Perf | Mitigated: Layer 0 is checked *first*, so exact repeats pay only a Redis GET. |
| 7 | **`_cache_*` private fields leak into the final graph state** returned by LangGraph. | Cosmetic | They're internal; if the API serialises the whole state, filter them at the API boundary. |
| 8 | **The `_cache_*` fields are still present in state on a Layer 1 hit** (the hit branch returns a fresh dict without them). | Harmless | On a hit the graph goes straight to `END`, and `cache_write` never runs. |
| 9 | **Linear scan over up to 500 × 768-dim vectors per Layer 1 lookup.** | Perf | ~380 k float operations — fine at this scale. Move to pgvector/Redis Stack if the cache grows to 100 k+ entries. |
| 10 | **No cache-hit metrics or logging.** | Ops | No way to measure the hit rate today. Worth adding when tuning the threshold. |
| 11 | **The Redis breaker state is per-process.** With multiple server processes, each has its own breaker. | Accepted | Consistent with the existing design rule for the Gemini/Postgres breakers. |
| 12 | **`synthesize.py` still hardcodes `HIGH_CONFIDENCE_THRESHOLD = 0.6` / `MEDIUM = 0.4`** even though `EngineConfig` has `high_confidence_threshold` / `medium_confidence_threshold`. | Pre-existing | Not introduced by this change, but now more visible because the cache keys off confidence. |
| 13 | **`web/` was not touched.** The web app's chat route uses the graph, so it inherits caching automatically — but there is no cache visibility (hit/miss) in the UI and no cache-control route. | Optional | Next logical step. |
| 14 | **The whole change is uncommitted** in the working tree. | Process | Commit before it grows further. |

---

## 10. Mental model / cheat sheet

```
                ┌──────────────────────────────────────────┐
                │  CACHE = optimisation, NEVER a dependency│
                └──────────────────────────────────────────┘

  question
     │
     ▼
  ┌───────────┐  not follow_up  →  terminal responses (chitchat/meta/actionable/escalation)
  │ classify  │
  └─────┬─────┘
        │ informational
        ▼
  ┌───────────┐   ┌─────────────────────────────────────────────┐
  │ resolve   ├──►│ (follow_up only) rewrite question standalone │
  │ follow_up │   └─────────────────────────────────────────────┘
  └─────┬─────┘
        │
        ▼
  ╔═══════════╗   HIT  →  answer straight from state → END
  ║cache_lookup║   MISS →  carry on
  ╚═════╤═════╝
        │
        ├── Layer 0: sha256(normalised_query + sorted document versions)   [cheap, exact]
        │
        └── Layer 1: cosine ≥ 0.88  AND  entity sets match exactly         [smart, guarded]
                                    │
                                    ▼
                            retrieve → rerank → synthesize
                                    │
                                    ▼
                            cache_write  (only if confidence ∈ {High, Medium}
                                         and not db_unavailable and answer non-empty)
                                    │
                                    ▼
                                   END
```

**Three sentences to remember:**

1. Layer 0 = same words + same document versions → free answer.
2. Layer 1 = similar meaning **AND** identical people/roles/topics → free answer;
   similarity alone is never enough.
3. Low-confidence answers, outage answers and database-down answers are **never** cached.

---

## 11. How to run and verify this yourself

```powershell
# 1. Install the new test dependency
pip install -r requirements.txt

# 2. Run the whole suite
python -m pytest tests -q

# 3. Run just the cache tests
python -m pytest tests/cache tests/graph/test_builder_cache.py -v

# 4. Run the real (non-faked) cache end-to-end tests
python -m pytest tests/graph/test_builder_cache.py -v -m real_cache
```

To exercise it against real services you need Redis running locally:

```powershell
docker run -d -p 6379:6379 redis:7
# or, if Redis is already installed:
redis-server
```

Then set in `.env`:

```
ENGINE_REDIS_HOST=localhost
ENGINE_REDIS_PORT=6379
ENGINE_CACHE_TTL_SECONDS=86400
ENGINE_CACHE_SEMANTIC_SIMILARITY_THRESHOLD=0.88
```

---

## 12. Where to look in the code (quick reference)

| I want to understand… | Look at |
|---|---|
| How the graph is wired | `app/graph/builder.py` |
| The cache read step | `app/graph/nodes/cache/cache_lookup.py` |
| The cache write step & the "never cache" rules | `app/graph/nodes/cache/cache_write.py` |
| Exact-match keying | `app/cache/layer0_exact.py` |
| Semantic matching + the entity guard | `app/cache/layer1_semantic.py` |
| The entity guard itself | `app/cache/layer1_semantic.py` → `_entities_match()` |
| The cosine similarity function | `app/cache/layer1_semantic.py` → `_cosine_similarity()` |
| Redis connection + Redis breaker | `app/cache/redis_client.py` |
| Version-tie invalidation source | `app/repositories/document_versions.py` |
| The new settings | `app/config/engine_config.py` |
| The state fields | `app/graph/state.py` |
| Circuit breaker mechanics (3 states) | `app/reliability/circuit_breaker.py` |
| Confidence computation | `app/graph/nodes/generation/synthesize.py` |
| Entity tagger (shared by ingestion + query side) | `app/ingestion/processors/entity_tagger.py` |

---

*Next change doc → `docs/changes/002-*.md` (to be written when the next change lands).*
