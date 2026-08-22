# Decision: Pause fact extraction

**Date:** during initial ingestion build, after fact extractor was built
and live-verified working.

**Decision:** Fact extraction (`app/ingestion/processors/fact_extractor.py`)
is fully built and tested, but deliberately NOT wired into the ingestion
pipeline orchestrator or any downstream retrieval code, for now.

**Why:** It was the most complex, most LLM-call-heavy piece built so far
(one Gemini call per chunk, subject to rate limits and cost). Sequencing
it after the simpler, more foundational pieces (entity tagging, embeddings,
storage, basic retrieval) are working end-to-end gets a working, testable
core pipeline sooner. Fact extraction is a precision upgrade on top of
that core loop, not a dependency of it.

**What this means concretely:**
- `app/ingestion/pipeline.py` (the real orchestrator, not yet built as of
  this note) should call: loader → chunker → entity tagger → embedder →
  storage. It should NOT call `extract_facts_for_chunks`.
- The fact extractor's tests (`tests/ingestion/test_fact_extractor.py`)
  and live script (`scripts/test_fact_extraction_live.py`) stay in the
  codebase and stay passing - this is what makes re-integration later
  free of rework.
- Query-time retrieval (fact_lookup tool, per the architecture docs)
  will not exist until this is resumed - questions requiring precise
  fact lookup will fall back to chunk-based hybrid search only, until
  this is revisited.

**Revisit when:** the core ingestion → storage → basic retrieval loop is
working and proven, and precision on numeric/approval questions becomes
the next priority.
