# Meldep SOP Assistant

Agentic RAG system for answering SOP questions in Meldep. Full architecture
is documented in `docs/architecture/` (not yet copied in - see note below).

## Build status (updated as pieces are completed)

| Piece | Status | Location |
|---|---|---|
| Loader interface (`BaseLoader`) | Done, tested | `app/ingestion/loaders/base.py` |
| `.txt` loader | Done, tested | `app/ingestion/loaders/txt_loader.py` |
| Rule-unit chunker | Done, tested | `app/ingestion/processors/chunker.py` |
| Fact extractor | **Built and tested, PAUSED** - not wired into the pipeline. See `docs/decisions/` | `app/ingestion/processors/fact_extractor.py` |
| Entity tagger | Done, tested (mocked) | `app/ingestion/processors/entity_tagger.py` |
| Embeddings | Done, tested (mocked) | `app/ingestion/embeddings/embedder.py` |
| Postgres + pgvector schema | Done, tested against real local Postgres | `app/database/schema.sql` |
| Document repository (atomic version-swap) | Done, tested against real local Postgres | `app/repositories/document_repository.py` |
| Ingestion pipeline orchestrator | Done, tested | `app/ingestion/pipeline.py` |
| Structure-quality validator | Done, tested against real messy vs. clean data | `app/ingestion/processors/structure_validator.py` |
| Hybrid retrieval (vector + keyword) | Done, tested against real local Postgres | `app/retrieval/retrievers/hybrid_retriever.py` |
| Intention router (6-way, LangGraph) | Done, tested via real compiled graph | `app/graph/builder.py`, `app/graph/nodes/routing/` |
| Retrieve + rerank + synthesize nodes | Done, tested via real compiled graph | `app/graph/nodes/retrieval/`, `app/graph/nodes/generation/` |
| Follow-up query resolution | Done, tested | `app/graph/nodes/retrieval/resolve_follow_up.py` |
| Everything else | Scaffolded (empty), not started | see full tree below |

**The full question-to-answer loop now genuinely works end to end.** `python scripts/test_full_pipeline_live.py` asks real questions through the complete compiled graph - classify_intention decides the path, informational/follow_up questions actually retrieve from your real ingested Postgres data, get reranked (currently a top-5 slice, see rerank.py's docstring for the honest state of this), and get synthesized into a real answer with citations and a confidence level. Rerank is intentionally simple for now (a slice, not a real cross-encoder model) - a documented, deliberate simplification, not an oversight.

**This is the first point where a real question gets a real answer from stored data.** Run `python scripts/test_retrieval_live.py` after ingesting your samples - it searches your actual stored chunks using real hybrid (vector + keyword) search, no mocking.

**The query-time pipeline now genuinely starts with LangGraph, not just a plan for it.** `app/graph/builder.py` compiles a real StateGraph: classify_intention is a node (one LLM call, returns one of six labels), and route_after_intention is a conditional-edge router function that reads the label and picks the next node - proving the six-way split is provably real (Section 3's requirement), not just implied by a prompt. The informational/follow_up paths currently point at a documented placeholder node until retrieve -> rerank -> synthesize exist.

**Fact extraction is paused, not abandoned.** It was fully built and verified working (live-tested against real WFH policy data, correctly preserved "Team Lead AND Vikas Sir" rather than blurring it to "or"). It was paused to reduce complexity while the core ingestion -> storage -> retrieval loop gets built and proven first. It will be re-integrated later as a precision upgrade. The pipeline orchestrator does NOT call it - confirmed in `app/ingestion/pipeline.py`.

**This is the first point where data actually persists.** Everything before this ran in memory and printed to your terminal. From here on, `python scripts/run_ingestion.py` writes real rows into Postgres that survive after the script exits - and running it again with an unchanged document correctly skips it, while a version bump correctly replaces the old chunks atomically (tested and proven in `tests/ingestion/test_pipeline.py::test_version_change_triggers_atomic_update`).

## Setup

1. **Install and start Postgres locally**, with the pgvector extension.
   On Ubuntu/Debian:
   ```
   sudo apt-get install postgresql postgresql-contrib
   sudo apt-get install postgresql-16-pgvector   # match your installed postgres version
   sudo service postgresql start
   ```
   On Mac: `brew install postgresql pgvector`, or use Postgres.app with
   the pgvector extension added.

2. **Create the dev database and user:**
   ```
   sudo -u postgres psql -c "CREATE USER meldep_dev WITH PASSWORD 'your_local_password' SUPERUSER;"
   sudo -u postgres psql -c "CREATE DATABASE meldep_sop_dev OWNER meldep_dev;"
   ```

3. **Apply the schema:**
   ```
   psql -U meldep_dev -d meldep_sop_dev -f app/database/schema.sql
   ```

4. Copy `.env.example` to `.env` and fill in BOTH your Gemini key and
   your Postgres connection details:
   ```
   cp .env.example .env
   ```
   Never commit `.env` - it's already in `.gitignore`.

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Run the automated test suite. Most tests are mocked (no credentials
   needed at all), but `tests/database/` and
   `tests/ingestion/test_pipeline.py` are integration tests requiring
   your real local Postgres to be running - they do NOT need a real
   Gemini key, AI calls are mocked even in these:
   ```
   python -m pytest tests/ -v
   ```
   Current test count: 39 passing.

7. To see the REAL full pipeline (uses your actual Gemini key, makes
   real paced calls, writes real rows to your Postgres):
   ```
   python scripts/run_ingestion.py
   ```
   Run it TWICE in a row. The second run should report all 3 sample
   documents as "Skipped (unchanged)" - that's the version-check
   working correctly, not a bug.

## Why the structure looks like this

- `app/ingestion/loaders/` is an interface (`base.py`) + implementation
  (`txt_loader.py`) split on purpose. Meldep's team is deciding the real
  data-delivery mechanism separately (confirmed: `.txt` files for now,
  mechanism for future updates not yet decided). When that's finalized,
  a new loader gets added here - nothing in `processors/` or beyond
  should need to change.
- `app/ingestion/processors/chunker.py` implements rule-unit chunking:
  never splits a heading's bullet list from its lead-in sentence, never
  splits a numbered FAQ question from its answer. See the docstring and
  inline comments for the one known, deliberate limitation (ambiguous
  3-level heading nesting in plain text with no indentation - fails
  safely with a flatter label, never a broken chunk boundary).
- `app/ingestion/processors/entity_tagger.py` uses category/sub_category
  straight from document metadata (zero AI cost, zero drift) and
  constrains the AI to a fixed fine_grained_topics vocabulary, built
  from real observed data - see the file's docstring for the specific
  bugs (casing drift, topic invention) this design fixes.
- `app/ingestion/embeddings/embedder.py` deliberately uses
  gemini-embedding-001 (stable/GA) over gemini-embedding-2 (preview,
  multimodal, incompatible embedding space, no need for us) - see the
  file's docstring.
- `app/repositories/document_repository.py` is the ONLY place that
  writes to sop_documents/sop_chunks. This is what implements atomic
  version-swap: a document update fully writes its new chunks before
  deleting the old version's chunks, in one transaction - see
  `tests/database/test_document_repository.py::test_update_swaps_old_chunks_for_new_atomically`
  for the concrete, tested proof of this behavior.
- `tests/` mirrors `app/` - every processor/loader should have a
  corresponding test file with real assertions, not print statements.
  `tests/database/` and `tests/ingestion/test_pipeline.py` are
  integration tests against a real local Postgres, not mocks - this is
  deliberate, since transaction/array/vector behavior is exactly what
  mocks would hide bugs in.
- `samples/` holds 3 real, hand-cleaned SOP files (Testing Process/STLC,
  WFH Policy, SDLC Module) used as the standing test fixtures for
  ingestion work until real Meldep data is connected.

## Full planned structure

See `docs/decisions/` for the folder-structure discussion and other
recorded decisions. Empty folders in this tree are intentional
scaffolding for pieces not yet built - not accidents.
