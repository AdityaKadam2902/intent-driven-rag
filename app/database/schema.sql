-- Meldep SOP Assistant - core schema
--
-- Design decisions this schema encodes, and why:
--
-- 1. Version-tied everything (sop_documents.version, sop_chunks.document_version).
--    Confirmed with Sir: Meldep's Version number only bumps on real content
--    edits (Published docs can't be edited directly - editing forces a
--    version bump). So Version is trusted as the primary change signal,
--    Updated Date is kept as a backup/sanity-check field, not the primary.
--
-- 2. status/is_active enforced at the document level.
--    Confirmed: only Status='Published' AND Is Active='Active' documents
--    should ever be served to the chatbot. Rows are kept (not deleted)
--    when a document goes inactive/unpublished, but a WHERE clause on
--    every query enforces this - see the repository layer that will sit
--    on top of this schema.
--
-- 3. Atomic version swap via ON DELETE CASCADE + transactional upsert.
--    When a document updates, the ingestion pipeline will: write the new
--    document row + new chunks in one transaction, then delete old
--    chunks for that document_id. CASCADE handles the chunk cleanup
--    automatically if a document is ever hard-deleted; the version-swap
--    itself is handled explicitly in application code (see
--    app/ingestion/versioning/version_manager.py, not yet built) so it
--    can be atomic without deleting-then-failing-to-reinsert.
--
-- 4. Chunks store BOTH the vector (for semantic search) and category/
--    sub_category/people/roles/fine_grained_topics (for entity-filtered
--    and hybrid search) - directly matching the entity tagger's output
--    schema, no translation layer needed between what entity_tagger.py
--    produces and what gets stored.
--
-- 5. sop_facts exists and is ready, but nothing writes to it yet - fact
--    extraction is paused (see docs/decisions/001-pause-fact-extraction.md).
--    The table is included now so re-integrating fact extraction later
--    doesn't require a schema migration, just wiring up the writes.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sop_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    category        TEXT,
    sub_category    TEXT,
    version         TEXT NOT NULL,          -- Meldep's own version string, e.g. "1.0"
    status          TEXT NOT NULL,          -- must be 'Published' to be servable
    is_active       TEXT NOT NULL,          -- must be 'Active' to be servable
    created_by      TEXT,
    created_date    TEXT,
    updated_by      TEXT,
    updated_date    TEXT,                   -- backup signal, not primary - see note above
    source_ref      TEXT,                   -- file path today, could be a webhook/API id later
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One document can have many historical versions over time if we choose
-- to keep them (undecided yet - see docs/decisions/ for this later). For
-- now, uniqueness is on (title, version) so re-ingesting the same
-- version of the same doc is a detectable no-op, not a duplicate row.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sop_documents_title_version
    ON sop_documents (title, version);

CREATE INDEX IF NOT EXISTS idx_sop_documents_servable
    ON sop_documents (status, is_active);


CREATE TABLE IF NOT EXISTS sop_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID NOT NULL REFERENCES sop_documents(id) ON DELETE CASCADE,
    document_version    TEXT NOT NULL,      -- redundant with sop_documents.version,
                                             -- kept here too so a chunk is self-describing
                                             -- for cache-invalidation tagging (Part 2, Section 4.4)
    chunk_index         INT NOT NULL,
    heading_path        TEXT,               -- e.g. "Test Planning > Test Plan Preparation"
    content             TEXT NOT NULL,

    -- from entity_tagger.py's EntityTags - same field names, no translation
    category            TEXT,
    sub_category         TEXT,
    people               TEXT[],
    roles                 TEXT[],
    fine_grained_topics   TEXT[],

    -- from embedder.py - 768 dimensions, matching EMBEDDING_DIMENSIONS
    embedding            vector(768),

    -- full-text search support, generated automatically from content
    tsv                   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_sop_chunks_document_id ON sop_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_sop_chunks_tsv ON sop_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_sop_chunks_people ON sop_chunks USING GIN (people);
CREATE INDEX IF NOT EXISTS idx_sop_chunks_roles ON sop_chunks USING GIN (roles);
CREATE INDEX IF NOT EXISTS idx_sop_chunks_topics ON sop_chunks USING GIN (fine_grained_topics);

-- ivfflat needs data present with a representative sample to build well;
-- for the small SOP-scale dataset this project targets, this is fine to
-- create even on an empty table. Revisit (e.g. move to hnsw) only if
-- real scale ever approaches the low tens of millions of vectors - see
-- architecture doc Part 3, Section 8.
CREATE INDEX IF NOT EXISTS idx_sop_chunks_embedding
    ON sop_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


-- Structured facts table - schema ready, nothing writes here yet.
-- See docs/decisions/001-pause-fact-extraction.md
CREATE TABLE IF NOT EXISTS sop_facts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID NOT NULL REFERENCES sop_documents(id) ON DELETE CASCADE,
    document_version    TEXT NOT NULL,
    chunk_id             UUID REFERENCES sop_chunks(id) ON DELETE CASCADE,
    policy               TEXT NOT NULL,
    value                 TEXT NOT NULL,
    condition             TEXT,
    source_heading        TEXT
);

CREATE INDEX IF NOT EXISTS idx_sop_facts_document_id ON sop_facts (document_id);
CREATE INDEX IF NOT EXISTS idx_sop_facts_policy ON sop_facts (policy);
