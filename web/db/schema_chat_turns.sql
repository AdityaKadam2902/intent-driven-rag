-- Run against meldep_sop_dev. Owned by the web layer, not the RAG core.
CREATE TABLE IF NOT EXISTS chat_turns (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    turn_index INT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    confidence FLOAT,
    sources JSONB,
    intention TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_turns_session_assistant
    ON chat_turns (session_id, assistant_id, turn_index);
