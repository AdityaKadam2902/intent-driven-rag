"""
Hybrid retrieval - vector (semantic) + keyword (full-text) search
combined in one Postgres query, per the architecture doc's design
(0.7 vector weight / 0.3 text weight).

Postgres circuit breaker: the actual query execution (connect + query
+ fetch) is wrapped through the shared DB breaker in
app/database/postgres.py. If Postgres is down, hybrid_search() raises
CircuitOpenError rather than a raw psycopg2 exception - callers (the
retrieve graph node) are responsible for catching this and returning
a clear, honest "temporarily unavailable" message, not a raw DB error
surfacing in the UI. This mirrors the Gemini breaker pattern, using a
SEPARATE breaker instance (see app/database/postgres.py's docstring
for why sharing one breaker between two independent dependencies would
be wrong).
"""
from dataclasses import dataclass

from app.database.postgres import get_connection, with_db_breaker
from app.ingestion.embeddings.embedder import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    document_version: str
    heading_path: str
    content: str
    category: str | None
    sub_category: str | None
    people: list[str]
    roles: list[str]
    fine_grained_topics: list[str]
    vector_score: float
    text_score: float
    combined_score: float


def _run_query(query_vector, query_text, vector_weight, text_weight, limit, filter_document_title):
    """The actual DB work - separated into its own function so it can
    be passed to with_db_breaker() as a single callable."""
    title_filter_sql = ""
    params = [query_vector, query_text]
    if filter_document_title is not None:
        title_filter_sql = "AND d.title = %s"
        params.append(filter_document_title)
    params.extend([vector_weight, query_vector, text_weight, query_text, limit])

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    c.id,
                    c.document_id,
                    d.title,
                    c.document_version,
                    c.heading_path,
                    c.content,
                    c.category,
                    c.sub_category,
                    c.people,
                    c.roles,
                    c.fine_grained_topics,
                    (1 - (c.embedding <=> %s::vector)) AS vector_score,
                    COALESCE(ts_rank(c.tsv, websearch_to_tsquery('english', %s)), 0) AS text_score
                FROM sop_chunks c
                JOIN sop_documents d ON d.id = c.document_id
                WHERE d.status = 'Published' AND d.is_active = 'Active'
                {title_filter_sql}
                ORDER BY (
                    %s * (1 - (c.embedding <=> %s::vector))
                    + %s * COALESCE(ts_rank(c.tsv, websearch_to_tsquery('english', %s)), 0)
                ) DESC
                LIMIT %s
                """,
                tuple(params),
            )
            return cur.fetchall()
    finally:
        conn.close()


def hybrid_search(
    query_text: str,
    limit: int = 20,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    _filter_document_title: str | None = None,
) -> list[RetrievedChunk]:
    """Embeds the query, then runs the combined SQL query through the
    shared Postgres circuit breaker. Raises CircuitOpenError if the
    breaker is open - see module docstring.

    Returns the top `limit` chunks, highest combined_score first.
    """
    query_vector = embed_query(query_text)

    rows = with_db_breaker(
        _run_query, query_vector, query_text, vector_weight, text_weight,
        limit, _filter_document_title,
    )

    results = []
    for row in rows:
        (chunk_id, document_id, title, version, heading_path, content,
         category, sub_category, people, roles, topics,
         vector_score, text_score) = row
        combined = vector_weight * vector_score + text_weight * text_score
        results.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                document_id=str(document_id),
                document_title=title,
                document_version=version,
                heading_path=heading_path,
                content=content,
                category=category,
                sub_category=sub_category,
                people=people or [],
                roles=roles or [],
                fine_grained_topics=topics or [],
                vector_score=float(vector_score),
                text_score=float(text_score),
                combined_score=float(combined),
            )
        )
    return results
