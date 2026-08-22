"""
Hybrid retrieval - vector (semantic) + keyword (full-text) search
combined in one Postgres query, per the architecture doc's design
(0.7 vector weight / 0.3 text weight).

Why hybrid, not vector-only: people search SOPs with exact terms
("WFH days per month", "who approves leave") that pure embedding
similarity sometimes misses, but keyword search alone misses questions
phrased differently than the source document. Combining both catches
more real questions than either alone - this was true in the original
architecture research and is why both were built into the schema from
the start (the tsv column has been sitting unused until now).

Only searches documents that are Published + Active - the servability
filter agreed on early in this project (Sir confirmed: all SOPs open
to all employees, so no per-user permission filter needed here, only
the Published/Active gate).
"""
from dataclasses import dataclass

from app.database.postgres import get_connection
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


def hybrid_search(
    query_text: str,
    limit: int = 20,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    _filter_document_title: str | None = None,
) -> list[RetrievedChunk]:
    """Embeds the query (RETRIEVAL_QUERY task type - see embedder.py's
    module docstring for why this specific task type matters), then
    runs one combined SQL query scoring every chunk by both vector
    cosine similarity and Postgres full-text rank, weighted and summed.

    Returns the top `limit` chunks, highest combined_score first.

    _filter_document_title: for TEST ISOLATION ONLY - not a real search
    feature. Real callers should never pass this; it exists so tests
    can scope a query to just their own fixture document instead of
    competing against the entire live dataset, which is fragile (a
    test's deliberately extreme score could be pushed out of an
    unfiltered LIMIT once enough real data exists - this is exactly
    what happened during development, see test_hybrid_retriever.py).
    """
    query_vector = embed_query(query_text)

    title_filter_sql = ""
    # Params must be built in the EXACT order their %s placeholders
    # appear in the final SQL text below - the title filter clause is
    # inserted right after WHERE, BEFORE the ORDER BY clause, so its
    # parameter must come before the ORDER BY params, not after.
    params = [query_vector, query_text]
    if _filter_document_title is not None:
        title_filter_sql = "AND d.title = %s"
        params.append(_filter_document_title)
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
            rows = cur.fetchall()
    finally:
        conn.close()

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
