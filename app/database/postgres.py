"""
Postgres connection handling. Reads connection details from environment
variables (.env), same pattern as app/llm/gemini.py reading GEMINI_API_KEY.

Kept separate from repositories/ - this file only knows how to get a
connection; it doesn't know about documents, chunks, or facts.
"""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Returns a new psycopg2 connection. Caller is responsible for
    closing it (or use it as a context manager)."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "meldep_sop_dev"),
        user=os.environ.get("POSTGRES_USER", "meldep_dev"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
