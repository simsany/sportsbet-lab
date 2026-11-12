from __future__ import annotations

import os
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def _dsn() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            "DATABASE_URL is missing. Attach Heroku Postgres before starting the app."
        )
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    parsed = urlparse(value)
    if parsed.scheme not in {"postgresql", "postgresql+psycopg2"}:
        raise RuntimeError("DATABASE_URL must point to PostgreSQL")
    query = parsed.query
    if "sslmode=" not in query:
        query = query + ("&" if query else "") + "sslmode=require"
    return urlunparse(parsed._replace(
        scheme="postgresql", query=query
    ))


@contextmanager
def _connection():
    conn = psycopg2.connect(_dsn(), connect_timeout=10)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize() -> None:
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prophitbet_recovery (
                record_type TEXT NOT NULL CHECK (record_type IN ('league', 'model')),
                record_id TEXT NOT NULL,
                parent_id TEXT,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (record_type, record_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS prophitbet_recovery_parent_idx
            ON prophitbet_recovery (parent_id, record_type)
        """)


def save_record(record_type: str, record_id: str, payload: dict,
                parent_id: str | None = None) -> None:
    if record_type not in {"league", "model"}:
        raise ValueError("Unsupported recovery record type")
    initialize()
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO prophitbet_recovery
                (record_type, record_id, parent_id, payload, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (record_type, record_id)
            DO UPDATE SET parent_id = EXCLUDED.parent_id,
                          payload = EXCLUDED.payload,
                          updated_at = NOW()
        """, (record_type, record_id, parent_id, Json(payload)))


def load_records(record_type: str) -> list[dict]:
    if record_type not in {"league", "model"}:
        raise ValueError("Unsupported recovery record type")
    initialize()
    with _connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT record_id, parent_id, payload
            FROM prophitbet_recovery
            WHERE record_type = %s
            ORDER BY updated_at, record_id
        """, (record_type,))
        return [dict(row) for row in cur.fetchall()]


def delete_record(record_type: str, record_id: str) -> None:
    if record_type not in {"league", "model"}:
        raise ValueError("Unsupported recovery record type")
    initialize()
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM prophitbet_recovery WHERE record_type=%s AND record_id=%s",
            (record_type, record_id),
        )
