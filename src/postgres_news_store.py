"""PostgreSQL implementation of the durable News Signal Lab store.

The module is loaded lazily only when ``THESISBOARD_DATABASE_URL`` is set, so
local SQLite use and offline tests do not require a running PostgreSQL server.
"""

from __future__ import annotations

from contextlib import contextmanager
import json

from .news_store import (
    RESEARCH_EVENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _canonical_identity_url,
    _clean_text,
    _content_hash,
    _item_key,
    _optional_utc_iso,
    _required_utc_iso,
    utc_now,
)


@contextmanager
def connect(database_url: str):
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_news_store(database_url: str) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS news_items (
            id BIGSERIAL PRIMARY KEY,
            item_key TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            ticker TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_item_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL,
            publisher TEXT NOT NULL DEFAULT '',
            canonical_url TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            first_seen_at TEXT NOT NULL,
            availability_basis TEXT NOT NULL DEFAULT 'observed_at',
            schema_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE (ticker, item_key, content_hash)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_news_items_first_seen ON news_items(first_seen_at, id)",
        "CREATE INDEX IF NOT EXISTS idx_news_items_ticker_seen ON news_items(ticker, first_seen_at, id)",
        """
        CREATE TABLE IF NOT EXISTS news_collection_runs (
            id BIGSERIAL PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            requested_tickers INTEGER NOT NULL,
            successful_tickers INTEGER NOT NULL,
            failed_tickers INTEGER NOT NULL,
            fetched_items INTEGER NOT NULL,
            inserted_versions INTEGER NOT NULL,
            existing_versions INTEGER NOT NULL,
            skipped_items INTEGER NOT NULL,
            error_json TEXT NOT NULL,
            schema_version TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_news_collection_runs_completed
        ON news_collection_runs(completed_at, id)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_events (
            id BIGSERIAL PRIMARY KEY,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            run_id TEXT NOT NULL,
            subject_id TEXT NOT NULL DEFAULT '',
            promotion_candidate_event_id BIGINT,
            payload_json TEXT NOT NULL,
            schema_version TEXT NOT NULL
        )
        """,
        """
        ALTER TABLE research_events
        ADD COLUMN IF NOT EXISTS promotion_candidate_event_id BIGINT
        """,
        "CREATE INDEX IF NOT EXISTS idx_research_events_run ON research_events(run_id, id)",
        """
        CREATE INDEX IF NOT EXISTS idx_research_events_type_time
        ON research_events(event_type, event_time, id)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_research_events_promotion_candidate
        ON research_events(promotion_candidate_event_id)
        WHERE promotion_candidate_event_id IS NOT NULL
        """,
    ]
    with connect(database_url) as conn:
        for statement in statements:
            conn.execute(statement)


def ingest_news_items(
    *,
    database_url: str,
    ticker: str,
    items: list[dict],
    observed_at=None,
    provider: str = "yfinance",
) -> dict:
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker is required")
    source = str(provider).strip().lower()
    if not source:
        raise ValueError("provider is required")
    common_seen = _required_utc_iso(observed_at or utc_now(), field="observed_at")
    init_news_store(database_url)
    inserted = 0
    existing = 0
    skipped = 0
    ids: list[int] = []

    with connect(database_url) as conn:
        for raw in items or []:
            if not isinstance(raw, dict):
                skipped += 1
                continue
            title = _clean_text(raw.get("title"))
            if not title:
                skipped += 1
                continue
            first_seen = _required_utc_iso(
                raw.get("first_seen_at") or common_seen,
                field="first_seen_at",
            )
            published_at = _optional_utc_iso(raw.get("published_at") or raw.get("timestamp"))
            publisher = _clean_text(raw.get("publisher"))
            canonical_url = _canonical_identity_url(
                _clean_text(raw.get("canonical_url") or raw.get("link"))
            )
            provider_item_id = _clean_text(raw.get("provider_item_id"))
            item_key = _item_key(
                ticker=symbol,
                provider=source,
                provider_item_id=provider_item_id,
                canonical_url=canonical_url,
                published_at=published_at,
                title=title,
            )
            content_hash = _content_hash(
                title=title,
                publisher=publisher,
                canonical_url=canonical_url,
                published_at=published_at,
            )
            payload = json.dumps(
                {
                    "title": title,
                    "publisher": publisher or None,
                    "canonical_url": canonical_url or None,
                    "published_at": published_at,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            row = conn.execute(
                """
                INSERT INTO news_items
                (item_key, content_hash, ticker, provider, provider_item_id,
                 title, publisher, canonical_url, published_at, first_seen_at,
                 availability_basis, schema_version, payload_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'observed_at', %s, %s)
                ON CONFLICT (ticker, item_key, content_hash) DO NOTHING
                RETURNING id
                """,
                (
                    item_key,
                    content_hash,
                    symbol,
                    source,
                    provider_item_id,
                    title,
                    publisher,
                    canonical_url,
                    published_at,
                    first_seen,
                    SCHEMA_VERSION,
                    payload,
                ),
            ).fetchone()
            if row is not None:
                inserted += 1
                ids.append(int(row["id"]))
                continue
            existing += 1
            row = conn.execute(
                """
                SELECT id FROM news_items
                WHERE ticker = %s AND item_key = %s AND content_hash = %s
                """,
                (symbol, item_key, content_hash),
            ).fetchone()
            if row is not None:
                ids.append(int(row["id"]))
    return {
        "inserted": inserted,
        "existing": existing,
        "skipped": skipped,
        "ids": ids,
        "observed_at": common_seen,
    }


def list_news_items(
    *,
    database_url: str,
    ticker: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    init_news_store(database_url)
    clauses = []
    params: list[object] = []
    if ticker:
        clauses.append("ticker = %s")
        params.append(str(ticker).strip().upper())
    if limit is not None and int(limit) <= 0:
        return []
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT %s"
        params.append(int(limit))
    with connect(database_url) as conn:
        rows = conn.execute(
            f"""
            SELECT id, item_key, content_hash, ticker, provider,
                   provider_item_id, title, publisher, canonical_url,
                   published_at, first_seen_at, availability_basis,
                   schema_version
            FROM news_items
            {where}
            ORDER BY first_seen_at DESC, id DESC
            {limit_sql}
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def news_store_summary(database_url: str) -> dict:
    init_news_store(database_url)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS item_count,
                   COUNT(DISTINCT ticker) AS ticker_count,
                   MIN(first_seen_at) AS first_seen_at,
                   MAX(first_seen_at) AS last_seen_at
            FROM news_items
            """
        ).fetchone()
    return dict(row)


def record_collection_run(
    *,
    database_url: str,
    started_at,
    completed_at,
    provider: str,
    requested_tickers: int,
    successful_tickers: int,
    failed_tickers: int,
    fetched_items: int,
    inserted_versions: int,
    existing_versions: int,
    skipped_items: int,
    errors: dict[str, str] | None = None,
) -> int:
    counts = {
        "requested_tickers": requested_tickers,
        "successful_tickers": successful_tickers,
        "failed_tickers": failed_tickers,
        "fetched_items": fetched_items,
        "inserted_versions": inserted_versions,
        "existing_versions": existing_versions,
        "skipped_items": skipped_items,
    }
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("collection counts must be non-negative")
    if int(successful_tickers) + int(failed_tickers) != int(requested_tickers):
        raise ValueError("successful_tickers plus failed_tickers must equal requested_tickers")
    source = str(provider).strip().lower()
    if not source:
        raise ValueError("provider is required")
    init_news_store(database_url)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO news_collection_runs
            (started_at, completed_at, provider, requested_tickers,
             successful_tickers, failed_tickers, fetched_items,
             inserted_versions, existing_versions, skipped_items,
             error_json, schema_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                _required_utc_iso(started_at, field="started_at"),
                _required_utc_iso(completed_at, field="completed_at"),
                source,
                *(int(counts[name]) for name in counts),
                json.dumps(errors or {}, ensure_ascii=False, sort_keys=True),
                SCHEMA_VERSION,
            ),
        ).fetchone()
    return int(row["id"])


def list_collection_runs(*, database_url: str, limit: int = 20) -> list[dict]:
    init_news_store(database_url)
    if int(limit) <= 0:
        return []
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, completed_at, provider, requested_tickers,
                   successful_tickers, failed_tickers, fetched_items,
                   inserted_versions, existing_versions, skipped_items,
                   error_json, schema_version
            FROM news_collection_runs
            ORDER BY completed_at DESC, id DESC
            LIMIT %s
            """,
            (int(limit),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["errors"] = json.loads(item.pop("error_json"))
        result.append(item)
    return result


def news_capture_coverage(database_url: str) -> list[dict]:
    init_news_store(database_url)
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT ticker,
                   COUNT(*) AS headline_versions,
                   COUNT(DISTINCT LEFT(first_seen_at, 10)) AS observed_dates,
                   MIN(first_seen_at) AS first_seen_at,
                   MAX(first_seen_at) AS last_seen_at
            FROM news_items
            GROUP BY ticker
            ORDER BY observed_dates DESC, headline_versions DESC, ticker ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def append_research_event(
    *,
    database_url: str,
    event_type: str,
    run_id: str,
    subject_id: str,
    promotion_candidate_event_id: int | None,
    event_time: str,
    payload_json: str,
) -> int:
    init_news_store(database_url)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO research_events
            (event_time, event_type, run_id, subject_id,
             promotion_candidate_event_id, payload_json, schema_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                event_time,
                event_type,
                run_id,
                subject_id,
                promotion_candidate_event_id,
                payload_json,
                RESEARCH_EVENT_SCHEMA_VERSION,
            ),
        ).fetchone()
    return int(row["id"])


def list_research_events(
    *,
    database_url: str,
    event_type: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    init_news_store(database_url)
    if int(limit) <= 0:
        return []
    clauses = []
    params: list[object] = []
    if event_type:
        clauses.append("event_type = %s")
        params.append(str(event_type).strip().lower())
    if run_id:
        clauses.append("run_id = %s")
        params.append(str(run_id).strip())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(int(limit))
    with connect(database_url) as conn:
        rows = conn.execute(
            f"""
            SELECT id, event_time, event_type, run_id, subject_id,
                   promotion_candidate_event_id,
                   payload_json, schema_version
            FROM research_events
            {where}
            ORDER BY event_time DESC, id DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
    return [_research_event_dict(row) for row in rows]


def get_research_event(database_url: str, event_id: int) -> dict | None:
    init_news_store(database_url)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT id, event_time, event_type, run_id, subject_id,
                   promotion_candidate_event_id,
                   payload_json, schema_version
            FROM research_events
            WHERE id = %s
            """,
            (int(event_id),),
        ).fetchone()
    return None if row is None else _research_event_dict(row)


def _research_event_dict(row) -> dict:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item
