"""Append-only point-in-time storage for News Signal Lab.

The research clock is ``first_seen_at``: when ThesisBoard actually observed a
headline. Vendor publication timestamps are retained as metadata, but they are
never used as a substitute for observation time. Existing rows are not updated;
if a provider changes a headline, the changed content is stored as a new
immutable version under the same article key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news_signal_lab.db"
SCHEMA_VERSION = "news-store.v1"
RESEARCH_EVENT_SCHEMA_VERSION = "research-events.v1"
DATABASE_URL_ENV = "THESISBOARD_DATABASE_URL"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_news_store(db_path: Path | str = DB_PATH) -> None:
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        postgres_news_store.init_news_store(database_url)
        return
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            );

            CREATE INDEX IF NOT EXISTS idx_news_items_first_seen
                ON news_items(first_seen_at, id);
            CREATE INDEX IF NOT EXISTS idx_news_items_ticker_seen
                ON news_items(ticker, first_seen_at, id);

            CREATE TABLE IF NOT EXISTS news_collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            );

            CREATE INDEX IF NOT EXISTS idx_news_collection_runs_completed
                ON news_collection_runs(completed_at, id);

            CREATE TABLE IF NOT EXISTS research_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                run_id TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                promotion_candidate_event_id INTEGER,
                payload_json TEXT NOT NULL,
                schema_version TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_research_events_run
                ON research_events(run_id, id);
            CREATE INDEX IF NOT EXISTS idx_research_events_type_time
                ON research_events(event_type, event_time, id);
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(research_events)").fetchall()
        }
        if "promotion_candidate_event_id" not in columns:
            conn.execute(
                "ALTER TABLE research_events "
                "ADD COLUMN promotion_candidate_event_id INTEGER"
            )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_research_events_promotion_candidate
            ON research_events(promotion_candidate_event_id)
            WHERE promotion_candidate_event_id IS NOT NULL
            """
        )


def ingest_news_items(
    *,
    ticker: str,
    items: list[dict],
    observed_at=None,
    provider: str = "yfinance",
    db_path: Path | str = DB_PATH,
) -> dict:
    """Store normalized headlines without mutating prior observations.

    ``items`` uses the normalized Market News shape (``title``, ``publisher``,
    ``link``, ``timestamp``). A caller may additionally supply
    ``provider_item_id`` and a per-row ``first_seen_at`` for a verified import.
    Title-less entries are rejected because they cannot contribute text
    features. Re-ingesting identical content is idempotent.
    """

    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.ingest_news_items(
            database_url=database_url,
            ticker=ticker,
            items=items,
            observed_at=observed_at,
            provider=provider,
        )

    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker is required")
    source = str(provider).strip().lower()
    if not source:
        raise ValueError("provider is required")

    common_seen = _required_utc_iso(observed_at or utc_now(), field="observed_at")
    init_news_store(db_path)
    inserted = 0
    existing = 0
    skipped = 0
    ids: list[int] = []

    with connect(db_path) as conn:
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
            payload = {
                "title": title,
                "publisher": publisher or None,
                "canonical_url": canonical_url or None,
                "published_at": published_at,
            }

            cur = conn.execute(
                """
                INSERT OR IGNORE INTO news_items
                (item_key, content_hash, ticker, provider, provider_item_id,
                 title, publisher, canonical_url, published_at, first_seen_at,
                 availability_basis, schema_version, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'observed_at', ?, ?)
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
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            if cur.rowcount == 1:
                inserted += 1
                ids.append(int(cur.lastrowid))
            else:
                existing += 1
                row = conn.execute(
                    """
                    SELECT id FROM news_items
                    WHERE ticker = ? AND item_key = ? AND content_hash = ?
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
    db_path: Path | str = DB_PATH,
    *,
    ticker: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.list_news_items(
            database_url=database_url,
            ticker=ticker,
            limit=limit,
        )
    init_news_store(db_path)
    where = ""
    params: list[object] = []
    if ticker:
        where = "WHERE ticker = ?"
        params.append(str(ticker).strip().upper())
    limit_sql = ""
    if limit is not None:
        if int(limit) <= 0:
            return []
        limit_sql = "LIMIT ?"
        params.append(int(limit))
    with connect(db_path) as conn:
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


def news_store_summary(db_path: Path | str = DB_PATH) -> dict:
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.news_store_summary(database_url)
    init_news_store(db_path)
    with connect(db_path) as conn:
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
    db_path: Path | str = DB_PATH,
) -> int:
    """Append one immutable collection-attempt audit record."""

    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.record_collection_run(
            database_url=database_url,
            started_at=started_at,
            completed_at=completed_at,
            provider=provider,
            requested_tickers=requested_tickers,
            successful_tickers=successful_tickers,
            failed_tickers=failed_tickers,
            fetched_items=fetched_items,
            inserted_versions=inserted_versions,
            existing_versions=existing_versions,
            skipped_items=skipped_items,
            errors=errors,
        )

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

    init_news_store(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO news_collection_runs
            (started_at, completed_at, provider, requested_tickers,
             successful_tickers, failed_tickers, fetched_items,
             inserted_versions, existing_versions, skipped_items,
             error_json, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_utc_iso(started_at, field="started_at"),
                _required_utc_iso(completed_at, field="completed_at"),
                source,
                *(int(counts[name]) for name in counts),
                json.dumps(errors or {}, ensure_ascii=False, sort_keys=True),
                SCHEMA_VERSION,
            ),
        )
        return int(cur.lastrowid)


def list_collection_runs(
    db_path: Path | str = DB_PATH,
    *,
    limit: int = 20,
) -> list[dict]:
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.list_collection_runs(
            database_url=database_url,
            limit=limit,
        )
    init_news_store(db_path)
    if int(limit) <= 0:
        return []
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, completed_at, provider, requested_tickers,
                   successful_tickers, failed_tickers, fetched_items,
                   inserted_versions, existing_versions, skipped_items,
                   error_json, schema_version
            FROM news_collection_runs
            ORDER BY completed_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["errors"] = json.loads(item.pop("error_json"))
        result.append(item)
    return result


def news_capture_coverage(db_path: Path | str = DB_PATH) -> list[dict]:
    """Return ticker-level observation density; calendar dates are UTC dates."""

    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.news_capture_coverage(database_url)
    init_news_store(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ticker,
                   COUNT(*) AS headline_versions,
                   COUNT(DISTINCT substr(first_seen_at, 1, 10)) AS observed_dates,
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
    event_type: str,
    run_id: str,
    payload: dict,
    subject_id: str = "",
    promotion_candidate_event_id: int | None = None,
    event_time=None,
    db_path: Path | str = DB_PATH,
) -> int:
    """Append an immutable automation, candidate, or promotion event."""

    kind = _clean_text(event_type).lower()
    research_run = _clean_text(run_id)
    subject = _clean_text(subject_id)
    if not kind:
        raise ValueError("event_type is required")
    if not research_run:
        raise ValueError("run_id is required")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")
    is_promotion = kind in {
        "strategy_promotion_approved",
        "strategy_promotion_rejected",
    }
    if is_promotion != (promotion_candidate_event_id is not None):
        raise ValueError(
            "promotion events require promotion_candidate_event_id and other events must omit it"
        )
    candidate_id = (
        None
        if promotion_candidate_event_id is None
        else int(promotion_candidate_event_id)
    )
    if candidate_id is not None and candidate_id <= 0:
        raise ValueError("promotion_candidate_event_id must be positive")
    timestamp = _required_utc_iso(event_time or utc_now(), field="event_time")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )

    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.append_research_event(
            database_url=database_url,
            event_type=kind,
            run_id=research_run,
            subject_id=subject,
            promotion_candidate_event_id=candidate_id,
            event_time=timestamp,
            payload_json=serialized,
        )

    init_news_store(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO research_events
            (event_time, event_type, run_id, subject_id,
             promotion_candidate_event_id, payload_json, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                kind,
                research_run,
                subject,
                candidate_id,
                serialized,
                RESEARCH_EVENT_SCHEMA_VERSION,
            ),
        )
        return int(cur.lastrowid)


def list_research_events(
    db_path: Path | str = DB_PATH,
    *,
    event_type: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    if int(limit) <= 0:
        return []
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.list_research_events(
            database_url=database_url,
            event_type=event_type,
            run_id=run_id,
            limit=limit,
        )

    init_news_store(db_path)
    clauses = []
    params: list[object] = []
    if event_type:
        clauses.append("event_type = ?")
        params.append(_clean_text(event_type).lower())
    if run_id:
        clauses.append("run_id = ?")
        params.append(_clean_text(run_id))
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(int(limit))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, event_time, event_type, run_id, subject_id,
                   promotion_candidate_event_id,
                   payload_json, schema_version
            FROM research_events
            {where}
            ORDER BY event_time DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_research_event_dict(row) for row in rows]


def get_research_event(
    event_id: int,
    db_path: Path | str = DB_PATH,
) -> dict | None:
    database_url = _postgres_database_url(db_path)
    if database_url:
        from . import postgres_news_store

        return postgres_news_store.get_research_event(database_url, event_id)
    init_news_store(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, event_time, event_type, run_id, subject_id,
                   promotion_candidate_event_id,
                   payload_json, schema_version
            FROM research_events
            WHERE id = ?
            """,
            (int(event_id),),
        ).fetchone()
    return None if row is None else _research_event_dict(row)


def storage_backend_info(db_path: Path | str = DB_PATH) -> dict:
    """Describe storage without exposing a database URL or credentials."""

    database_url = _postgres_database_url(db_path)
    return {
        "backend": "postgresql" if database_url else "sqlite",
        "durable_for_scheduled_runs": bool(database_url),
        "database_url_env": DATABASE_URL_ENV,
    }


def _required_utc_iso(value, *, field: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.tz_convert("UTC").isoformat()


def _optional_utc_iso(value) -> str | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC").isoformat()


def _clean_text(value) -> str:
    return "" if value is None else " ".join(str(value).split()).strip()


def _canonical_identity_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value.strip()
    if not parts.netloc:
        return value.strip()
    path = parts.path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if not _is_tracking_query_key(key)
        )
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _is_tracking_query_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered.startswith("utm_") or lowered in {
        "tracking",
        "guccounter",
        "guce_referrer",
        "guce_referrer_sig",
    }


def _item_key(
    *,
    ticker: str,
    provider: str,
    provider_item_id: str,
    canonical_url: str,
    published_at: str | None,
    title: str,
) -> str:
    if provider_item_id:
        return f"{provider}:id:{provider_item_id}"
    identity_url = _canonical_identity_url(canonical_url)
    if identity_url:
        return f"{provider}:url:{identity_url}"
    seed = "|".join((ticker, published_at or "", title.casefold()))
    return f"{provider}:fallback:{_sha256(seed)}"


def _content_hash(
    *,
    title: str,
    publisher: str,
    canonical_url: str,
    published_at: str | None,
) -> str:
    seed = json.dumps(
        {
            "title": title,
            "publisher": publisher,
            "canonical_url": canonical_url,
            "published_at": published_at,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(seed)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _postgres_database_url(db_path: Path | str) -> str | None:
    value = str(db_path).strip()
    if value.startswith(("postgresql://", "postgres://")):
        return value
    try:
        is_default = Path(db_path).expanduser().resolve() == DB_PATH.resolve()
    except (OSError, TypeError, ValueError):
        is_default = False
    if not is_default:
        return None
    configured = os.getenv(DATABASE_URL_ENV, "").strip()
    if not configured:
        return None
    if not configured.startswith(("postgresql://", "postgres://")):
        raise ValueError(f"{DATABASE_URL_ENV} must be a PostgreSQL URL")
    return configured


def _research_event_dict(row) -> dict:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def _json_default(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        return timestamp.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
