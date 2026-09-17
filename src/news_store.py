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
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news_signal_lab.db"
SCHEMA_VERSION = "news-store.v1"


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
