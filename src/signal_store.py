from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | str):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_signal_store(db_path: Path | str) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS signal_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                classification TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                theme TEXT DEFAULT '',
                benchmark TEXT NOT NULL DEFAULT 'SPY',
                sector_proxy TEXT DEFAULT '',
                price_at_creation REAL NOT NULL,
                rule_version TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL UNIQUE REFERENCES signal_snapshots(id) ON DELETE CASCADE,
                forward_return REAL,
                forward_abnormal_return REAL,
                max_drawdown REAL,
                max_runup REAL,
                hit INTEGER,
                evaluated_at TEXT NOT NULL,
                is_matured INTEGER NOT NULL DEFAULT 0
            );
            """
        )


def insert_signal_snapshot(
    *,
    ticker: str,
    created_at: str,
    horizon_days: int,
    classification: str,
    score: float,
    theme: str,
    benchmark: str,
    sector_proxy: str | None,
    price_at_creation: float,
    rule_version: str,
    db_path: Path | str,
) -> int:
    init_signal_store(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO signal_snapshots
            (ticker, created_at, horizon_days, classification, score, theme, benchmark, sector_proxy, price_at_creation, rule_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.strip().upper(),
                created_at,
                int(horizon_days),
                classification,
                float(score),
                theme,
                benchmark,
                sector_proxy or "",
                float(price_at_creation),
                rule_version,
            ),
        )
        return int(cur.lastrowid)


def upsert_signal_outcome(
    *,
    signal_id: int,
    forward_return: float | None,
    forward_abnormal_return: float | None,
    max_drawdown: float | None,
    max_runup: float | None,
    hit: bool | None,
    evaluated_at: str,
    is_matured: bool,
    db_path: Path | str,
) -> None:
    init_signal_store(db_path)
    hit_value = None if hit is None else int(hit)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO signal_outcomes
            (signal_id, forward_return, forward_abnormal_return, max_drawdown, max_runup, hit, evaluated_at, is_matured)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                forward_return=excluded.forward_return,
                forward_abnormal_return=excluded.forward_abnormal_return,
                max_drawdown=excluded.max_drawdown,
                max_runup=excluded.max_runup,
                hit=excluded.hit,
                evaluated_at=excluded.evaluated_at,
                is_matured=excluded.is_matured
            """,
            (int(signal_id), forward_return, forward_abnormal_return, max_drawdown, max_runup, hit_value, evaluated_at, int(is_matured)),
        )


def list_signal_snapshots(db_path: Path | str, *, include_outcomes: bool = False) -> list[dict]:
    init_signal_store(db_path)
    with connect(db_path) as conn:
        if not include_outcomes:
            rows = conn.execute("SELECT * FROM signal_snapshots ORDER BY created_at, id").fetchall()
        else:
            rows = conn.execute(
                """
                SELECT s.*, o.forward_return, o.forward_abnormal_return, o.max_drawdown,
                       o.max_runup, o.hit, o.evaluated_at, o.is_matured
                FROM signal_snapshots s
                LEFT JOIN signal_outcomes o ON o.signal_id = s.id
                ORDER BY s.created_at, s.id
                """
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def list_matured_signal_records(db_path: Path | str) -> list[dict]:
    init_signal_store(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, o.forward_return, o.forward_abnormal_return, o.max_drawdown,
                   o.max_runup, o.hit, o.evaluated_at, o.is_matured
            FROM signal_snapshots s
            JOIN signal_outcomes o ON o.signal_id = s.id
            WHERE o.is_matured = 1
            ORDER BY s.created_at, s.id
            """
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    if "hit" in item and item["hit"] is not None:
        item["hit"] = bool(item["hit"])
    if "is_matured" in item and item["is_matured"] is not None:
        item["is_matured"] = bool(item["is_matured"])
    return item
