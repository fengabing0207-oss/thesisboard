import sqlite3

from src.signal_store import (
    DEPRECATED_HIT_FIELD_NOTE,
    init_signal_store,
    insert_signal_snapshot,
    list_matured_signal_records,
    list_signal_snapshots,
    upsert_signal_outcome,
)


def test_insert_signal_snapshot(tmp_path):
    db_path = tmp_path / "signals.db"

    signal_id = insert_signal_snapshot(
        ticker="nvda",
        created_at="2026-01-02T00:00:00",
        horizon_days=5,
        classification="Tradeable",
        score=82,
        theme="AI Infrastructure",
        benchmark="SPY",
        sector_proxy="SMH",
        price_at_creation=100,
        rule_version="rules.v1",
        db_path=db_path,
    )

    snapshot = list_signal_snapshots(db_path)[0]
    assert signal_id == snapshot["id"]
    assert snapshot["ticker"] == "NVDA"
    assert snapshot["sector_proxy"] == "SMH"


def test_upsert_outcome_and_boolean_conversion(tmp_path):
    db_path = tmp_path / "signals.db"
    signal_id = insert_signal_snapshot(
        ticker="UAL",
        created_at="2026-01-02T00:00:00",
        horizon_days=5,
        classification="Watch",
        score=64,
        theme="Airlines",
        benchmark="SPY",
        sector_proxy="JETS",
        price_at_creation=50,
        rule_version="rules.v1",
        db_path=db_path,
    )

    upsert_signal_outcome(
        signal_id=signal_id,
        forward_return=0.04,
        forward_abnormal_return=0.03,
        max_drawdown=-0.01,
        max_runup=0.05,
        hit=None,
        trade_hit=None,
        watch_followthrough=True,
        avoided_bad_trade=None,
        false_negative=False,
        evaluated_at="2026-01-09T00:00:00",
        is_matured=True,
        db_path=db_path,
    )

    outcome = list_matured_signal_records(db_path)[0]
    assert outcome["watch_followthrough"] is True
    assert outcome["false_negative"] is False
    assert outcome["trade_hit"] is None


def test_init_signal_store_migrates_legacy_outcome_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE signal_snapshots (
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

            CREATE TABLE signal_outcomes (
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

    init_signal_store(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)")}
    assert {"trade_hit", "watch_followthrough", "avoided_bad_trade", "false_negative"}.issubset(columns)


def test_hit_field_is_documented_as_deprecated():
    assert "deprecated" in DEPRECATED_HIT_FIELD_NOTE
