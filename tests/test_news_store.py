import sqlite3

import pandas as pd
import pytest

import src.news_store as news_store
from src.news_store import (
    append_research_event,
    get_research_event,
    ingest_news_items,
    init_news_store,
    list_collection_runs,
    list_news_items,
    list_research_events,
    news_store_summary,
    record_collection_run,
    storage_backend_info,
)


def test_news_store_is_idempotent_and_preserves_changed_versions(tmp_path):
    db_path = tmp_path / "news.db"
    first = ingest_news_items(
        ticker="nvda",
        observed_at="2026-09-17T12:00:00-04:00",
        items=[
            {
                "title": "Chipmaker beats estimates",
                "publisher": "Wire",
                "link": "https://example.com/story?tracking=one",
                "timestamp": "2026-09-17T14:00:00Z",
            },
            {"publisher": "Titleless"},
        ],
        db_path=db_path,
    )
    assert first["inserted"] == 1
    assert first["skipped"] == 1
    assert first["observed_at"] == "2026-09-17T16:00:00+00:00"

    repeated = ingest_news_items(
        ticker="NVDA",
        observed_at="2026-09-18T12:00:00-04:00",
        items=[
            {
                "title": "Chipmaker beats estimates",
                "publisher": "Wire",
                "link": "https://example.com/story?tracking=two",
                "timestamp": "2026-09-17T14:00:00Z",
            }
        ],
        db_path=db_path,
    )
    assert repeated["inserted"] == 0
    assert repeated["existing"] == 1

    changed = ingest_news_items(
        ticker="NVDA",
        observed_at="2026-09-18T12:00:00-04:00",
        items=[
            {
                "title": "Chipmaker sharply beats estimates",
                "publisher": "Wire",
                "link": "https://example.com/story",
                "timestamp": "2026-09-17T14:00:00Z",
            }
        ],
        db_path=db_path,
    )
    assert changed["inserted"] == 1

    rows = list_news_items(db_path)
    assert len(rows) == 2
    assert {row["item_key"] for row in rows} == {rows[0]["item_key"]}
    assert {row["first_seen_at"] for row in rows} == {
        "2026-09-17T16:00:00+00:00",
        "2026-09-18T16:00:00+00:00",
    }
    assert news_store_summary(db_path)["ticker_count"] == 1


def test_news_store_rejects_ambiguous_observation_time(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        ingest_news_items(
            ticker="NVDA",
            observed_at="2026-09-17 12:00:00",
            items=[{"title": "Headline"}],
            db_path=tmp_path / "news.db",
        )


def test_news_store_filters_ticker_and_limit(tmp_path):
    db_path = tmp_path / "news.db"
    for ticker in ("NVDA", "AAPL"):
        ingest_news_items(
            ticker=ticker,
            observed_at="2026-09-17T16:00:00Z",
            items=[{"title": f"{ticker} headline"}],
            db_path=db_path,
        )
    assert len(list_news_items(db_path, ticker="nvda")) == 1
    assert len(list_news_items(db_path, limit=1)) == 1
    assert list_news_items(db_path, limit=0) == []


def test_news_store_keeps_identity_query_parameters(tmp_path):
    db_path = tmp_path / "news.db"
    result = ingest_news_items(
        ticker="NVDA",
        observed_at="2026-09-17T16:00:00Z",
        items=[
            {"title": "First", "link": "https://example.com/story?id=one&utm_source=test"},
            {"title": "Second", "link": "https://example.com/story?id=two&utm_source=test"},
        ],
        db_path=db_path,
    )
    assert result["inserted"] == 2
    rows = list_news_items(db_path)
    assert {row["canonical_url"] for row in rows} == {
        "https://example.com/story?id=one",
        "https://example.com/story?id=two",
    }


def test_collection_run_rejects_inconsistent_counts(tmp_path):
    with pytest.raises(ValueError, match="must equal"):
        record_collection_run(
            started_at="2026-09-17T16:00:00Z",
            completed_at="2026-09-17T16:01:00Z",
            provider="yfinance",
            requested_tickers=2,
            successful_tickers=1,
            failed_tickers=0,
            fetched_items=0,
            inserted_versions=0,
            existing_versions=0,
            skipped_items=0,
            db_path=tmp_path / "news.db",
        )
    assert list_collection_runs(tmp_path / "news.db") == []


def test_research_events_are_append_only_filterable_and_json_safe(tmp_path):
    db_path = tmp_path / "news.db"
    event_id = append_research_event(
        event_type="Strategy_Candidate_Proposed",
        run_id=" run-1 ",
        subject_id="spec-1",
        event_time="2026-09-17T12:00:00-04:00",
        payload={"as_of": pd.Timestamp("2026-09-17T16:00:00Z"), "score": 0.75},
        db_path=db_path,
    )
    append_research_event(
        event_type="automation_run_completed",
        run_id="run-2",
        event_time="2026-09-17T17:00:00Z",
        payload={"status": "no_captured_headlines"},
        db_path=db_path,
    )

    event = get_research_event(event_id, db_path)
    assert event["event_type"] == "strategy_candidate_proposed"
    assert event["run_id"] == "run-1"
    assert event["payload"] == {
        "as_of": "2026-09-17T16:00:00+00:00",
        "score": 0.75,
    }
    assert event["schema_version"] == "research-events.v1"
    assert list_research_events(db_path, event_type="STRATEGY_CANDIDATE_PROPOSED") == [
        event
    ]
    assert list_research_events(db_path, run_id="run-2")[0]["payload"]["status"] == (
        "no_captured_headlines"
    )


def test_research_store_enforces_one_promotion_decision_per_candidate(tmp_path):
    db_path = tmp_path / "news.db"
    candidate_id = append_research_event(
        event_type="strategy_candidate_proposed",
        run_id="run-1",
        payload={"strategy_spec_id": "spec-1"},
        db_path=db_path,
    )
    append_research_event(
        event_type="strategy_promotion_approved",
        run_id="review-1",
        promotion_candidate_event_id=candidate_id,
        payload={"candidate_event_id": candidate_id},
        db_path=db_path,
    )

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        append_research_event(
            event_type="strategy_promotion_rejected",
            run_id="review-2",
            promotion_candidate_event_id=candidate_id,
            payload={"candidate_event_id": candidate_id},
            db_path=db_path,
        )


def test_research_event_schema_migration_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE research_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                run_id TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                schema_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO research_events
            (event_time, event_type, run_id, subject_id, payload_json, schema_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-09-17T16:00:00+00:00",
                "automation_run_completed",
                "legacy-run",
                "",
                '{"status": "legacy"}',
                "research-events.v1",
            ),
        )

    init_news_store(db_path)

    event = list_research_events(db_path, run_id="legacy-run")[0]
    assert event["payload"]["status"] == "legacy"
    assert event["promotion_candidate_event_id"] is None
    with pytest.raises(ValueError, match="require promotion_candidate_event_id"):
        append_research_event(
            event_type="strategy_promotion_approved",
            run_id="review-3",
            payload={},
            db_path=db_path,
        )


def test_store_backend_info_and_postgres_dispatch(monkeypatch, tmp_path):
    assert storage_backend_info(tmp_path / "news.db") == {
        "backend": "sqlite",
        "durable_for_scheduled_runs": False,
        "database_url_env": "THESISBOARD_DATABASE_URL",
    }

    monkeypatch.setenv("THESISBOARD_DATABASE_URL", "postgresql://db.example/test")
    monkeypatch.setattr(
        "src.postgres_news_store.news_store_summary",
        lambda database_url: {"database_url_seen": database_url},
    )
    assert news_store.news_store_summary() == {
        "database_url_seen": "postgresql://db.example/test"
    }
    assert storage_backend_info()["durable_for_scheduled_runs"] is True


def test_store_rejects_malformed_configured_database_url(monkeypatch):
    monkeypatch.setenv("THESISBOARD_DATABASE_URL", "sqlite:///not-allowed.db")
    with pytest.raises(ValueError, match="must be a PostgreSQL URL"):
        news_store.news_store_summary()
