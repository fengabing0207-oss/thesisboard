import pytest

from src.news_store import ingest_news_items, list_news_items, news_store_summary


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
