from datetime import datetime, timedelta

from scripts.collect_news import _latest_fresh_collection
from src.news_collector import collect_ticker_news
from src.news_store import list_collection_runs, list_news_items, news_capture_coverage


def test_collector_isolates_failures_and_records_audit(tmp_path):
    db_path = tmp_path / "news.db"

    def fetcher(symbol):
        if symbol == "BAD":
            raise RuntimeError("provider unavailable")
        return [{"title": f"{symbol} headline", "providerPublishTime": 1_789_646_400}]

    result = collect_ticker_news(
        [" nvda ", "NVDA", "bad"],
        fetcher=fetcher,
        observed_at="2026-09-17T16:00:00Z",
        db_path=db_path,
    )

    assert result["requested_tickers"] == ["NVDA", "BAD"]
    assert result["successful_tickers"] == 1
    assert result["failed_tickers"] == 1
    assert result["inserted_versions"] == 1
    assert "RuntimeError" in result["errors"]["BAD"]
    assert len(list_news_items(db_path)) == 1

    runs = list_collection_runs(db_path)
    assert len(runs) == 1
    assert runs[0]["requested_tickers"] == 2
    assert runs[0]["errors"] == {"BAD": "RuntimeError: provider unavailable"}


def test_collector_repeated_run_is_idempotent_but_audited(tmp_path):
    db_path = tmp_path / "news.db"
    fetcher = lambda symbol: [{"title": f"{symbol} headline"}]

    first = collect_ticker_news(
        ["AAPL"], fetcher=fetcher, observed_at="2026-09-17T16:00:00Z", db_path=db_path
    )
    second = collect_ticker_news(
        ["AAPL"], fetcher=fetcher, observed_at="2026-09-18T16:00:00Z", db_path=db_path
    )

    assert first["inserted_versions"] == 1
    assert second["inserted_versions"] == 0
    assert second["existing_versions"] == 1
    assert len(list_collection_runs(db_path)) == 2


def test_fresh_collection_gate_skips_only_recent_healthy_run(tmp_path):
    db_path = tmp_path / "news.db"
    collect_ticker_news(
        ["AAPL"],
        fetcher=lambda symbol: [{"title": f"{symbol} headline"}],
        observed_at="2026-09-17T16:00:00Z",
        db_path=db_path,
    )
    completed_at = datetime.fromisoformat(
        list_collection_runs(db_path)[0]["completed_at"]
    )

    fresh = _latest_fresh_collection(
        ("AAPL",),
        db_path=db_path,
        min_interval_minutes=90,
        now=completed_at + timedelta(minutes=60),
    )
    stale = _latest_fresh_collection(
        ("AAPL",),
        db_path=db_path,
        min_interval_minutes=90,
        now=completed_at + timedelta(minutes=120),
    )

    assert fresh is not None
    assert fresh["age_minutes"] == 60.0
    assert stale is None


def test_fresh_collection_gate_retries_after_failure(tmp_path):
    db_path = tmp_path / "news.db"
    collect_ticker_news(
        ["AAPL"],
        fetcher=lambda symbol: (_ for _ in ()).throw(RuntimeError("down")),
        observed_at="2026-09-17T16:00:00Z",
        db_path=db_path,
    )

    assert (
        _latest_fresh_collection(
            ("AAPL",),
            db_path=db_path,
            min_interval_minutes=90,
        )
        is None
    )


def test_capture_coverage_counts_distinct_observed_dates(tmp_path):
    db_path = tmp_path / "news.db"
    collect_ticker_news(
        ["NVDA"],
        fetcher=lambda symbol: [{"title": "One"}, {"title": "Two"}],
        observed_at="2026-09-17T16:00:00Z",
        db_path=db_path,
    )
    collect_ticker_news(
        ["NVDA"],
        fetcher=lambda symbol: [{"title": "Three"}],
        observed_at="2026-09-18T16:00:00Z",
        db_path=db_path,
    )

    coverage = news_capture_coverage(db_path)
    assert coverage == [
        {
            "ticker": "NVDA",
            "headline_versions": 3,
            "observed_dates": 2,
            "first_seen_at": "2026-09-17T16:00:00+00:00",
            "last_seen_at": "2026-09-18T16:00:00+00:00",
        }
    ]
