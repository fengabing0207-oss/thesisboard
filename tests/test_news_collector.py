from datetime import datetime, timedelta
import json

import pytest

from scripts.collect_news import _latest_fresh_collection
from src.news_collector import collect_ticker_news
from src.market_news import NewsFetchResult
from src.news_store import (
    list_collection_runs, list_news_items, news_capture_coverage, record_collection_run,
)


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


@pytest.mark.parametrize("raw", [[], None, {"message": "Not Found"}, [{}],
                                [{"title": " "}], [{"title": {"bad": "shape"}}]])
def test_empty_or_unusable_feed_is_audited_as_failure(tmp_path, raw):
    db_path = tmp_path / "news.db"
    result = collect_ticker_news(["NVDA"], fetcher=lambda symbol: raw, db_path=db_path)
    assert result["status"] == "collection_incomplete"
    assert result["successful_tickers"] == 0
    assert result["failed_tickers"] == 1
    assert "NewsFetchError" in result["errors"]["NVDA"]
    assert list_collection_runs(db_path)[0]["failed_tickers"] == 1
    assert list_news_items(db_path) == []
    assert _latest_fresh_collection(("NVDA",), db_path=db_path, min_interval_minutes=90) is None


def test_partial_empty_batch_preserves_other_tickers_and_reports_failure(tmp_path):
    result = collect_ticker_news(
        ["NVDA", "MU"], db_path=tmp_path / "news.db",
        fetcher=lambda symbol: [] if symbol == "MU" else [{"title": "Good headline"}],
    )
    assert result["status"] == "collection_incomplete"
    assert result["inserted_versions"] == 1
    assert result["successful_tickers"] == result["failed_tickers"] == 1


def test_fallback_deduplicates_across_routes_and_keeps_observation_clock(tmp_path):
    db_path = tmp_path / "news.db"
    published = "2026-09-17T12:00:00Z"
    first = collect_ticker_news(
        ["NVDA"], db_path=db_path, observed_at="2026-09-17T16:00:00Z",
        fetcher=lambda symbol: [{"content": {
            "title": "Original", "canonicalUrl": {"url": "https://example.com/a"},
            "pubDate": published,
        }}],
    )
    second = collect_ticker_news(
        ["NVDA"], db_path=db_path, observed_at="2026-10-27T12:00:00Z",
        fetcher=lambda symbol: NewsFetchResult(items=[
            {"title": "Original", "link": "https://example.com/a", "providerPublishTime": published},
            {"title": "Previously unseen", "link": "https://example.com/b", "providerPublishTime": published},
        ], method="search_news", fallback_reason="ticker_news returned no usable headlines"),
    )
    assert first["inserted_versions"] == 1
    assert second["status"] == "ok"
    assert second["existing_versions"] == second["inserted_versions"] == 1
    assert second["per_ticker"]["NVDA"]["fetch_method"] == "search_news"
    seen = {item["title"]: item["first_seen_at"] for item in list_news_items(db_path)}
    assert seen == {"Original": "2026-09-17T16:00:00+00:00", "Previously unseen": "2026-10-27T12:00:00+00:00"}


def test_fresh_gate_rejects_legacy_green_empty_run(tmp_path):
    db_path = tmp_path / "news.db"
    now = datetime.now().astimezone().isoformat()
    record_collection_run(
        started_at=now, completed_at=now, provider="yfinance", requested_tickers=1, successful_tickers=1,
        failed_tickers=0, fetched_items=0, inserted_versions=0, existing_versions=0,
        skipped_items=0, errors={}, db_path=db_path,
    )
    assert _latest_fresh_collection(("NVDA",), db_path=db_path, min_interval_minutes=90) is None


@pytest.mark.parametrize("raw, expected_exit", [([], 1), ([{"title": "Headline"}], 0)])
def test_collection_cli_exit_code_follows_actual_yield(tmp_path, monkeypatch, capsys, raw, expected_exit):
    import scripts.collect_news as cli

    monkeypatch.setattr("sys.argv", ["collect_news", "NVDA", "--db", str(tmp_path / "news.db")])
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(cli, "collect_ticker_news", lambda tickers, **kwargs: collect_ticker_news(
        tickers, fetcher=lambda symbol: raw, **kwargs,
    ))
    assert cli.main() == expected_exit
    assert "per_ticker" in json.loads(capsys.readouterr().out)
