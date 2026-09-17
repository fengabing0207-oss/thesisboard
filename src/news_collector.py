"""Repeatable, auditable headline collection for News Signal Lab."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .market_news import fetch_ticker_news_strict, normalize_news
from .news_store import DB_PATH, ingest_news_items, record_collection_run, utc_now


def collect_ticker_news(
    tickers: Iterable[str],
    *,
    fetcher: Callable[[str], list] = fetch_ticker_news_strict,
    observed_at=None,
    provider: str = "yfinance",
    db_path=DB_PATH,
) -> dict:
    """Fetch each ticker once, ingest normalized rows, and audit the attempt."""

    symbols = list(dict.fromkeys(str(value).strip().upper() for value in tickers if str(value).strip()))
    if not symbols:
        raise ValueError("at least one ticker is required")
    started_at = utc_now()
    totals = {"fetched_items": 0, "inserted_versions": 0, "existing_versions": 0, "skipped_items": 0}
    errors: dict[str, str] = {}
    per_ticker: dict[str, dict] = {}

    for symbol in symbols:
        try:
            normalized = normalize_news(fetcher(symbol))
            first_seen_at = observed_at or utc_now()
            result = ingest_news_items(
                ticker=symbol,
                items=normalized,
                observed_at=first_seen_at,
                provider=provider,
                db_path=db_path,
            )
            per_ticker[symbol] = result
            totals["fetched_items"] += len(normalized)
            totals["inserted_versions"] += int(result["inserted"])
            totals["existing_versions"] += int(result["existing"])
            totals["skipped_items"] += int(result["skipped"])
        except Exception as exc:  # isolate one provider/ticker failure from the batch
            errors[symbol] = f"{type(exc).__name__}: {exc}"

    completed_at = utc_now()
    run_id = record_collection_run(
        started_at=started_at,
        completed_at=completed_at,
        provider=provider,
        requested_tickers=len(symbols),
        successful_tickers=len(symbols) - len(errors),
        failed_tickers=len(errors),
        errors=errors,
        db_path=db_path,
        **totals,
    )
    return {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "requested_tickers": symbols,
        "successful_tickers": len(symbols) - len(errors),
        "failed_tickers": len(errors),
        "errors": errors,
        "per_ticker": per_ticker,
        **totals,
    }
