"""Repeatable, auditable headline collection for News Signal Lab."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .market_news import (
    NewsFetchError, NewsFetchResult, fetch_ticker_news_result, normalize_news,
)
from .news_store import DB_PATH, ingest_news_items, record_collection_run, utc_now


def collect_ticker_news(
    tickers: Iterable[str],
    *,
    fetcher: Callable[[str], list | NewsFetchResult] = fetch_ticker_news_result,
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
            fetched = fetcher(symbol)
            raw_news = fetched.items if isinstance(fetched, NewsFetchResult) else fetched
            method = fetched.method if isinstance(fetched, NewsFetchResult) else "injected"
            reason = fetched.fallback_reason if isinstance(fetched, NewsFetchResult) else None
            per_ticker[symbol] = {"fetch_method": method, "fallback_reason": reason}
            if not isinstance(raw_news, (list, tuple)):
                raise NewsFetchError("provider returned a non-list news payload")
            normalized = normalize_news(raw_news)
            if not normalized:
                raise NewsFetchError("provider returned no headlines; collection is unverified")
            # Non-text and blank titles cannot supply research features.
            for item in normalized:
                if not isinstance(item.get("title"), str):
                    item["title"] = None
            first_seen_at = observed_at or utc_now()
            result = ingest_news_items(
                ticker=symbol,
                items=normalized,
                observed_at=first_seen_at,
                provider=provider,
                db_path=db_path,
            )
            per_ticker[symbol].update(
                {key: value for key, value in result.items() if key != "ids"}
            )
            per_ticker[symbol]["fetched_items"] = len(normalized)
            totals["fetched_items"] += len(normalized)
            totals["inserted_versions"] += int(result["inserted"])
            totals["existing_versions"] += int(result["existing"])
            totals["skipped_items"] += int(result["skipped"])
            if int(result["inserted"]) + int(result["existing"]) == 0:
                raise NewsFetchError("all returned headlines were unusable")
        except Exception as exc:  # isolate one provider/ticker failure from the batch
            errors[symbol] = f"{type(exc).__name__}: {exc}"
            per_ticker.setdefault(symbol, {})["status"] = "failed"
        else:
            per_ticker[symbol]["status"] = "ok"

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
        "status": "collection_incomplete" if errors else "ok",
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
