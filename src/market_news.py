"""Market News tab helpers — yfinance snapshot, news normalization, AI summary.

Market-context reference only: NOT investment advice and NOT a buy/sell signal.
No web scraping, no paid news API, no brokerage. ``yfinance`` and ``anthropic``
are imported lazily inside the fetch seams so this module (and the app) import
cleanly without those packages or any network/API key.

Design: thin fetch seams (network) are kept separate from pure normalizer/compute
functions, which are what the tests exercise with mocked data.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pandas as pd

# Broad market / regime symbols. ^VIX is the volatility index; the rest are ETFs.
MARKET_SYMBOLS = ["SPY", "QQQ", "SOXX", "^VIX"]
MARKET_LABELS = {"^VIX": "VIX"}

SUMMARY_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_SYSTEM = (
    "You summarize public news headlines at a neutral, topic level only. "
    "Given a list of headlines, produce: the main topics being discussed; "
    "positive-leaning headline themes, if any; cautionary headline themes, if any; "
    'and a factual count line such as "3 positive-leaning headlines, 2 cautionary '
    'headlines, 4 neutral/mixed headlines". '
    "This is headline-framing only, not sentiment prediction. Do NOT output an "
    "overall bullish/bearish verdict, do NOT say the stock will go up or down, do "
    "NOT say buy/sell/hold, do NOT say whether it is a good or bad time to enter, "
    "do NOT predict price direction, and do NOT give investment advice. Summarize "
    "only the provided headlines; do not use outside knowledge."
)


# --- price series helpers (pure) ------------------------------------------


def _clean_series(series) -> pd.Series | None:
    if series is None:
        return None
    cleaned = pd.Series(series).dropna()
    if cleaned.empty:
        return cleaned
    try:
        cleaned = cleaned.sort_index()
    except TypeError:
        pass
    return cleaned


def _trailing_return(series: pd.Series, sessions: int):
    if len(series) <= sessions:
        return None
    prior = float(series.iloc[-(sessions + 1)])
    if prior == 0:
        return None
    return float(series.iloc[-1]) / prior - 1


def summarize_index_series(series) -> dict:
    """Latest level + the series for a broad-market symbol. Graceful on no data."""
    cleaned = _clean_series(series)
    if cleaned is None or cleaned.empty:
        return {"available": False}
    return {"available": True, "latest": float(cleaned.iloc[-1]), "series": cleaned}


def compute_ticker_metrics(series) -> dict:
    """5D/20D/60D returns + distance-from-high for a ticker. Graceful on no data."""
    cleaned = _clean_series(series)
    if cleaned is None or cleaned.empty:
        return {"available": False}
    high = float(cleaned.max())
    return {
        "available": True,
        "latest": float(cleaned.iloc[-1]),
        "return_5d": _trailing_return(cleaned, 5),
        "return_20d": _trailing_return(cleaned, 20),
        "return_60d": _trailing_return(cleaned, 60),
        "distance_from_high": (float(cleaned.iloc[-1]) / high - 1) if high else None,
        "series": cleaned,
    }


def build_market_snapshot(prices_by_symbol: dict, *, ticker: str | None = None, market_symbols=None) -> dict:
    """Pure snapshot builder over a ``{symbol: series}`` map (already fetched)."""
    market_symbols = market_symbols or MARKET_SYMBOLS
    prices_by_symbol = prices_by_symbol or {}
    market = {symbol: summarize_index_series(prices_by_symbol.get(symbol)) for symbol in market_symbols}
    ticker_metrics = compute_ticker_metrics(prices_by_symbol.get(ticker)) if ticker else None
    return {"market": market, "ticker": ticker_metrics}


def days_until(target, *, as_of=None):
    """Whole days from today (or as_of) to a target date. None-safe."""
    if target is None:
        return None
    try:
        target_date = pd.Timestamp(target).date()
        ref = pd.Timestamp(as_of).date() if as_of is not None else date.today()
    except (ValueError, TypeError):
        return None
    return (target_date - ref).days


# --- news normalization (pure) --------------------------------------------


def _nested(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalize_timestamp(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _normalize_news_item(entry: dict) -> dict:
    content = entry.get("content") if isinstance(entry.get("content"), dict) else {}
    title = entry.get("title") or content.get("title")
    publisher = entry.get("publisher") or _nested(content, "provider", "displayName")
    link = (
        entry.get("link")
        or _nested(content, "canonicalUrl", "url")
        or _nested(content, "clickThroughUrl", "url")
    )
    timestamp = _normalize_timestamp(entry.get("providerPublishTime") or content.get("pubDate"))
    return {
        "title": title or None,
        "publisher": publisher or None,
        "link": link or None,
        "timestamp": timestamp,
    }


def normalize_news(raw_news) -> list:
    """Normalize yfinance ``ticker.news`` into a stable list of dicts.

    Handles empty input, non-list shapes, non-dict entries, the legacy flat
    shape and the newer nested ``content`` shape, and any missing field.
    """
    if not isinstance(raw_news, (list, tuple)):
        return []
    return [_normalize_news_item(entry) for entry in raw_news if isinstance(entry, dict)]


def headlines_from_news(normalized_news) -> list:
    """Just the non-empty headline titles, in order."""
    return [item["title"] for item in (normalized_news or []) if item.get("title")]


# --- AI topic summary (optional) ------------------------------------------


def anthropic_api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _summary_prompt(headlines: list) -> str:
    listed = "\n".join(f"- {headline}" for headline in headlines)
    return (
        "Summarize ONLY the following public news headlines at a topic level. "
        "Do not use any outside knowledge or price data.\n\nHeadlines:\n" + listed
    )


def summarize_headlines(headlines: list, *, client) -> str:
    """Topic-level summary of ONLY the given headlines via an Anthropic client.

    ``client`` is injected (Anthropic SDK client or a mock in tests). Returns the
    summary text.
    """
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=500,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": _summary_prompt(headlines)}],
    )
    return _extract_text(response)


def _extract_text(response) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts = [getattr(block, "text", "") for block in content if getattr(block, "text", "")]
        if parts:
            return "\n".join(parts).strip()
    return str(content if content is not None else response).strip()


# --- fetch seams (network; not exercised by tests) ------------------------


def fetch_market_prices(symbols, *, lookback_days: int = 180, provider=None, end=None) -> dict:
    """Fetch recent adjusted-close series via the existing yfinance provider."""
    from .price_provider import YFinancePriceProvider

    provider = provider or YFinancePriceProvider()
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(date.today())
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    bundle = provider.get_history(list(symbols), start_ts, end_ts)
    return dict(bundle.prices)


def fetch_ticker_news(symbol: str) -> list:
    import yfinance as yf

    try:
        news = yf.Ticker(symbol).news
    except Exception:
        return []
    return news or []


def fetch_next_earnings_date(symbol: str):
    """Best-effort next earnings date via yfinance; None on any problem."""
    import yfinance as yf

    try:
        calendar = yf.Ticker(symbol).calendar
    except Exception:
        return None
    value = None
    if isinstance(calendar, dict):
        value = calendar.get("Earnings Date")
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
    else:
        try:
            value = calendar.loc["Earnings Date"][0]
        except Exception:
            value = None
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        return None


def get_anthropic_client():
    import anthropic

    return anthropic.Anthropic()
