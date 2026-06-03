from __future__ import annotations

import pandas as pd

DEFAULT_MARKET_BENCHMARK = "SPY"

THEME_PROXIES = {
    "AI Infrastructure": "SMH",
    "Semiconductor": "SMH",
    "Airlines": "JETS",
    "Travel Rebound": "JETS",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Utilities": "XLU",
    "Power": "XLU",
    "Enterprise Software": "IGV",
    "Cybersecurity": "HACK",
}


def raw_return(price_series: pd.Series, start_date, end_date) -> float | None:
    prices = _clean_prices(price_series)
    if prices.empty:
        return None
    start_price = _price_on_or_after(prices, start_date)
    end_price = _price_on_or_before(prices, end_date)
    if start_price is None or end_price is None or start_price == 0:
        return None
    return float(end_price / start_price - 1)


def rolling_beta(ticker_returns: pd.Series, benchmark_returns: pd.Series, lookback_days: int = 120) -> float | None:
    joined = pd.concat([ticker_returns, benchmark_returns], axis=1, join="inner").dropna()
    min_points = min(lookback_days, 30)
    if len(joined) < min_points:
        return None
    joined = joined.tail(lookback_days)
    benchmark = joined.iloc[:, 1]
    variance = benchmark.var()
    if variance == 0 or pd.isna(variance):
        return None
    return float(joined.iloc[:, 0].cov(benchmark) / variance)


def beta_adjusted_abnormal_return(ticker_return: float, beta: float, market_return: float) -> float:
    return float(ticker_return - beta * market_return)


def sector_adjusted_abnormal_return(ticker_return: float, sector_return: float) -> float:
    return float(ticker_return - sector_return)


def combined_abnormal_return(
    ticker_return: float,
    beta: float,
    market_return: float,
    sector_return: float | None = None,
    sector_weight: float = 0.5,
) -> float:
    if sector_return is None:
        return beta_adjusted_abnormal_return(ticker_return, beta, market_return)
    market_weight = 1 - sector_weight
    return float(ticker_return - market_weight * beta * market_return - sector_weight * sector_return)


def abnormal_return_summary(
    ticker_prices: pd.Series,
    benchmark_prices: pd.Series,
    start_date,
    end_date,
    sector_prices: pd.Series | None = None,
    sector_proxy: str | None = None,
    lookback_days: int = 120,
) -> dict:
    ticker_ret = raw_return(ticker_prices, start_date, end_date)
    market_ret = raw_return(benchmark_prices, start_date, end_date)
    if ticker_ret is None:
        return _summary(None, market_ret, None, None, sector_proxy, None, None, "insufficient_price_history")
    if market_ret is None:
        return _summary(ticker_ret, None, None, None, sector_proxy, None, None, "missing_benchmark_data")

    ticker_returns = _clean_prices(ticker_prices).pct_change()
    benchmark_returns = _clean_prices(benchmark_prices).pct_change()
    beta = rolling_beta(ticker_returns, benchmark_returns, lookback_days) or 1.0
    beta_adjusted = beta_adjusted_abnormal_return(ticker_ret, beta, market_ret)

    sector_ret = raw_return(sector_prices, start_date, end_date) if sector_prices is not None else None
    combined = combined_abnormal_return(ticker_ret, beta, market_ret, sector_ret)
    flag = "ok" if sector_proxy is None or sector_ret is not None else "missing_sector_proxy"
    return _summary(ticker_ret, market_ret, beta, beta_adjusted, sector_proxy, sector_ret, combined, flag)


def proxy_for_theme(theme: str | None) -> str | None:
    if not theme:
        return None
    lowered = theme.lower()
    for name, proxy in THEME_PROXIES.items():
        if name.lower() in lowered:
            return proxy
    return None


def _summary(raw, market, beta, beta_adj, sector_proxy, sector_ret, combined, quality) -> dict:
    return {
        "raw_return": raw,
        "market_return": market,
        "beta": beta,
        "beta_adjusted_abnormal_return": beta_adj,
        "sector_proxy": sector_proxy,
        "sector_adjusted_abnormal_return": None if raw is None or sector_ret is None else sector_adjusted_abnormal_return(raw, sector_ret),
        "sector_return": sector_ret,
        "combined_abnormal_return": combined,
        "data_quality_flag": quality,
    }


def _clean_prices(price_series: pd.Series | None) -> pd.Series:
    if price_series is None:
        return pd.Series(dtype=float)
    prices = pd.Series(price_series).dropna()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def _normalize_timestamp(timestamp: pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    if isinstance(index, pd.DatetimeIndex) and index.tz is None and timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None and timestamp.tzinfo is None:
        return timestamp.tz_localize(index.tz)
    return timestamp


def _price_on_or_after(prices: pd.Series, value) -> float | None:
    timestamp = _normalize_timestamp(pd.Timestamp(value), prices.index)
    candidates = prices.loc[prices.index >= timestamp]
    return None if candidates.empty else float(candidates.iloc[0])


def _price_on_or_before(prices: pd.Series, value) -> float | None:
    timestamp = _normalize_timestamp(pd.Timestamp(value), prices.index)
    candidates = prices.loc[prices.index <= timestamp]
    return None if candidates.empty else float(candidates.iloc[-1])
