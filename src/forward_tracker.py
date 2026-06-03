from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .abnormal_returns import DEFAULT_MARKET_BENCHMARK, abnormal_return_summary
from .horizons import validate_horizon
from .signal_store import insert_signal_snapshot, list_signal_snapshots, upsert_signal_outcome


BULLISH_CLASSIFICATIONS = {"Tradeable", "Watch"}
AVOID_CLASSIFICATIONS = {"Avoid", "Avoid Chase"}


def record_signal(
    *,
    ticker: str,
    created_at,
    horizon_days: int,
    classification: str,
    score: float,
    theme: str,
    price_at_creation: float,
    benchmark: str = DEFAULT_MARKET_BENCHMARK,
    sector_proxy: str | None = None,
    rule_version: str = "rules.v1",
    db_path: Path | str,
) -> int:
    validate_horizon(horizon_days)
    return insert_signal_snapshot(
        ticker=ticker,
        created_at=pd.Timestamp(created_at).isoformat(),
        horizon_days=horizon_days,
        classification=classification,
        score=score,
        theme=theme,
        benchmark=benchmark,
        sector_proxy=sector_proxy,
        price_at_creation=price_at_creation,
        rule_version=rule_version,
        db_path=db_path,
    )


def evaluate_forward_returns(
    *,
    db_path: Path | str,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    sector_prices_by_proxy: dict[str, pd.Series] | None = None,
    as_of=None,
) -> list[dict]:
    sector_prices_by_proxy = sector_prices_by_proxy or {}
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else _latest_timestamp(benchmark_prices)
    evaluated: list[dict] = []

    for signal in list_signal_snapshots(db_path):
        created_at = pd.Timestamp(signal["created_at"])
        horizon_end = created_at + timedelta(days=int(signal["horizon_days"]))
        is_matured = as_of_ts >= horizon_end
        if not is_matured:
            upsert_signal_outcome(
                signal_id=signal["id"],
                forward_return=None,
                forward_abnormal_return=None,
                max_drawdown=None,
                max_runup=None,
                hit=None,
                evaluated_at=as_of_ts.isoformat(),
                is_matured=False,
                db_path=db_path,
            )
            continue

        ticker_prices = prices_by_ticker.get(signal["ticker"])
        if ticker_prices is None:
            raise KeyError(f"Missing price history for {signal['ticker']}")
        sector_proxy = signal.get("sector_proxy") or ""
        sector_prices = sector_prices_by_proxy.get(sector_proxy) if sector_proxy else None
        summary = abnormal_return_summary(
            ticker_prices=ticker_prices,
            benchmark_prices=benchmark_prices,
            start_date=created_at,
            end_date=horizon_end,
            sector_prices=sector_prices,
            sector_proxy=sector_proxy or None,
        )
        forward_return = summary["raw_return"]
        forward_abnormal = summary["combined_abnormal_return"]
        max_drawdown, max_runup = max_drawdown_and_runup(ticker_prices, created_at, horizon_end)
        hit = classify_hit(signal["classification"], forward_abnormal)

        upsert_signal_outcome(
            signal_id=signal["id"],
            forward_return=forward_return,
            forward_abnormal_return=forward_abnormal,
            max_drawdown=max_drawdown,
            max_runup=max_runup,
            hit=hit,
            evaluated_at=as_of_ts.isoformat(),
            is_matured=True,
            db_path=db_path,
        )
        evaluated.append({**signal, **summary, "hit": hit, "max_drawdown": max_drawdown, "max_runup": max_runup})

    return evaluated


def classify_hit(classification: str, forward_abnormal_return: float | None) -> bool | None:
    if forward_abnormal_return is None:
        return None
    if classification in BULLISH_CLASSIFICATIONS:
        return forward_abnormal_return > 0
    if classification in AVOID_CLASSIFICATIONS:
        return forward_abnormal_return <= 0
    return None


def max_drawdown_and_runup(price_series: pd.Series, start_date, end_date) -> tuple[float | None, float | None]:
    prices = pd.Series(price_series).dropna()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    window = prices.loc[(prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))]
    if window.empty:
        return None, None
    start_price = float(window.iloc[0])
    if start_price == 0:
        return None, None
    returns = window / start_price - 1
    return float(returns.min()), float(returns.max())


def _latest_timestamp(price_series: pd.Series) -> pd.Timestamp:
    prices = pd.Series(price_series).dropna()
    if prices.empty:
        raise ValueError("benchmark price history is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    return pd.Timestamp(prices.index.max())
