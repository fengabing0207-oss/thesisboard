from __future__ import annotations

from pathlib import Path

import pandas as pd

from .abnormal_returns import DEFAULT_MARKET_BENCHMARK, abnormal_return_summary
from .horizons import validate_horizon
from .signal_store import insert_signal_snapshot, list_signal_snapshots, upsert_signal_outcome


TRADEABLE_CLASSIFICATIONS = {"Tradeable"}
NON_TRADEABLE_CLASSIFICATIONS = {"Avoid", "Avoid Chase", "Wait for Confirmation", "Watch"}


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
        ticker_prices = prices_by_ticker.get(signal["ticker"])
        if ticker_prices is None:
            raise KeyError(f"Missing price history for {signal['ticker']}")
        horizon_end = trading_session_horizon_end(
            created_at,
            int(signal["horizon_days"]),
            ticker_prices.index,
            benchmark_prices.index,
        )
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
        outcome = classify_outcome_semantics(signal["classification"], forward_abnormal)

        upsert_signal_outcome(
            signal_id=signal["id"],
            forward_return=forward_return,
            forward_abnormal_return=forward_abnormal,
            max_drawdown=max_drawdown,
            max_runup=max_runup,
            hit=outcome["trade_hit"],
            trade_hit=outcome["trade_hit"],
            avoided_bad_trade=outcome["avoided_bad_trade"],
            false_negative=outcome["false_negative"],
            evaluated_at=as_of_ts.isoformat(),
            is_matured=True,
            db_path=db_path,
        )
        evaluated.append({**signal, **summary, **outcome, "max_drawdown": max_drawdown, "max_runup": max_runup})

    return evaluated


def classify_outcome_semantics(classification: str, forward_abnormal_return: float | None) -> dict:
    outcome = {"trade_hit": None, "avoided_bad_trade": None, "false_negative": None}
    if forward_abnormal_return is None:
        return outcome
    if classification in TRADEABLE_CLASSIFICATIONS:
        outcome["trade_hit"] = forward_abnormal_return > 0
        outcome["false_negative"] = False
    elif classification in NON_TRADEABLE_CLASSIFICATIONS:
        outcome["avoided_bad_trade"] = forward_abnormal_return <= 0
        outcome["false_negative"] = forward_abnormal_return > 0
    return outcome


def trading_session_horizon_end(created_at, horizon_days: int, *indices: pd.Index) -> pd.Timestamp:
    validate_horizon(horizon_days)
    created_ts = pd.Timestamp(created_at)
    sessions = _combined_sessions(*indices)
    future_sessions = sessions[sessions > created_ts]
    if len(future_sessions) < horizon_days:
        raise ValueError(f"Not enough trading sessions after {created_ts} for {horizon_days}D horizon")
    return pd.Timestamp(future_sessions[horizon_days - 1])


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


def _combined_sessions(*indices: pd.Index) -> pd.DatetimeIndex:
    sessions: list[pd.Timestamp] = []
    for index in indices:
        values = pd.DatetimeIndex(pd.to_datetime(index)).dropna()
        sessions.extend(pd.Timestamp(value) for value in values)
    if not sessions:
        raise ValueError("at least one trading-session index is required")
    return pd.DatetimeIndex(sorted(set(sessions)))
