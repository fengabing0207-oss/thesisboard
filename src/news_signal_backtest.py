"""Stateful position-book audit for locked News Signal Lab predictions.

The backtest starts flat, holds equal weights across tickers with at least one
active selected event, and charges cost on every change in portfolio weight.
It is deliberately long-only and contains no order-routing functionality.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .news_signal_policy import evaluate_event_policy, select_and_evaluate_holdout_policy


BACKTEST_VERSION = "equal-weight-overlap-aware.v1"


def run_long_only_position_backtest(
    predictions: pd.DataFrame,
    *,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    probability_threshold: float,
    one_way_cost_bps: float,
    start_session=None,
    book_start_session=None,
    book_end_session=None,
) -> dict:
    """Audit a fixed model/threshold with daily holdings and realized turnover.

    A selected event is active from its signal close up to, but not including,
    its horizon-end close. Overlapping events for the same ticker create one
    position rather than leveraged duplicates. Active tickers are equal
    weighted; unallocated capital remains in zero-return cash.
    """

    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    if float(one_way_cost_bps) < 0.0:
        raise ValueError("one_way_cost_bps must be non-negative")
    required = {"ticker", "signal_session", "horizon_end", "predicted_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")

    frame = predictions.dropna(subset=list(required)).copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["signal_session"] = _normalize_dates(frame["signal_session"])
    frame["horizon_end"] = _normalize_dates(frame["horizon_end"])
    if start_session is not None:
        frame = frame.loc[frame["signal_session"] >= _normalize_date(start_session)]
    frame = frame.sort_values(["signal_session", "ticker"]).reset_index(drop=True)
    if frame.empty:
        return _empty_backtest("no_oos_rows")
    invalid_horizons = frame.loc[frame["horizon_end"] <= frame["signal_session"]]
    if not invalid_horizons.empty:
        raise ValueError("every horizon_end must follow its signal_session")
    if frame.duplicated(["ticker", "signal_session"]).any():
        raise ValueError("predictions contain duplicate ticker/session keys")

    selected = frame.loc[
        frame["predicted_probability"].astype(float) >= float(probability_threshold)
    ].copy()
    if selected.empty:
        result = _empty_backtest("no_selected_signals")
        result["eligible_prediction_count"] = int(len(frame))
        return result

    benchmark = _clean_prices(benchmark_prices)
    if benchmark.empty:
        raise ValueError("benchmark price history is required")
    normalized_prices = {
        str(ticker).strip().upper(): _clean_prices(series)
        for ticker, series in prices_by_ticker.items()
    }
    first_selected = pd.Timestamp(selected["signal_session"].min())
    final_selected = pd.Timestamp(selected["horizon_end"].max())
    first_session = (
        _normalize_date(book_start_session) if book_start_session is not None else first_selected
    )
    final_session = _normalize_date(book_end_session) if book_end_session is not None else final_selected
    if first_session > first_selected or final_session < final_selected:
        raise ValueError("book boundaries must contain every selected signal horizon")
    sessions = benchmark.index[(benchmark.index >= first_session) & (benchmark.index <= final_session)]
    if first_session not in sessions or final_session not in sessions:
        return _failed_backtest(
            "missing_benchmark_sessions",
            selected,
            [f"benchmark does not contain exact boundary sessions {first_session.date()} and {final_session.date()}"],
            eligible_count=len(frame),
        )

    issues: list[str] = []
    rows: list[dict] = []
    previous_weights: dict[str, float] = {}
    cost_rate = float(one_way_cost_bps) / 10_000.0
    for index, session in enumerate(sessions):
        active = selected.loc[
            (selected["signal_session"] <= session) & (selected["horizon_end"] > session),
            "ticker",
        ].drop_duplicates()
        tickers = sorted(active.tolist())
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers} if tickers else {}
        names = set(previous_weights) | set(weights)
        turnover = float(sum(abs(weights.get(name, 0.0) - previous_weights.get(name, 0.0)) for name in names))
        period_end = pd.Timestamp(sessions[index + 1]) if index + 1 < len(sessions) else None
        gross_return = 0.0
        market_return = 0.0
        if period_end is not None:
            market_return = float(benchmark.loc[period_end] / benchmark.loc[session] - 1.0)
            for ticker, weight in weights.items():
                prices = normalized_prices.get(ticker, pd.Series(dtype=float))
                if session not in prices.index or period_end not in prices.index:
                    issues.append(
                        f"{ticker} missing adjusted close for active interval "
                        f"{session.date()} to {period_end.date()}"
                    )
                    continue
                gross_return += weight * float(prices.loc[period_end] / prices.loc[session] - 1.0)
        rows.append(
            {
                "session": pd.Timestamp(session),
                "period_end": period_end,
                "active_tickers": ", ".join(tickers),
                "active_ticker_count": int(len(tickers)),
                "gross_exposure": float(sum(weights.values())),
                "turnover": turnover,
                "transaction_cost": turnover * cost_rate,
                "gross_return": gross_return,
                "net_return": gross_return - turnover * cost_rate,
                "market_return": market_return,
                "exposure_matched_market_return": float(sum(weights.values())) * market_return,
                "weights": weights,
            }
        )
        previous_weights = weights

    if issues:
        return _failed_backtest(
            "missing_active_prices",
            selected,
            sorted(set(issues)),
            eligible_count=len(frame),
        )

    daily = pd.DataFrame(rows)
    gross_cumulative = _compound(daily["gross_return"])
    net_cumulative = _compound(daily["net_return"])
    matched_market = _compound(daily["exposure_matched_market_return"])
    wealth = (1.0 + daily["net_return"].astype(float)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return_sessions = int(daily["period_end"].notna().sum())
    total_turnover = float(daily["turnover"].sum())
    return {
        "status": "ok",
        "backtest_version": BACKTEST_VERSION,
        "daily": daily,
        "selected_signals": selected,
        "metrics": {
            "eligible_prediction_count": int(len(frame)),
            "selected_event_count": int(len(selected)),
            "selection_rate": float(len(selected) / len(frame)),
            "unique_ticker_count": int(selected["ticker"].nunique()),
            "return_session_count": return_sessions,
            "active_return_sessions": int(
                ((daily["gross_exposure"] > 0.0) & daily["period_end"].notna()).sum()
            ),
            "average_gross_exposure": float(
                daily.loc[daily["period_end"].notna(), "gross_exposure"].mean()
            ),
            "total_turnover": total_turnover,
            "average_daily_turnover": (
                total_turnover / return_sessions if return_sessions else None
            ),
            "annualized_turnover": (
                total_turnover * 252.0 / return_sessions if return_sessions else None
            ),
            "round_trip_equivalents": total_turnover / 2.0,
            "total_transaction_cost": float(daily["transaction_cost"].sum()),
            "gross_cumulative_return": gross_cumulative,
            "net_cumulative_return": net_cumulative,
            "exposure_matched_market_return": matched_market,
            "net_minus_exposure_matched_market": net_cumulative - matched_market,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else None,
        },
        "issues": [],
    }


def select_and_evaluate_holdout_strategy(
    model_predictions: Mapping[str, pd.DataFrame],
    *,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    thresholds: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    validation_fraction: float = 0.60,
    min_validation_sessions: int = 10,
    min_test_sessions: int = 5,
    min_validation_signals: int = 5,
    max_validation_selection_rate: float = 0.50,
    max_validation_average_daily_turnover: float = 1.0,
    one_way_cost_bps: float = 5.0,
) -> dict:
    """Choose model/threshold on validation portfolio results, then lock test."""

    if not 0.0 < float(max_validation_average_daily_turnover) <= 2.0:
        raise ValueError("max_validation_average_daily_turnover must be in (0, 2]")
    for name, frame in model_predictions.items():
        if "horizon_end" not in frame.columns or frame["horizon_end"].isna().any():
            raise ValueError(f"{name} predictions require a non-null horizon_end")
    split = select_and_evaluate_holdout_policy(
        model_predictions,
        thresholds=thresholds,
        validation_fraction=validation_fraction,
        min_validation_sessions=min_validation_sessions,
        min_test_sessions=min_test_sessions,
        min_validation_signals=min_validation_signals,
        max_validation_selection_rate=max_validation_selection_rate,
        round_trip_cost_bps=float(one_way_cost_bps) * 2.0,
    )
    if split["status"] not in {"ok", "no_eligible_validation_policy"}:
        return {
            "status": split["status"],
            "backtest_version": BACKTEST_VERSION,
            "candidates": split.get("candidates", pd.DataFrame()),
        }
    event_candidates = split["candidates"].copy()
    if event_candidates.empty or not event_candidates["eligible"].any():
        return {
            "status": "no_eligible_validation_strategy",
            "backtest_version": BACKTEST_VERSION,
            "candidates": event_candidates,
        }

    validation_frames = split["validation_predictions"]
    test_frames = split["test_predictions"]
    reference_validation = next(iter(validation_frames.values()))
    reference_test = next(iter(test_frames.values()))
    validation_start = pd.Timestamp(reference_validation["signal_session"].min())
    validation_end = pd.Timestamp(reference_validation["horizon_end"].max())
    test_start = pd.Timestamp(split["first_test_session"])
    test_end = pd.Timestamp(reference_test["horizon_end"].max())

    strategy_rows: list[dict] = []
    validation_backtests: dict[tuple[str, float], dict] = {}
    for candidate in event_candidates.loc[event_candidates["eligible"]].to_dict("records"):
        model = str(candidate["model"])
        threshold = float(candidate["probability_threshold"])
        backtest = run_long_only_position_backtest(
            validation_frames[model],
            prices_by_ticker=prices_by_ticker,
            benchmark_prices=benchmark_prices,
            probability_threshold=threshold,
            one_way_cost_bps=one_way_cost_bps,
            book_start_session=validation_start,
            book_end_session=validation_end,
        )
        validation_backtests[(model, threshold)] = backtest
        metrics = backtest.get("metrics", {})
        average_turnover = metrics.get("average_daily_turnover")
        turnover_eligible = (
            backtest["status"] == "ok"
            and average_turnover is not None
            and average_turnover <= float(max_validation_average_daily_turnover)
        )
        strategy_rows.append(
            {
                **candidate,
                "strategy_status": backtest["status"],
                "validation_net_cumulative_return": metrics.get("net_cumulative_return"),
                "validation_net_minus_exposure_matched_market": metrics.get(
                    "net_minus_exposure_matched_market"
                ),
                "validation_average_daily_turnover": average_turnover,
                "validation_total_turnover": metrics.get("total_turnover"),
                "validation_max_drawdown": metrics.get("max_drawdown"),
                "turnover_eligible": bool(turnover_eligible),
            }
        )
    candidates = pd.DataFrame(strategy_rows)
    eligible = candidates.loc[candidates["turnover_eligible"]].copy()
    if eligible.empty:
        return {
            "status": "no_eligible_validation_strategy",
            "backtest_version": BACKTEST_VERSION,
            "candidates": candidates,
        }

    chosen = eligible.sort_values(
        [
            "validation_net_minus_exposure_matched_market",
            "validation_average_daily_turnover",
            "selected_count",
            "probability_threshold",
            "model",
        ],
        ascending=[False, True, False, False, True],
    ).iloc[0]
    model = str(chosen["model"])
    threshold = float(chosen["probability_threshold"])
    test_backtest = run_long_only_position_backtest(
        test_frames[model],
        prices_by_ticker=prices_by_ticker,
        benchmark_prices=benchmark_prices,
        probability_threshold=threshold,
        one_way_cost_bps=one_way_cost_bps,
        book_start_session=test_start,
        book_end_session=test_end,
    )
    if test_backtest["status"] != "ok":
        return {
            "status": "locked_test_backtest_unavailable",
            "backtest_version": BACKTEST_VERSION,
            "chosen_model": model,
            "probability_threshold": threshold,
            "test_backtest": test_backtest,
            "candidates": candidates,
        }
    return {
        "status": "ok",
        "backtest_version": BACKTEST_VERSION,
        "chosen_model": model,
        "probability_threshold": threshold,
        "one_way_cost_bps": float(one_way_cost_bps),
        "max_validation_average_daily_turnover": float(
            max_validation_average_daily_turnover
        ),
        "first_test_session": test_start,
        "validation_session_count": split["validation_session_count"],
        "test_session_count": split["test_session_count"],
        "purged_validation_rows": split["purged_validation_rows"],
        "validation_event": evaluate_event_policy(
            validation_frames[model],
            probability_threshold=threshold,
            round_trip_cost_bps=float(one_way_cost_bps) * 2.0,
        ),
        "test_event": evaluate_event_policy(
            test_frames[model],
            probability_threshold=threshold,
            round_trip_cost_bps=float(one_way_cost_bps) * 2.0,
        ),
        "validation_backtest": validation_backtests[(model, threshold)],
        "test_backtest": test_backtest,
        "candidates": candidates,
    }


def _compound(returns: pd.Series) -> float:
    values = returns.astype(float)
    return float(np.prod(1.0 + values) - 1.0)


def _normalize_dates(values) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert(None).dt.normalize()


def _normalize_date(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _clean_prices(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    result = pd.Series(series).dropna().astype(float)
    result.index = pd.to_datetime(result.index, errors="coerce", utc=True)
    result = result.loc[~result.index.isna()]
    result.index = result.index.tz_convert(None).normalize()
    return result.sort_index()[lambda value: ~value.index.duplicated(keep="last")]


def _empty_backtest(status: str) -> dict:
    return {
        "status": status,
        "backtest_version": BACKTEST_VERSION,
        "daily": pd.DataFrame(),
        "selected_signals": pd.DataFrame(),
        "metrics": {},
        "issues": [],
    }


def _failed_backtest(
    status: str,
    selected: pd.DataFrame,
    issues: list[str],
    *,
    eligible_count: int,
) -> dict:
    result = _empty_backtest(status)
    result.update(
        {
            "selected_signals": selected,
            "eligible_prediction_count": int(eligible_count),
            "issues": issues,
        }
    )
    return result
