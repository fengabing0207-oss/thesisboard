import pandas as pd
import pytest

from src.abnormal_returns import (
    abnormal_return_summary,
    beta_adjusted_abnormal_return,
    combined_abnormal_return,
    proxy_for_theme,
    raw_return,
    rolling_beta,
    sector_adjusted_abnormal_return,
)


def test_raw_return_uses_start_and_end_prices():
    prices = pd.Series([100, 110, 121], index=pd.date_range("2026-01-01", periods=3))
    assert raw_return(prices, "2026-01-01", "2026-01-03") == pytest.approx(0.21)


def test_rolling_beta_and_abnormal_return_math():
    dates = pd.date_range("2026-01-01", periods=40)
    benchmark_returns = pd.Series([0.01, -0.005, 0.02, 0.0] * 10, index=dates)
    ticker_returns = benchmark_returns * 2

    assert rolling_beta(ticker_returns, benchmark_returns) == pytest.approx(2)
    assert beta_adjusted_abnormal_return(0.08, 1.5, 0.04) == pytest.approx(0.02)
    assert sector_adjusted_abnormal_return(0.08, 0.03) == pytest.approx(0.05)
    assert combined_abnormal_return(0.08, 1.0, 0.04, 0.02) == pytest.approx(0.05)


def test_theme_proxy_lookup():
    assert proxy_for_theme("AI Infrastructure / Semiconductor") == "SMH"
    assert proxy_for_theme("Airlines / Travel Rebound") == "JETS"


def test_beta_estimation_ignores_future_returns_after_signal_start():
    dates = pd.date_range("2026-01-01", periods=70)
    start_date = dates[45]
    end_date = dates[50]
    benchmark_returns = pd.Series(([0.01, -0.005, 0.02, 0.0] * 18)[:70], index=dates)
    ticker_returns = benchmark_returns.copy()
    ticker_returns.loc[ticker_returns.index >= start_date] = benchmark_returns.loc[benchmark_returns.index >= start_date] * 8
    benchmark_prices = (1 + benchmark_returns).cumprod() * 100
    ticker_prices = (1 + ticker_returns).cumprod() * 100

    summary = abnormal_return_summary(
        ticker_prices=ticker_prices,
        benchmark_prices=benchmark_prices,
        start_date=start_date,
        end_date=end_date,
        lookback_days=40,
    )

    assert summary["beta"] == pytest.approx(1)
    assert summary["beta_estimation_end"] == pd.Timestamp(start_date).isoformat()
