import pandas as pd
import pytest

from src.forward_tracker import evaluate_forward_returns, record_signal
from src.signal_store import list_matured_signal_records


def test_forward_tracker_records_matured_outcome(tmp_path):
    db_path = tmp_path / "signals.db"
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    ticker_prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107], index=dates)
    benchmark_prices = pd.Series([100, 100, 100, 100, 100, 100, 100, 100], index=dates)

    signal_id = record_signal(
        ticker="NVDA",
        created_at=dates[0],
        horizon_days=5,
        classification="Tradeable",
        score=82,
        theme="AI Infrastructure",
        benchmark="SPY",
        sector_proxy=None,
        price_at_creation=100,
        rule_version="rules.v1",
        db_path=db_path,
    )

    evaluate_forward_returns(
        db_path=db_path,
        prices_by_ticker={"NVDA": ticker_prices},
        benchmark_prices=benchmark_prices,
        as_of=dates[-1],
    )
    records = list_matured_signal_records(db_path)
    assert records[0]["id"] == signal_id
    assert records[0]["forward_return"] == pytest.approx(0.05)
    assert records[0]["forward_abnormal_return"] == pytest.approx(0.05)
    assert records[0]["hit"] is True
    assert records[0]["is_matured"] is True
