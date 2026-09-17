"""Run a deterministic synthetic News Signal Lab pipeline demonstration.

This proves the data-flow and leakage guards offline. The generated relationship
is intentionally artificial and must not be interpreted as predictive evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_signal_lab import build_news_return_dataset, session_close_utc, walk_forward_baseline


def main() -> None:
    sessions = pd.bdate_range("2025-01-02", periods=100)
    market_returns = np.asarray([0.002, -0.001, 0.0015, -0.0005, 0.001] * 20)
    ticker_returns = 1.1 * market_returns
    news_items = []

    for index in range(35, 90):
        positive = index % 2 == 1
        ticker_returns[index + 1] += 0.012 if positive else -0.012
        news_items.append(
            {
                "ticker": "DEMO",
                "title": "Strong demand and profit growth" if positive else "Weak demand and loss warning",
                "first_seen_at": session_close_utc(sessions[index]) - pd.Timedelta(minutes=30),
            }
        )

    benchmark = pd.Series(100 * np.cumprod(1 + market_returns), index=sessions)
    ticker = pd.Series(80 * np.cumprod(1 + ticker_returns), index=sessions)
    dataset = build_news_return_dataset(
        news_items,
        prices_by_ticker={"DEMO": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    )
    result = walk_forward_baseline(dataset, min_train_rows=20, min_train_sessions=20)
    if result["status"] != "ok":
        raise RuntimeError(f"synthetic News Signal Lab demo failed: {result['status']}")

    metrics = result["metrics"]
    print("ThesisBoard News Signal Lab synthetic demonstration")
    print("WARNING: generated data proves pipeline behavior only; it is not alpha evidence")
    print(f"dataset rows: {len(dataset)}")
    print(f"out-of-sample rows: {metrics['prediction_count']}")
    print(f"directional accuracy: {metrics['directional_accuracy']}")
    print(f"historical-rate accuracy: {metrics['historical_rate_accuracy']}")
    print(f"brier score: {metrics['brier_score']}")


if __name__ == "__main__":
    main()
