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

from src.news_signal_lab import build_news_return_dataset, compare_walk_forward_models, session_close_utc
from src.news_signal_policy import select_and_evaluate_holdout_policy


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
    result = compare_walk_forward_models(dataset, min_train_rows=20, min_train_sessions=20)
    if result["status"] != "ok":
        raise RuntimeError(f"synthetic News Signal Lab model comparison failed: {result['status']}")
    policy = select_and_evaluate_holdout_policy(
        result["common_predictions"],
        min_validation_sessions=10,
        min_test_sessions=5,
        min_validation_signals=5,
        max_validation_selection_rate=0.6,
        round_trip_cost_bps=10,
    )
    if policy["status"] != "ok":
        raise RuntimeError(f"synthetic News Signal Lab policy audit failed: {policy['status']}")

    print("ThesisBoard News Signal Lab synthetic demonstration")
    print("WARNING: generated data proves pipeline behavior only; it is not alpha evidence")
    print(f"dataset rows: {len(dataset)}")
    for row in result["comparison"].to_dict("records"):
        print(
            f"{row['model']} OOS rows={row['prediction_count']} "
            f"accuracy={row['directional_accuracy']:.3f} brier={row['brier_score']:.4f}"
        )
    print(
        f"validation-selected policy: model={policy['chosen_model']} "
        f"threshold={policy['probability_threshold']:.2f}"
    )
    print(
        f"held-out selected events={policy['test']['selected_count']} "
        f"mean net event return={policy['test']['mean_net_abnormal_return']}"
    )
    print("WARNING: event returns are not portfolio P&L or turnover")


if __name__ == "__main__":
    main()
