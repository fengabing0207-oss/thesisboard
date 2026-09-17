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

from src.news_signal_backtest import select_and_evaluate_holdout_strategy
from src.news_signal_lab import build_news_return_dataset, compare_walk_forward_models, session_close_utc


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
    policy = select_and_evaluate_holdout_strategy(
        result["common_predictions"],
        prices_by_ticker={"DEMO": ticker},
        benchmark_prices=benchmark,
        min_validation_sessions=10,
        min_test_sessions=5,
        min_validation_signals=5,
        max_validation_selection_rate=0.6,
        max_validation_average_daily_turnover=1.0,
        one_way_cost_bps=5,
    )
    if policy["status"] != "ok":
        raise RuntimeError(f"synthetic News Signal Lab policy audit failed: {policy['status']}")
    backtest = policy["test_backtest"]

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
        f"strategy spec={policy['strategy_spec_id']} "
        f"calculation parity={policy['calculation_parity_audit']['status']}"
    )
    print(
        f"held-out selected events={policy['test_event']['selected_count']} "
        f"mean net event return={policy['test_event']['mean_net_abnormal_return']}"
    )
    print(
        f"held-out net portfolio return={backtest['metrics']['net_cumulative_return']:.6f} "
        f"total turnover={backtest['metrics']['total_turnover']:.3f}"
    )
    print("WARNING: synthetic adjusted-close results are not executable live performance")


if __name__ == "__main__":
    main()
