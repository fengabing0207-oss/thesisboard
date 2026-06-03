from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abnormal_returns import proxy_for_theme
from src.event_study import classify_event_reaction
from src.forward_tracker import evaluate_forward_returns, record_signal
from src.rule_based_validator import classify_signal, is_technically_overextended
from src.signal_evaluator import evaluate_signal_records
from src.signal_store import list_matured_signal_records


def main() -> None:
    db_path = ROOT / "data" / "validation_demo.db"
    if db_path.exists():
        db_path.unlink()

    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    prices_by_ticker = {
        "NVDA": pd.Series([100, 102, 104, 105, 108, 112, 111, 113, 114, 116, 118, 119], index=dates),
        "UAL": pd.Series([50, 49, 48, 47, 47.5, 48, 48.2, 48.5, 49, 49.4, 49.6, 50], index=dates),
        "CRWD": pd.Series([70, 72, 75, 78, 82, 84, 83, 82, 81, 80, 79, 78], index=dates),
    }
    benchmark_prices = pd.Series([100, 100.5, 101, 101.5, 102, 102.5, 102.8, 103, 103.2, 103.4, 103.5, 103.7], index=dates)
    sector_prices = {
        "SMH": pd.Series([100, 101, 102, 102.5, 103, 104, 104.5, 105, 105.5, 106, 106.2, 106.4], index=dates),
        "JETS": pd.Series([100, 99.5, 99, 98.8, 99, 99.4, 99.7, 100, 100.1, 100.3, 100.5, 100.8], index=dates),
        "HACK": pd.Series([100, 101, 102, 103, 104, 104.5, 104.7, 105, 105.2, 105.4, 105.5, 105.7], index=dates),
    }

    demo_inputs = [
        {"ticker": "NVDA", "theme": "AI Infrastructure / Semiconductor", "sentiment": "positive", "abnormal": 0.035, "metrics": {"rsi": 62, "return_20d": 0.12}, "score": 78},
        {"ticker": "UAL", "theme": "Airlines / Travel Rebound", "sentiment": "positive", "abnormal": -0.02, "metrics": {"rsi": 51, "return_20d": 0.02}, "score": 58},
        {"ticker": "CRWD", "theme": "Cybersecurity", "sentiment": "positive", "abnormal": 0.028, "metrics": {"rsi": 81, "return_20d": 0.31}, "score": 82},
    ]

    for item in demo_inputs:
        reaction = classify_event_reaction(item["sentiment"], item["abnormal"], data_granularity="daily")
        decision = classify_signal(
            catalyst_sentiment=item["sentiment"],
            event_reaction=reaction.label,
            technically_overextended=is_technically_overextended(item["metrics"]),
            setup_score=item["score"],
            volume_confirmation=True,
        )
        created_at = dates[0]
        record_signal(
            ticker=item["ticker"],
            created_at=created_at,
            horizon_days=5,
            classification=decision.classification,
            score=decision.score,
            theme=item["theme"],
            benchmark="SPY",
            sector_proxy=proxy_for_theme(item["theme"]),
            price_at_creation=float(prices_by_ticker[item["ticker"]].loc[created_at]),
            rule_version=decision.rule_version,
            db_path=db_path,
        )

    evaluate_forward_returns(
        db_path=db_path,
        prices_by_ticker=prices_by_ticker,
        benchmark_prices=benchmark_prices,
        sector_prices_by_proxy=sector_prices,
        as_of=dates[-1],
    )
    metrics = evaluate_signal_records(list_matured_signal_records(db_path))

    print("ThesisBoard validation demo")
    print(f"records: {metrics['sample_size']}")
    print(f"tradeable hit rate: {metrics['hit_rate']}")
    print(f"base rate: {metrics['base_rate']}")
    print(f"excess hit rate: {metrics['excess_hit_rate']}")
    print(f"avg forward abnormal return: {metrics['average_forward_abnormal_return']}")
    print(f"false positives: {metrics['false_positives']}")
    print(f"false negatives: {metrics['false_negatives']}")
    print(f"avoid chase records: {metrics['avoid_chase_count']}")


if __name__ == "__main__":
    main()
