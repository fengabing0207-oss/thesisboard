from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abnormal_returns import abnormal_return_summary, proxy_for_theme
from src.event_study import classify_event_reaction
from src.forward_tracker import evaluate_forward_returns, record_signal, trading_session_horizon_end
from src.rule_based_validator import classify_signal, is_technically_overextended
from src.signal_evaluator import evaluate_signal_records
from src.signal_store import list_matured_signal_records


def main() -> None:
    db_path = ROOT / "data" / "validation_demo.db"
    if db_path.exists():
        db_path.unlink()

    dates = pd.bdate_range("2026-01-01", periods=12)
    prices_by_ticker = {
        "NVDA": pd.Series([100, 102, 104, 105, 108, 112, 111, 113, 114, 116, 118, 119], index=dates),
        "UAL": pd.Series([50, 49, 48, 47, 47.5, 48, 48.2, 48.5, 49, 49.4, 49.6, 50], index=dates),
        "CRWD": pd.Series([70, 72, 75, 78, 82, 84, 83, 82, 81, 80, 79, 78], index=dates),
        "MSFT": pd.Series([200, 201, 202, 202, 203, 204, 204, 205, 205, 206, 206, 207], index=dates),
        "DAL": pd.Series([40, 39.8, 39.6, 39.5, 39.4, 39.7, 39.9, 40.1, 40.2, 40.3, 40.5, 40.6], index=dates),
    }
    benchmark_prices = pd.Series([100, 100.5, 101, 101.5, 102, 102.5, 102.8, 103, 103.2, 103.4, 103.5, 103.7], index=dates)
    sector_prices = {
        "SMH": pd.Series([100, 101, 102, 102.5, 103, 104, 104.5, 105, 105.5, 106, 106.2, 106.4], index=dates),
        "JETS": pd.Series([100, 99.5, 99, 98.8, 99, 99.4, 99.7, 100, 100.1, 100.3, 100.5, 100.8], index=dates),
        "HACK": pd.Series([100, 101, 102, 103, 104, 104.5, 104.7, 105, 105.2, 105.4, 105.5, 105.7], index=dates),
    }

    demo_inputs = [
        {
            "ticker": "NVDA",
            "theme": "AI Infrastructure / Semiconductor",
            "sentiment": "positive",
            "metrics": {"rsi": 62, "return_20d": 0.12},
            "score": 78,
        },
        {
            "ticker": "UAL",
            "theme": "Airlines / Travel Rebound",
            "sentiment": "positive",
            "metrics": {"rsi": 51, "return_20d": 0.02},
            "score": 58,
        },
        {
            "ticker": "CRWD",
            "theme": "Cybersecurity",
            "sentiment": "positive",
            "metrics": {"rsi": 81, "return_20d": 0.31},
            "score": 82,
        },
    ]

    for item in demo_inputs:
        created_at = dates[0]
        sector_proxy = proxy_for_theme(item["theme"])
        event_end = trading_session_horizon_end(created_at, 1, benchmark_prices.index)
        event_summary = abnormal_return_summary(
            ticker_prices=prices_by_ticker[item["ticker"]],
            benchmark_prices=benchmark_prices,
            start_date=created_at,
            end_date=event_end,
            sector_prices=sector_prices.get(sector_proxy),
            sector_proxy=sector_proxy,
            beta_estimation_end=created_at,
        )
        reaction = classify_event_reaction(
            item["sentiment"],
            event_summary["combined_abnormal_return"],
            data_granularity="daily",
        )
        decision = classify_signal(
            catalyst_sentiment=item["sentiment"],
            event_reaction=reaction.label,
            technically_overextended=is_technically_overextended(item["metrics"]),
            setup_score=item["score"],
            volume_confirmation=True,
        )
        record_signal(
            ticker=item["ticker"],
            created_at=created_at,
            horizon_days=5,
            classification=decision.classification,
            score=decision.score,
            theme=item["theme"],
            benchmark="SPY",
            sector_proxy=sector_proxy,
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
    universe_cohort = _build_demo_universe_cohort(
        created_at=dates[0],
        horizon_days=5,
        prices_by_ticker=prices_by_ticker,
        benchmark_prices=benchmark_prices,
        sector_prices=sector_prices,
    )
    metrics = evaluate_signal_records(list_matured_signal_records(db_path), universe_cohort=universe_cohort)

    print("ThesisBoard validation demo")
    print(f"records: {metrics['sample_size']}")
    tradeable_group = next((group for group in metrics["groups"] if group["classification"] == "Tradeable"), None)
    print(f"tradeable hit rate: {None if tradeable_group is None else tradeable_group['trade_hit_rate']}")
    print(f"cohort base rate: {None if tradeable_group is None else tradeable_group['base_rate']}")
    print(f"excess trade hit rate: {None if tradeable_group is None else tradeable_group['excess_trade_hit_rate']}")
    print(f"avg forward abnormal return: {metrics['average_forward_abnormal_return']}")
    print(f"false positives: {metrics['false_positives']}")
    print(f"false negatives: {metrics['false_negatives']}")
    print(f"avoid chase records: {metrics['avoid_chase_count']}")


def _build_demo_universe_cohort(
    *,
    created_at,
    horizon_days: int,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    sector_prices: dict[str, pd.Series],
) -> list[dict]:
    records = []
    end_date = trading_session_horizon_end(created_at, horizon_days, benchmark_prices.index)
    proxy_by_ticker = {"NVDA": "SMH", "UAL": "JETS", "CRWD": "HACK", "MSFT": None, "DAL": "JETS"}
    for ticker, prices in prices_by_ticker.items():
        sector_proxy = proxy_by_ticker.get(ticker)
        summary = abnormal_return_summary(
            ticker_prices=prices,
            benchmark_prices=benchmark_prices,
            start_date=created_at,
            end_date=end_date,
            sector_prices=sector_prices.get(sector_proxy) if sector_proxy else None,
            sector_proxy=sector_proxy,
            beta_estimation_end=created_at,
        )
        records.append(
            {
                "ticker": ticker,
                "created_at": pd.Timestamp(created_at).isoformat(),
                "horizon_days": horizon_days,
                "forward_abnormal_return": summary["combined_abnormal_return"],
            }
        )
    return records


if __name__ == "__main__":
    main()
