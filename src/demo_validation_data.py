"""Reusable demo-data builders for the ThesisBoard validation spine.

This module is the single source of synthetic data and the wiring that runs it
through the real validation core (event study -> rule-based classification ->
forward-return tracking -> cohort-relative evaluation). Both the Streamlit
``Validation Lab`` and ``scripts/run_validation_demo.py`` consume it so the app
shows the validation core's actual output instead of hard-coded final metrics.

Everything here is demo data only. The synthetic universe cohort exists to
validate the *shape* of the workflow, not to demonstrate predictive power.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .abnormal_returns import abnormal_return_summary, proxy_for_theme
from .event_study import classify_event_reaction
from .forward_tracker import (
    evaluate_forward_returns,
    record_signal,
    trading_session_horizon_end,
)
from .rule_based_validator import classify_signal, is_technically_overextended
from .signal_evaluator import evaluate_signal_records
from .signal_store import list_matured_signal_records, list_signal_snapshots

DEMO_HORIZON_DAYS = 5
DEMO_BENCHMARK = "SPY"

# The generic ``hit`` column is deprecated (see signal_store.DEPRECATED_HIT_FIELD_NOTE).
# UI-facing data must rely on the explicit, classification-specific outcomes below.
DEPRECATED_OUTCOME_FIELD = "hit"
EXPLICIT_OUTCOME_FIELDS = (
    "trade_hit",
    "watch_followthrough",
    "avoided_bad_trade",
    "false_negative",
)

# Maps each demo ticker to the theme/sector proxy used when building the
# synthetic cohort. MSFT has no proxy on purpose, to exercise the market-only path.
_COHORT_PROXY_BY_TICKER = {
    "NVDA": "SMH",
    "UAL": "JETS",
    "CRWD": "HACK",
    "MSFT": None,
    "DAL": "JETS",
}


def build_demo_price_data() -> dict:
    """Return the synthetic price universe used across the demo.

    Includes ticker prices, the market benchmark, and theme/sector proxies on a
    shared business-day index so horizons resolve against the benchmark calendar.
    """
    dates = pd.bdate_range("2026-01-01", periods=12)
    prices_by_ticker = {
        "NVDA": pd.Series([100, 102, 104, 105, 108, 112, 111, 113, 114, 116, 118, 119], index=dates),
        "UAL": pd.Series([50, 49, 48, 47, 47.5, 48, 48.2, 48.5, 49, 49.4, 49.6, 50], index=dates),
        "CRWD": pd.Series([70, 72, 75, 78, 82, 84, 83, 82, 81, 80, 79, 78], index=dates),
        "MSFT": pd.Series([200, 201, 202, 202, 203, 204, 204, 205, 205, 206, 206, 207], index=dates),
        "DAL": pd.Series([40, 39.8, 39.6, 39.5, 39.4, 39.7, 39.9, 40.1, 40.2, 40.3, 40.5, 40.6], index=dates),
    }
    benchmark_prices = pd.Series(
        [100, 100.5, 101, 101.5, 102, 102.5, 102.8, 103, 103.2, 103.4, 103.5, 103.7],
        index=dates,
    )
    sector_prices = {
        "SMH": pd.Series([100, 101, 102, 102.5, 103, 104, 104.5, 105, 105.5, 106, 106.2, 106.4], index=dates),
        "JETS": pd.Series([100, 99.5, 99, 98.8, 99, 99.4, 99.7, 100, 100.1, 100.3, 100.5, 100.8], index=dates),
        "HACK": pd.Series([100, 101, 102, 103, 104, 104.5, 104.7, 105, 105.2, 105.4, 105.5, 105.7], index=dates),
    }
    return {
        "dates": dates,
        "prices_by_ticker": prices_by_ticker,
        "benchmark_prices": benchmark_prices,
        "sector_prices": sector_prices,
    }


def build_demo_signal_inputs() -> list[dict]:
    """Return the demo catalyst/setup inputs fed into the validation core."""
    return [
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


def build_demo_validation_database(db_path: Path | str) -> dict:
    """Populate ``db_path`` by running the demo inputs through the validation core.

    Each input is run through the event study and rule-based classifier, recorded
    as a signal snapshot, then evaluated for forward returns. Returns a dict with
    the price universe (``price_data``) and the enriched evaluation records
    (``evaluation``) carrying data-quality flags such as ``beta_fallback_used``.
    """
    _remove_db_files(db_path)

    price_data = build_demo_price_data()
    dates = price_data["dates"]
    prices_by_ticker = price_data["prices_by_ticker"]
    benchmark_prices = price_data["benchmark_prices"]
    sector_prices = price_data["sector_prices"]
    created_at = dates[0]

    for item in build_demo_signal_inputs():
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
            horizon_days=DEMO_HORIZON_DAYS,
            classification=decision.classification,
            score=decision.score,
            theme=item["theme"],
            benchmark=DEMO_BENCHMARK,
            sector_proxy=sector_proxy,
            price_at_creation=float(prices_by_ticker[item["ticker"]].loc[created_at]),
            rule_version=decision.rule_version,
            db_path=db_path,
        )

    evaluation = evaluate_forward_returns(
        db_path=db_path,
        prices_by_ticker=prices_by_ticker,
        benchmark_prices=benchmark_prices,
        sector_prices_by_proxy=sector_prices,
        as_of=dates[-1],
    )
    return {"price_data": price_data, "evaluation": evaluation}


def build_demo_universe_cohort(
    *,
    created_at,
    horizon_days: int,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    sector_prices: dict[str, pd.Series],
) -> list[dict]:
    """Build the synthetic universe cohort used for the base-rate comparison.

    The cohort is every ticker in the demo universe evaluated over the same
    creation date and horizon, so hit rates can be read against a base rate for
    the same universe rather than in isolation.
    """
    records = []
    end_date = trading_session_horizon_end(created_at, horizon_days, benchmark_prices.index)
    for ticker, prices in prices_by_ticker.items():
        sector_proxy = _COHORT_PROXY_BY_TICKER.get(ticker)
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


def prepare_validation_lab_data(db_path: Path | str | None = None) -> dict:
    """Run the full demo pipeline and return UI-ready validation-core output.

    When ``db_path`` is None a throwaway temp database is used and cleaned up.
    The returned ``snapshots`` and ``outcomes`` are scrubbed of the deprecated
    ``hit`` field so the Lab only ever presents the explicit outcome semantics.

    Returns a dict with:
      - ``snapshots``: creation-time signal snapshots (+ persisted outcomes)
      - ``outcomes``: matured outcomes enriched with abnormal-return data quality
      - ``metrics``: grouped + summary metrics from ``evaluate_signal_records``
      - ``cohort``: the synthetic universe cohort behind the base rate
    """
    owns_db = db_path is None
    if owns_db:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        db_path = handle.name

    try:
        built = build_demo_validation_database(db_path)
        price_data = built["price_data"]

        snapshots = [_strip_deprecated_outcome(row) for row in list_signal_snapshots(db_path, include_outcomes=True)]
        outcomes = [_strip_deprecated_outcome(row) for row in built["evaluation"]]
        cohort = build_demo_universe_cohort(
            created_at=price_data["dates"][0],
            horizon_days=DEMO_HORIZON_DAYS,
            prices_by_ticker=price_data["prices_by_ticker"],
            benchmark_prices=price_data["benchmark_prices"],
            sector_prices=price_data["sector_prices"],
        )
        metrics = evaluate_signal_records(list_matured_signal_records(db_path), universe_cohort=cohort)
        return {
            "snapshots": snapshots,
            "outcomes": outcomes,
            "metrics": metrics,
            "cohort": cohort,
        }
    finally:
        if owns_db:
            _remove_db_files(db_path)


def _strip_deprecated_outcome(record: dict) -> dict:
    """Return a copy of ``record`` without the deprecated generic ``hit`` field."""
    return {key: value for key, value in record.items() if key != DEPRECATED_OUTCOME_FIELD}


def _remove_db_files(db_path: Path | str) -> None:
    """Remove a demo SQLite database and its WAL/SHM sidecars if present."""
    base = Path(db_path)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = base.with_name(base.name + suffix) if suffix else base
        if candidate.exists():
            candidate.unlink()
