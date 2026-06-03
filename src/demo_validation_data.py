"""Reusable demo-data builders for the ThesisBoard validation spine.

This module is the single source of synthetic data and the wiring that runs it
through the real validation core (event study -> rule-based classification ->
forward-return tracking -> cohort-relative evaluation). Both the Streamlit
``Validation Lab`` and ``scripts/run_validation_demo.py`` consume it so the app
shows the validation core's actual output instead of hard-coded final metrics.

As of PR #15A, price data flows in through a :class:`PriceProvider` rather than
being constructed inline. The default provider wraps the same synthetic data, so
behavior is unchanged; a ``CSVPriceProvider`` (or, later, an online source) can
be injected to feed real adjusted-close prices through the identical pipeline.

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
from .price_provider import DemoPriceProvider, PriceDataBundle, normalize_price_series
from .rule_based_validator import classify_signal, is_technically_overextended
from .signal_evaluator import evaluate_signal_records
from .signal_store import list_matured_signal_records, list_signal_snapshots

DEMO_HORIZON_DAYS = 5
DEMO_BENCHMARK = "SPY"
DEMO_REQUESTED_START = pd.Timestamp("2026-01-01")
DEMO_PERIODS = 12

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
    dates = pd.bdate_range(DEMO_REQUESTED_START, periods=DEMO_PERIODS)
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


def demo_price_universe() -> dict[str, pd.Series]:
    """Flatten the synthetic data into a single uppercase ``{symbol: series}`` map."""
    data = build_demo_price_data()
    universe = {DEMO_BENCHMARK: data["benchmark_prices"]}
    universe.update(data["sector_prices"])
    universe.update(data["prices_by_ticker"])
    return {symbol.upper(): normalize_price_series(series) for symbol, series in universe.items()}


def default_demo_provider() -> DemoPriceProvider:
    """Return the default in-memory provider backed by the synthetic universe."""
    return DemoPriceProvider(demo_price_universe())


def _demo_symbol_roles() -> dict:
    """Classify demo symbols into required vs optional for the validation run.

    Required: the benchmark, the signal tickers, and their sector proxies — the
    run cannot proceed without them. Optional: cohort-only symbols, which may be
    absent as long as the cohort does not become empty.
    """
    inputs = build_demo_signal_inputs()
    signal_tickers = [item["ticker"].upper() for item in inputs]
    signal_proxies = []
    for item in inputs:
        proxy = proxy_for_theme(item["theme"])
        if proxy:
            signal_proxies.append(proxy.upper())

    required = {DEMO_BENCHMARK, *signal_tickers, *signal_proxies}
    cohort_tickers = {ticker.upper() for ticker in _COHORT_PROXY_BY_TICKER}
    cohort_proxies = {proxy.upper() for proxy in _COHORT_PROXY_BY_TICKER.values() if proxy}
    optional = (cohort_tickers | cohort_proxies) - required

    return {
        "signal_tickers": signal_tickers,
        "signal_proxies": signal_proxies,
        "required": required,
        "optional": optional,
    }


def build_demo_validation_database(db_path: Path | str, provider=None) -> dict:
    """Populate ``db_path`` by running the demo inputs through the validation core.

    Price data is sourced from ``provider`` (defaulting to the in-memory demo
    provider). Each input is run through the event study and rule-based
    classifier, recorded as a signal snapshot, then evaluated for forward
    returns.

    Returns a dict with:
      - ``price_data``: ``bundle.prices`` (kept for backward compatibility)
      - ``bundle``: the full :class:`PriceDataBundle` with provenance metadata
      - ``evaluation``: enriched evaluation records (incl. ``data_quality_flag``,
        ``beta_fallback_used``)
    """
    _remove_db_files(db_path)
    provider = provider or default_demo_provider()
    roles = _demo_symbol_roles()

    window = pd.bdate_range(DEMO_REQUESTED_START, periods=DEMO_PERIODS)
    bundle = provider.get_history(sorted(roles["required"] | roles["optional"]), window[0], window[-1])

    missing_required = sorted(roles["required"].intersection(bundle.missing_symbols))
    if missing_required:
        raise ValueError(
            f"provider {bundle.source!r} is missing required price history for {missing_required}"
        )

    benchmark_prices = bundle.prices[DEMO_BENCHMARK]
    created_at = pd.Timestamp(benchmark_prices.index.min())
    as_of = pd.Timestamp(benchmark_prices.index.max())

    signal_prices = {ticker: bundle.prices[ticker] for ticker in roles["signal_tickers"]}
    sector_prices = {proxy: bundle.prices[proxy] for proxy in roles["signal_proxies"]}

    for item in build_demo_signal_inputs():
        ticker = item["ticker"].upper()
        proxy = proxy_for_theme(item["theme"])
        sector_proxy = proxy.upper() if proxy else None
        event_end = trading_session_horizon_end(created_at, 1, benchmark_prices.index)
        event_summary = abnormal_return_summary(
            ticker_prices=signal_prices[ticker],
            benchmark_prices=benchmark_prices,
            start_date=created_at,
            end_date=event_end,
            sector_prices=sector_prices.get(sector_proxy) if sector_proxy else None,
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
            ticker=ticker,
            created_at=created_at,
            horizon_days=DEMO_HORIZON_DAYS,
            classification=decision.classification,
            score=decision.score,
            theme=item["theme"],
            benchmark=DEMO_BENCHMARK,
            sector_proxy=sector_proxy,
            price_at_creation=float(signal_prices[ticker].loc[created_at]),
            rule_version=decision.rule_version,
            db_path=db_path,
        )

    evaluation = evaluate_forward_returns(
        db_path=db_path,
        prices_by_ticker=signal_prices,
        benchmark_prices=benchmark_prices,
        sector_prices_by_proxy=sector_prices,
        as_of=as_of,
    )
    return {"price_data": bundle.prices, "bundle": bundle, "evaluation": evaluation}


def build_demo_universe_cohort(
    *,
    created_at,
    horizon_days: int,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    sector_prices: dict[str, pd.Series],
) -> list[dict]:
    """Build the synthetic universe cohort used for the base-rate comparison.

    The cohort is every available ticker in the demo universe evaluated over the
    same creation date and horizon, so hit rates can be read against a base rate
    for the same universe. Tickers with no price data are skipped (optional);
    an empty cohort is an error, since the base rate would be undefined.
    """
    records = []
    end_date = trading_session_horizon_end(created_at, horizon_days, benchmark_prices.index)
    for ticker, prices in prices_by_ticker.items():
        if prices is None or prices.empty:
            continue
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
    if not records:
        raise ValueError("universe cohort is empty; cannot compute a base rate")
    return records


def prepare_validation_lab_data(db_path: Path | str | None = None, provider=None) -> dict:
    """Run the full demo pipeline and return UI-ready validation-core output.

    When ``db_path`` is None a throwaway temp database is used and cleaned up.
    ``provider`` defaults to the in-memory demo provider; injecting another
    provider (e.g. ``CSVPriceProvider``) feeds real prices through the same
    pipeline. The returned ``snapshots`` and ``outcomes`` are scrubbed of the
    deprecated ``hit`` field; ``outcomes`` carry price provenance.

    Returns a dict with:
      - ``snapshots``: creation-time signal snapshots (+ persisted outcomes)
      - ``outcomes``: matured outcomes enriched with abnormal-return data quality
        and price provenance (source / adjustment / coverage / quality flags)
      - ``metrics``: grouped + summary metrics from ``evaluate_signal_records``
      - ``cohort``: the synthetic universe cohort behind the base rate
      - ``price_provenance``: bundle-level source / adjustment / missing symbols
    """
    owns_db = db_path is None
    if owns_db:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        db_path = handle.name

    try:
        built = build_demo_validation_database(db_path, provider=provider)
        bundle = built["bundle"]

        snapshots = [_strip_deprecated_outcome(row) for row in list_signal_snapshots(db_path, include_outcomes=True)]
        outcomes = [_attach_price_provenance(_strip_deprecated_outcome(row), bundle) for row in built["evaluation"]]
        cohort = _build_cohort_from_bundle(bundle)
        metrics = evaluate_signal_records(list_matured_signal_records(db_path), universe_cohort=cohort)
        return {
            "snapshots": snapshots,
            "outcomes": outcomes,
            "metrics": metrics,
            "cohort": cohort,
            "price_provenance": {
                "source": bundle.source,
                "adjustment": bundle.adjustment,
                "missing_symbols": list(bundle.missing_symbols),
            },
        }
    finally:
        if owns_db:
            _remove_db_files(db_path)


def _build_cohort_from_bundle(bundle: PriceDataBundle) -> list[dict]:
    """Construct the universe cohort from a bundle, honoring same-provider data."""
    benchmark_prices = bundle.prices[DEMO_BENCHMARK]
    created_at = pd.Timestamp(benchmark_prices.index.min())
    cohort_prices = {
        ticker.upper(): bundle.prices[ticker.upper()]
        for ticker in _COHORT_PROXY_BY_TICKER
        if ticker.upper() in bundle.prices
    }
    cohort_sector_prices = {
        proxy.upper(): bundle.prices[proxy.upper()]
        for proxy in _COHORT_PROXY_BY_TICKER.values()
        if proxy and proxy.upper() in bundle.prices
    }
    return build_demo_universe_cohort(
        created_at=created_at,
        horizon_days=DEMO_HORIZON_DAYS,
        prices_by_ticker=cohort_prices,
        benchmark_prices=benchmark_prices,
        sector_prices=cohort_sector_prices,
    )


def _attach_price_provenance(record: dict, bundle: PriceDataBundle) -> dict:
    """Attach per-symbol price provenance and a unified quality-flag list.

    The unified ``data_quality_flags`` is the union of provider-side metadata
    flags (empty in PR #15A) and the engine's own ``data_quality_flag`` (e.g.
    ``beta_fallback_used``), so the UI has one place to read data caveats.
    """
    enriched = dict(record)
    symbol = str(record.get("ticker", "")).upper()
    meta = bundle.metadata.get(symbol)

    if meta is not None:
        enriched["price_source"] = meta.source
        enriched["price_adjustment"] = meta.adjustment
        enriched["price_actual_start"] = None if meta.actual_start is None else pd.Timestamp(meta.actual_start).isoformat()
        enriched["price_actual_end"] = None if meta.actual_end is None else pd.Timestamp(meta.actual_end).isoformat()
        flags = list(meta.data_quality_flags)
    else:
        enriched["price_source"] = bundle.source
        enriched["price_adjustment"] = bundle.adjustment
        enriched["price_actual_start"] = None
        enriched["price_actual_end"] = None
        flags = []

    engine_flag = record.get("data_quality_flag")
    if engine_flag and engine_flag != "ok":
        flags = flags + [part for part in str(engine_flag).split(",") if part]
    enriched["data_quality_flags"] = flags
    return enriched


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
