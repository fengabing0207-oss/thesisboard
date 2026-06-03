from pathlib import Path

import pandas as pd
import pytest

from src.abnormal_returns import abnormal_return_summary
from src.price_provider import (
    CSVPriceProvider,
    DemoPriceProvider,
    PriceDataBundle,
    PriceSeriesMetadata,
    assemble_bundle,
    normalize_price_series,
)

FIXTURES = Path(__file__).parent / "fixtures" / "prices"
CLEAN = FIXTURES / "clean_long.csv"
DIRTY = FIXTURES / "dirty_long.csv"

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-01-31")


def test_normalize_sorts_dedupes_drops_nan_and_strips_tz():
    raw = pd.Series(
        [30.0, 10.0, 20.0, 25.0, None, 40.0],
        index=[
            "2026-01-05",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-03",
            "2026-01-06T00:00:00+00:00",
        ],
    )
    out = normalize_price_series(raw)

    assert list(out.index) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
    ]
    assert out.loc["2026-01-02"] == 25.0  # duplicate date keeps last
    assert out.index.tz is None
    assert out.dtype == float
    assert not out.isna().any()


def test_csv_provider_normalizes_dirty_input():
    bundle = CSVPriceProvider(DIRTY).get_history(["DIRTY"], START, END)
    series = bundle.prices["DIRTY"]

    assert list(series.index) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-07"),
    ]
    assert series.loc["2026-01-02"] == 25.0
    assert series.index.is_monotonic_increasing
    assert series.index.is_unique


def test_csv_provider_reads_present_symbols_and_uppercases():
    bundle = CSVPriceProvider(CLEAN).get_history(["spy", "AAA", "bbb"], START, END)

    assert set(bundle.prices) == {"SPY", "AAA", "BBB"}
    assert bundle.missing_symbols == []
    assert bundle.prices["SPY"].loc["2026-01-05"] == 102.0
    assert bundle.source == "local-csv"
    assert bundle.adjustment == "as_provided"


def test_csv_provider_reports_missing_symbols():
    bundle = CSVPriceProvider(CLEAN).get_history(["SPY", "ZZZ"], START, END)

    assert "ZZZ" in bundle.missing_symbols
    assert "ZZZ" not in bundle.prices
    assert bundle.metadata["ZZZ"].actual_start is None
    assert bundle.metadata["ZZZ"].actual_end is None


def test_metadata_records_provenance_and_coverage():
    bundle = CSVPriceProvider(CLEAN).get_history(["SPY"], START, END)
    meta = bundle.metadata["SPY"]

    assert meta.symbol == "SPY"
    assert meta.source == "local-csv"
    assert meta.adjustment == "as_provided"
    assert meta.requested_start == START
    assert meta.actual_start == pd.Timestamp("2026-01-01")
    assert meta.actual_end == pd.Timestamp("2026-01-05")
    # PR #15A leaves the reserved #15B fields inert.
    assert meta.stale_price_dates == []
    assert meta.survivorship_verified is None


def test_csv_provider_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol,date,close\nSPY,2026-01-01,100\n")
    with pytest.raises(ValueError, match="missing required columns"):
        CSVPriceProvider(bad).get_history(["SPY"], START, END)


def test_bundle_rejects_mixed_source():
    meta = {
        "AAA": PriceSeriesMetadata("AAA", "demo", "demo-synthetic", START, END, START, END),
        "BBB": PriceSeriesMetadata("BBB", "local-csv", "demo-synthetic", START, END, START, END),
    }
    with pytest.raises(ValueError, match="mixes price sources"):
        PriceDataBundle(prices={}, metadata=meta, missing_symbols=[], source="demo", adjustment="demo-synthetic")


def test_bundle_rejects_mixed_adjustment():
    meta = {
        "AAA": PriceSeriesMetadata("AAA", "demo", "demo-synthetic", START, END, START, END),
        "BBB": PriceSeriesMetadata("BBB", "demo", "as_provided", START, END, START, END),
    }
    with pytest.raises(ValueError, match="mixes price adjustments"):
        PriceDataBundle(prices={}, metadata=meta, missing_symbols=[], source="demo", adjustment="demo-synthetic")


def test_demo_provider_serves_in_memory_universe():
    universe = {
        "spy": pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"])),
        "AAA": pd.Series([10.0, 11.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"])),
    }
    bundle = DemoPriceProvider(universe).get_history(["SPY", "AAA"], START, END)

    assert set(bundle.prices) == {"SPY", "AAA"}
    assert bundle.source == "demo"
    assert bundle.adjustment == "demo-synthetic"


def test_assemble_bundle_slices_to_requested_window():
    universe = {
        "AAA": normalize_price_series(
            pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-01-01", "2026-01-10", "2026-02-01"]))
        )
    }
    bundle = assemble_bundle(
        universe=universe,
        symbols=["AAA"],
        start=pd.Timestamp("2026-01-01"),
        end=pd.Timestamp("2026-01-15"),
        source="demo",
        adjustment="demo-synthetic",
    )
    assert list(bundle.prices["AAA"].index) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10")]


def test_bundle_series_are_consumable_by_the_engine():
    bundle = CSVPriceProvider(CLEAN).get_history(["SPY", "AAA"], START, END)
    summary = abnormal_return_summary(
        ticker_prices=bundle.prices["AAA"],
        benchmark_prices=bundle.prices["SPY"],
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-05"),
    )
    assert "combined_abnormal_return" in summary
    assert summary["raw_return"] is not None
