"""Price-provider boundary for the ThesisBoard validation spine (PR #15A).

This module introduces the seam between *where prices come from* and the
validation core that consumes them. The core only ever sees ``pd.Series`` of
adjusted-close prices, so a provider's job is narrow: turn a set of symbols and
a date range into a :class:`PriceDataBundle` of normalized series plus
provenance metadata.

PR #15A is deliberately offline: it ships only the in-memory ``DemoPriceProvider``
(default, preserves existing app behavior) and a local ``CSVPriceProvider`` that
reads adjusted-close prices from a long-format file. Online EOD sources
(yfinance / Stooq / Alpha Vantage) are left to PR #15B and slot in behind this
same interface.

Two contracts are enforced here:

1. **Adjusted-close-like prices.** Every provider declares an ``adjustment``
   string recorded in metadata. Local files are trusted as ``"as_provided"``.
2. **Same source / same adjustment per run.** A :class:`PriceDataBundle`
   validates that every series it carries shares the bundle's declared source
   and adjustment, so a single validation run can never silently mix providers
   or adjustment conventions (which would corrupt abnormal-return math).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class PriceSeriesMetadata:
    """Provenance and coverage for a single symbol's price series.

    The reserved fields at the bottom are intentionally inert in PR #15A; they
    exist so PR #15B can record stale-price / missing-session / survivorship
    signals without changing this interface. Note that ``beta_fallback_used`` is
    deliberately NOT here: beta fallback is engine/evaluation quality, not a
    property of the price data, so it stays on the evaluation record.
    """

    symbol: str
    source: str
    adjustment: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    actual_start: pd.Timestamp | None
    actual_end: pd.Timestamp | None
    data_quality_flags: list[str] = field(default_factory=list)

    # Reserved for PR #15B (online sources). Left empty/None in PR #15A.
    expected_sessions: int | None = None
    missing_session_dates: list = field(default_factory=list)
    stale_price_dates: list = field(default_factory=list)
    survivorship_verified: bool | None = None


@dataclass(frozen=True)
class PriceDataBundle:
    """Normalized price series for a request, plus per-symbol metadata.

    ``prices`` contains only symbols that resolved to data; symbols that could
    not be served appear in ``missing_symbols`` (and carry metadata with
    ``actual_start``/``actual_end`` set to None). ``source`` and ``adjustment``
    are validated to be consistent across every series — see __post_init__.
    """

    prices: dict[str, pd.Series]
    metadata: dict[str, PriceSeriesMetadata]
    missing_symbols: list[str]
    source: str
    adjustment: str

    def __post_init__(self) -> None:
        sources = {meta.source for meta in self.metadata.values()}
        if sources - {self.source}:
            raise ValueError(
                f"bundle mixes price sources {sorted(sources)} against declared {self.source!r}; "
                "a single validation run must use one source"
            )
        adjustments = {meta.adjustment for meta in self.metadata.values()}
        if adjustments - {self.adjustment}:
            raise ValueError(
                f"bundle mixes price adjustments {sorted(adjustments)} against declared {self.adjustment!r}; "
                "a single validation run must use one adjustment convention"
            )


@runtime_checkable
class PriceProvider(Protocol):
    """A source of adjusted-close price history.

    Implementations must return a :class:`PriceDataBundle` whose series are
    normalized (tz-naive, ascending, de-duplicated, NaN-dropped, float).
    """

    name: str
    adjustment: str

    def get_history(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> PriceDataBundle: ...


def normalize_price_series(series: pd.Series) -> pd.Series:
    """Return a tz-naive, ascending, de-duplicated, NaN-free float price series.

    Dates are coerced to UTC then flattened to naive midnight so mixed
    tz-aware / tz-naive / non-ISO inputs land on one consistent calendar. Values
    that cannot be parsed as numbers are dropped. On duplicate dates the last
    row wins.
    """
    raw = pd.Series(series)
    # format="mixed" parses per-element so mixed ISO / tz-aware / non-ISO dates
    # each resolve instead of being coerced to NaT by a single inferred format.
    index = pd.to_datetime(pd.Index(raw.index), errors="coerce", utc=True, format="mixed")
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    index = index.normalize()

    values = pd.to_numeric(pd.Series(list(raw.values)), errors="coerce")
    cleaned = pd.Series(values.to_numpy(dtype="float64"), index=index, name=raw.name)
    cleaned = cleaned[~cleaned.index.isna()]
    cleaned = cleaned.dropna()
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned


def assemble_bundle(
    *,
    universe: dict[str, pd.Series],
    symbols: Sequence[str],
    start,
    end,
    source: str,
    adjustment: str,
) -> PriceDataBundle:
    """Build a bundle by slicing a normalized ``universe`` to the requested window.

    ``universe`` is expected to be keyed by uppercase symbol with already
    normalized series. Requested symbols are uppercased and de-duplicated;
    anything with no data in range is reported via ``missing_symbols``.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    prices: dict[str, pd.Series] = {}
    metadata: dict[str, PriceSeriesMetadata] = {}
    missing: list[str] = []
    seen: set[str] = set()

    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if symbol in seen:
            continue
        seen.add(symbol)

        series = universe.get(symbol)
        sliced = None
        if series is not None and not series.empty:
            sliced = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]

        if sliced is None or sliced.empty:
            missing.append(symbol)
            metadata[symbol] = PriceSeriesMetadata(
                symbol=symbol,
                source=source,
                adjustment=adjustment,
                requested_start=start_ts,
                requested_end=end_ts,
                actual_start=None,
                actual_end=None,
            )
            continue

        prices[symbol] = sliced
        metadata[symbol] = PriceSeriesMetadata(
            symbol=symbol,
            source=source,
            adjustment=adjustment,
            requested_start=start_ts,
            requested_end=end_ts,
            actual_start=pd.Timestamp(sliced.index.min()),
            actual_end=pd.Timestamp(sliced.index.max()),
        )

    return PriceDataBundle(
        prices=prices,
        metadata=metadata,
        missing_symbols=missing,
        source=source,
        adjustment=adjustment,
    )


class DemoPriceProvider:
    """Serves prices from an in-memory, already-synthetic universe.

    This is the default provider; it wraps the existing demo data so the app's
    behavior is unchanged. It does no I/O and never touches the network.
    """

    name = "demo"
    adjustment = "demo-synthetic"

    def __init__(self, universe: dict[str, pd.Series]):
        self._universe = {
            str(symbol).strip().upper(): normalize_price_series(series)
            for symbol, series in universe.items()
        }

    def get_history(self, symbols: Sequence[str], start, end) -> PriceDataBundle:
        return assemble_bundle(
            universe=self._universe,
            symbols=symbols,
            start=start,
            end=end,
            source=self.name,
            adjustment=self.adjustment,
        )


class CSVPriceProvider:
    """Reads adjusted-close prices from a single long-format CSV.

    Expected columns: ``symbol``, ``date`` (ISO ``YYYY-MM-DD``), ``adjusted_close``.
    One file holds many symbols. Symbols are uppercased on read. This is the
    first path by which real (non-synthetic) prices can flow into the engine,
    entirely from local files — no network.
    """

    name = "local-csv"
    adjustment = "as_provided"
    REQUIRED_COLUMNS = ("symbol", "date", "adjusted_close")

    def __init__(self, path: Path | str, *, name: str | None = None, adjustment: str | None = None):
        self.path = Path(path)
        if name is not None:
            self.name = name
        if adjustment is not None:
            self.adjustment = adjustment

    def _load_frame(self) -> pd.DataFrame:
        frame = pd.read_csv(self.path)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV {self.path} is missing required columns: {missing}")
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        return frame

    def get_history(self, symbols: Sequence[str], start, end) -> PriceDataBundle:
        frame = self._load_frame()
        requested = {str(symbol).strip().upper() for symbol in symbols}
        universe: dict[str, pd.Series] = {}
        for symbol, group in frame.groupby("symbol"):
            if symbol not in requested:
                continue
            series = pd.Series(
                group["adjusted_close"].to_numpy(),
                index=group["date"].to_numpy(),
                name=symbol,
            )
            universe[symbol] = normalize_price_series(series)
        return assemble_bundle(
            universe=universe,
            symbols=symbols,
            start=start,
            end=end,
            source=self.name,
            adjustment=self.adjustment,
        )
