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


class YFinancePriceProvider:
    """Fetches split/dividend-adjusted EOD closes from Yahoo Finance (PR #15B).

    Uses ``auto_adjust=True`` so the returned ``Close`` is already adjusted; the
    declared ``adjustment`` is ``"auto_adjust"``. A single instance is meant to
    serve every symbol in one validation run (signal tickers, the benchmark, and
    sector ETFs), so the bundle's single-source / single-adjustment invariant
    holds automatically.

    LOOKBACK REMINDER: with real data, request ``start`` at least ~120 trading
    days BEFORE the earliest signal date. With less history, rolling beta cannot
    be estimated and silently falls back to 1.0, distorting abnormal returns
    without raising. Automatic detection of this is deferred to PR #15C.

    ``yfinance`` is imported lazily inside :meth:`_fetch_close_frames` so this
    module stays importable (and the offline test suite stays green) without the
    dependency present.
    """

    name = "yfinance"
    adjustment = "auto_adjust"

    def get_history(self, symbols: Sequence[str], start, end) -> PriceDataBundle:
        requested = [str(symbol).strip().upper() for symbol in symbols]
        raw = self._fetch_close_frames(requested, start, end)
        universe = {
            str(symbol).strip().upper(): normalize_price_series(series)
            for symbol, series in raw.items()
        }
        return assemble_bundle(
            universe=universe,
            symbols=requested,
            start=start,
            end=end,
            source=self.name,
            adjustment=self.adjustment,
        )

    def _fetch_close_frames(self, symbols: Sequence[str], start, end) -> dict[str, pd.Series]:
        """Return ``{symbol: adjusted-close Series}`` from yfinance.

        This is the network seam; offline tests monkeypatch it, and the only
        test that exercises the real call is marked ``network`` and skipped by
        default. yfinance's ``end`` is exclusive, so it is bumped by one day.
        """
        import yfinance  # lazy import; see class docstring

        requested = [str(symbol).strip().upper() for symbol in symbols]
        frame = yfinance.download(
            tickers=requested,
            start=pd.Timestamp(start).normalize().date().isoformat(),
            end=(pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )

        frames: dict[str, pd.Series] = {}
        if frame is None or frame.empty:
            return frames

        columns = frame.columns
        for symbol in requested:
            close = None
            if isinstance(columns, pd.MultiIndex):
                if symbol in columns.get_level_values(0):
                    block = frame[symbol]
                    if "Close" in block.columns:
                        close = block["Close"]
                elif "Close" in columns.get_level_values(0):
                    close_block = frame["Close"]
                    if symbol in getattr(close_block, "columns", []):
                        close = close_block[symbol]
            elif "Close" in columns:
                close = frame["Close"]

            if close is not None:
                close = close.dropna()
                if not close.empty:
                    frames[symbol] = close
        return frames


def _safe_path_part(value: str) -> str:
    """Make a string safe to use as a single path component."""
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in str(value)) or "_"


class CachingPriceProvider:
    """Transparent cache-first decorator over any :class:`PriceProvider`.

    Cache identity is ``(symbol, start, end, snapshot_id)``. A hit requires an
    exact match on all four — in particular a cached window will NOT serve a
    request for a different window. ``snapshot_id`` pins a pull: reusing the same
    id makes a backtest reproducible even though the upstream source may
    re-adjust history over time; an unpinned provider defaults to the fetch date,
    so distinct days are distinct, non-overwriting snapshots.

    The cache is also partitioned by the inner provider's source and adjustment
    convention, so the same ``cache_dir`` / ``snapshot_id`` / symbol / window can
    never replay one source's data under a different provider (e.g. a demo cache
    served as ``source="yfinance"`` / ``adjustment="auto_adjust"``).

    Snapshots are stored as per-symbol long-format CSVs::

        <cache_dir>/<source>/<adjustment>/<snapshot_id>/<SYMBOL>__<start>__<end>.csv

    which are themselves replayable offline via :class:`CSVPriceProvider`.
    """

    def __init__(self, inner, cache_dir: Path | str, *, snapshot_id: str | None = None):
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._snapshot_id = snapshot_id
        self.name = inner.name
        self.adjustment = inner.adjustment

    def get_history(self, symbols: Sequence[str], start, end) -> PriceDataBundle:
        snapshot_id = self._resolve_snapshot_id()
        start_key = pd.Timestamp(start).normalize().date().isoformat()
        end_key = pd.Timestamp(end).normalize().date().isoformat()

        requested: list[str] = []
        seen: set[str] = set()
        for raw_symbol in symbols:
            symbol = str(raw_symbol).strip().upper()
            if symbol not in seen:
                seen.add(symbol)
                requested.append(symbol)

        universe: dict[str, pd.Series] = {}
        misses: list[str] = []
        for symbol in requested:
            path = self._cache_file(snapshot_id, symbol, start_key, end_key)
            if path.exists():
                universe[symbol] = self._read_cache(path, symbol)
            else:
                misses.append(symbol)

        if misses:
            fetched = self._inner.get_history(misses, start, end)
            for symbol in misses:
                series = fetched.prices.get(symbol)
                if series is None or series.empty:
                    continue  # do not cache misses; they re-fetch next time
                self._write_cache(self._cache_file(snapshot_id, symbol, start_key, end_key), symbol, series)
                universe[symbol] = series

        return assemble_bundle(
            universe=universe,
            symbols=requested,
            start=start,
            end=end,
            source=self.name,
            adjustment=self.adjustment,
        )

    def _resolve_snapshot_id(self) -> str:
        if self._snapshot_id is not None:
            return str(self._snapshot_id)
        from datetime import date

        return date.today().isoformat()

    def _cache_file(self, snapshot_id: str, symbol: str, start_key: str, end_key: str) -> Path:
        return (
            self._cache_dir
            / _safe_path_part(self.name)
            / _safe_path_part(self.adjustment)
            / snapshot_id
            / f"{symbol}__{start_key}__{end_key}.csv"
        )

    def _read_cache(self, path: Path, symbol: str) -> pd.Series:
        frame = pd.read_csv(path)
        series = pd.Series(frame["adjusted_close"].to_numpy(), index=frame["date"].to_numpy(), name=symbol)
        return normalize_price_series(series)

    def _write_cache(self, path: Path, symbol: str, series: pd.Series) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": [pd.Timestamp(value).date().isoformat() for value in series.index],
                "adjusted_close": series.to_numpy(),
            }
        )
        out.to_csv(path, index=False)


def get_price_provider(
    name: str = "demo",
    *,
    path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    snapshot_id: str | None = None,
):
    """Opt-in factory for price providers; defaults to the offline demo provider.

    Pass ``cache_dir`` to wrap the chosen provider in a :class:`CachingPriceProvider`.
    Only ``"demo"`` requires no configuration; ``"csv"`` needs ``path=``.
    """
    key = (name or "demo").lower()
    if key == "demo":
        from .demo_validation_data import default_demo_provider

        provider = default_demo_provider()
    elif key == "csv":
        if path is None:
            raise ValueError("csv provider requires path=")
        provider = CSVPriceProvider(path)
    elif key == "yfinance":
        provider = YFinancePriceProvider()
    else:
        raise ValueError(f"unknown price provider: {name!r}")

    if cache_dir is not None:
        provider = CachingPriceProvider(provider, cache_dir, snapshot_id=snapshot_id)
    return provider
