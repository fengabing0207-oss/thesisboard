"""Collect a point-in-time headline snapshot for a ticker watchlist."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_collector import collect_ticker_news
from src.news_store import DB_PATH, storage_backend_info


DEFAULT_WATCHLIST = ("NVDA", "AVGO", "MRVL", "MU", "VRT", "ORCL", "INTC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Ticker symbols, separated by spaces")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite store path")
    parser.add_argument(
        "--require-durable",
        action="store_true",
        help="Refuse local SQLite; intended for scheduled runners",
    )
    args = parser.parse_args()
    if args.require_durable and not storage_backend_info(args.db)[
        "durable_for_scheduled_runs"
    ]:
        parser.error(
            "scheduled collection requires THESISBOARD_DATABASE_URL; "
            "local SQLite is not durable"
        )
    tickers = tuple(args.tickers) or _configured_watchlist()
    result = collect_ticker_news(tickers, db_path=args.db)
    printable = {key: value for key, value in result.items() if key != "per_ticker"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 1 if result["failed_tickers"] else 0


def _configured_watchlist() -> tuple[str, ...]:
    raw = os.getenv("THESISBOARD_WATCHLIST", "").replace(",", " ")
    values = tuple(value.strip().upper() for value in raw.split() if value.strip())
    return values or DEFAULT_WATCHLIST


if __name__ == "__main__":
    raise SystemExit(main())
