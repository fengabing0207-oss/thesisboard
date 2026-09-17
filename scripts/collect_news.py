"""Collect a point-in-time headline snapshot for a ticker watchlist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_collector import collect_ticker_news
from src.news_store import DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, separated by spaces")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite store path")
    args = parser.parse_args()
    result = collect_ticker_news(args.tickers, db_path=args.db)
    printable = {key: value for key, value in result.items() if key != "per_ticker"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 1 if result["failed_tickers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
