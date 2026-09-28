"""Collect a point-in-time headline snapshot for a ticker watchlist."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_collector import collect_ticker_news
from src.news_store import DB_PATH, list_collection_runs, storage_backend_info


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
    parser.add_argument(
        "--min-interval-minutes",
        type=int,
        default=0,
        help="Skip collection when the latest comparable successful run is newer than this",
    )
    args = parser.parse_args()
    if args.min_interval_minutes < 0:
        parser.error("--min-interval-minutes must be non-negative")
    if args.require_durable and not storage_backend_info(args.db)[
        "durable_for_scheduled_runs"
    ]:
        parser.error(
            "scheduled collection requires THESISBOARD_DATABASE_URL; "
            "local SQLite is not durable"
        )
    tickers = tuple(args.tickers) or _configured_watchlist()
    fresh_run = _latest_fresh_collection(
        tickers,
        db_path=args.db,
        min_interval_minutes=args.min_interval_minutes,
    )
    if fresh_run is not None:
        print(
            json.dumps(
                {
                    "status": "skipped_fresh",
                    "requested_tickers": list(tickers),
                    "successful_tickers": fresh_run["successful_tickers"],
                    "failed_tickers": fresh_run["failed_tickers"],
                    "errors": fresh_run["errors"],
                    "fetched_items": 0,
                    "inserted_versions": 0,
                    "completed_at": fresh_run["completed_at"],
                    "run_id": fresh_run["id"],
                    "age_minutes": fresh_run["age_minutes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = collect_ticker_news(tickers, db_path=args.db)
    printable = {key: value for key, value in result.items() if key != "per_ticker"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 1 if result["failed_tickers"] else 0


def _latest_fresh_collection(
    tickers: tuple[str, ...],
    *,
    db_path: Path | str = DB_PATH,
    min_interval_minutes: int,
    now: datetime | None = None,
) -> dict | None:
    if min_interval_minutes <= 0:
        return None
    runs = list_collection_runs(db_path=db_path, limit=1)
    if not runs:
        return None
    latest = runs[0]
    expected = len(tickers)
    if (
        latest["requested_tickers"] != expected
        or latest["successful_tickers"] != expected
        or latest["failed_tickers"] != 0
        or latest["errors"]
    ):
        return None
    completed_at = datetime.fromisoformat(
        str(latest["completed_at"]).replace("Z", "+00:00")
    )
    current = now or datetime.now(timezone.utc)
    age_minutes = max(0.0, (current - completed_at).total_seconds() / 60)
    if age_minutes >= min_interval_minutes:
        return None
    return {**latest, "age_minutes": round(age_minutes, 1)}


def _configured_watchlist() -> tuple[str, ...]:
    raw = os.getenv("THESISBOARD_WATCHLIST", "").replace(",", " ")
    values = tuple(value.strip().upper() for value in raw.split() if value.strip())
    return values or DEFAULT_WATCHLIST


if __name__ == "__main__":
    raise SystemExit(main())
