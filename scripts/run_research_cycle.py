"""Run one auditable News Signal Lab collection/calibration cycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_store import storage_backend_info
from src.research_automation import ResearchCycleConfig, run_research_cycle


DEFAULT_WATCHLIST = ("NVDA", "AVGO", "MRVL", "MU", "VRT", "ORCL", "INTC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Ticker symbols separated by spaces")
    parser.add_argument("--horizon", type=int, default=1, help="Forward benchmark sessions")
    parser.add_argument("--min-train-rows", type=int, default=30)
    parser.add_argument("--min-train-sessions", type=int, default=10)
    parser.add_argument(
        "--require-durable",
        action="store_true",
        help="Refuse local SQLite; intended for scheduled runners",
    )
    args = parser.parse_args()
    tickers = tuple(args.tickers) or _configured_watchlist()
    backend = storage_backend_info()
    if args.require_durable and not backend["durable_for_scheduled_runs"]:
        parser.error(
            "scheduled runs require THESISBOARD_DATABASE_URL; local SQLite is not durable"
        )
    try:
        result = run_research_cycle(
            tickers,
            config=ResearchCycleConfig(
                horizon_days=args.horizon,
                min_train_rows=args.min_train_rows,
                min_train_sessions=args.min_train_sessions,
            ),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": "Research cycle failed; inspect the append-only audit log.",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 1 if result["status"] == "failed" else 0


def _configured_watchlist() -> tuple[str, ...]:
    raw = os.getenv("THESISBOARD_WATCHLIST", "").replace(",", " ")
    values = tuple(value.strip().upper() for value in raw.split() if value.strip())
    return values or DEFAULT_WATCHLIST


if __name__ == "__main__":
    raise SystemExit(main())
