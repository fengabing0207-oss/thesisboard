"""Append a human approval or rejection for one proposed StrategySpec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research_automation import record_strategy_promotion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_event_id", type=int)
    parser.add_argument("decision", choices=("approved", "rejected"))
    parser.add_argument("--decided-by", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    result = record_strategy_promotion(
        args.candidate_event_id,
        decision=args.decision,
        decided_by=args.decided_by,
        note=args.note,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
