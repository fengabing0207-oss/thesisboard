from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_validation_data import prepare_validation_lab_data


def main() -> None:
    db_path = ROOT / "data" / "validation_demo.db"
    lab = prepare_validation_lab_data(db_path=db_path)
    metrics = lab["metrics"]

    print("ThesisBoard validation demo")
    print(f"records: {metrics['sample_size']}")
    tradeable_group = next(
        (group for group in metrics["groups"] if group["classification"] == "Tradeable"),
        None,
    )
    print(f"tradeable hit rate: {None if tradeable_group is None else tradeable_group['trade_hit_rate']}")
    print(f"cohort base rate: {None if tradeable_group is None else tradeable_group['base_rate']}")
    print(f"excess trade hit rate: {None if tradeable_group is None else tradeable_group['excess_trade_hit_rate']}")
    print(f"avg forward abnormal return: {metrics['average_forward_abnormal_return']}")
    print(f"median forward abnormal return: {metrics['median_forward_abnormal_return']}")
    print(f"false positives: {metrics['false_positives']}")
    print(f"false negatives: {metrics['false_negatives']}")
    print(f"avoid chase records: {metrics['avoid_chase_count']}")


if __name__ == "__main__":
    main()
