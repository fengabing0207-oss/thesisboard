from __future__ import annotations

from collections import defaultdict
from statistics import mean, median


TRADEABLE_CLASSES = {"Tradeable"}
CHASE_CLASSES = {"Avoid Chase"}


def compute_cohort_base_rate(
    universe_cohort: list[dict],
    *,
    created_dates: set[str],
    horizon_days: int,
) -> float | None:
    outcomes = [
        float(record["forward_abnormal_return"]) > 0
        for record in universe_cohort
        if _date_key(record.get("created_at")) in created_dates
        and int(record.get("horizon_days")) == int(horizon_days)
        and record.get("forward_abnormal_return") is not None
    ]
    return _rate(outcomes)


def evaluate_signal_records(records: list[dict], universe_cohort: list[dict] | None = None) -> dict:
    matured = [
        record
        for record in records
        if record.get("is_matured") and record.get("forward_abnormal_return") is not None
    ]
    abnormal_returns = [float(record["forward_abnormal_return"]) for record in matured]
    groups = _group_metrics(matured, universe_cohort or [])
    false_positives = [record for record in matured if record.get("classification") in TRADEABLE_CLASSES and record.get("trade_hit") is False]
    false_negatives = [record for record in matured if record.get("false_negative") is True]

    return {
        "sample_size": len(matured),
        "sample_positive_rate": _rate([record["forward_abnormal_return"] > 0 for record in matured]),
        "groups": groups,
        "average_forward_abnormal_return": None if not abnormal_returns else mean(abnormal_returns),
        "median_forward_abnormal_return": None if not abnormal_returns else median(abnormal_returns),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "avoid_chase_count": sum(1 for record in matured if record.get("classification") in CHASE_CLASSES),
    }


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _group_metrics(records: list[dict], universe_cohort: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(int(record["horizon_days"]), record["classification"])].append(record)

    metrics = []
    for (horizon_days, classification), group in sorted(grouped.items()):
        created_dates = {_date_key(record.get("created_at")) for record in group}
        base_rate = compute_cohort_base_rate(
            universe_cohort,
            created_dates=created_dates,
            horizon_days=horizon_days,
        )
        trade_hits = [bool(record["trade_hit"]) for record in group if record.get("trade_hit") is not None]
        avoided_bad_trades = [
            bool(record["avoided_bad_trade"]) for record in group if record.get("avoided_bad_trade") is not None
        ]
        sample_positive_rate = _rate([record["forward_abnormal_return"] > 0 for record in group])
        trade_hit_rate = _rate(trade_hits)
        metrics.append(
            {
                "horizon_days": horizon_days,
                "classification": classification,
                "sample_size": len(group),
                "sample_positive_rate": sample_positive_rate,
                "base_rate": base_rate,
                "trade_hit_rate": trade_hit_rate,
                "excess_trade_hit_rate": None if trade_hit_rate is None or base_rate is None else trade_hit_rate - base_rate,
                "avoided_bad_trade_rate": _rate(avoided_bad_trades),
                "false_negatives": sum(1 for record in group if record.get("false_negative") is True),
                "average_forward_abnormal_return": mean(float(record["forward_abnormal_return"]) for record in group),
            }
        )
    return metrics


def _date_key(value) -> str | None:
    if value is None:
        return None
    return str(value)[:10]
