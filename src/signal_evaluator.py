from __future__ import annotations

from statistics import mean, median


TRADEABLE_CLASSES = {"Tradeable"}
CHASE_CLASSES = {"Avoid Chase"}


def evaluate_signal_records(records: list[dict]) -> dict:
    matured = [
        record
        for record in records
        if record.get("is_matured") and record.get("forward_abnormal_return") is not None
    ]
    base_outcomes = [record["forward_abnormal_return"] > 0 for record in matured]
    base_rate = _rate(base_outcomes)

    tradeable = [record for record in matured if record.get("classification") in TRADEABLE_CLASSES]
    tradeable_hits = [bool(record.get("hit")) for record in tradeable if record.get("hit") is not None]
    hit_rate = _rate(tradeable_hits)

    abnormal_returns = [float(record["forward_abnormal_return"]) for record in matured]
    false_positives = [record for record in tradeable if float(record["forward_abnormal_return"]) <= 0]
    false_negatives = [
        record
        for record in matured
        if record.get("classification") not in TRADEABLE_CLASSES | CHASE_CLASSES
        and float(record["forward_abnormal_return"]) > 0
    ]
    avoid_chase = [record for record in matured if record.get("classification") in CHASE_CLASSES]

    return {
        "sample_size": len(matured),
        "tradeable_sample_size": len(tradeable),
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "excess_hit_rate": None if hit_rate is None or base_rate is None else hit_rate - base_rate,
        "average_forward_abnormal_return": None if not abnormal_returns else mean(abnormal_returns),
        "median_forward_abnormal_return": None if not abnormal_returns else median(abnormal_returns),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "avoid_chase_count": len(avoid_chase),
    }


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)
