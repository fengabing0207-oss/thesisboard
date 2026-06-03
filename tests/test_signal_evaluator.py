from src.signal_evaluator import evaluate_signal_records


def test_hit_rate_base_rate_and_excess_hit_rate_are_reported_together():
    records = [
        {"classification": "Tradeable", "forward_abnormal_return": 0.04, "hit": True, "is_matured": True},
        {"classification": "Tradeable", "forward_abnormal_return": -0.02, "hit": False, "is_matured": True},
        {"classification": "Avoid", "forward_abnormal_return": 0.03, "hit": False, "is_matured": True},
        {"classification": "Avoid Chase", "forward_abnormal_return": 0.05, "hit": False, "is_matured": True},
    ]

    metrics = evaluate_signal_records(records)
    assert metrics["hit_rate"] == 0.5
    assert metrics["base_rate"] == 0.75
    assert metrics["excess_hit_rate"] == -0.25
    assert metrics["average_forward_abnormal_return"] == 0.025
    assert metrics["median_forward_abnormal_return"] == 0.035
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["avoid_chase_count"] == 1
