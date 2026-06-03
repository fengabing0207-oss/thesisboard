from src.signal_evaluator import compute_cohort_base_rate, evaluate_signal_records


def test_metrics_use_cohort_base_rate_and_split_hit_semantics():
    records = [
        {"created_at": "2026-01-02", "horizon_days": 5, "classification": "Tradeable", "forward_abnormal_return": 0.04, "trade_hit": True, "is_matured": True},
        {"created_at": "2026-01-02", "horizon_days": 5, "classification": "Tradeable", "forward_abnormal_return": -0.02, "trade_hit": False, "is_matured": True},
        {"created_at": "2026-01-02", "horizon_days": 5, "classification": "Avoid", "forward_abnormal_return": 0.03, "avoided_bad_trade": False, "false_negative": True, "is_matured": True},
        {"created_at": "2026-01-02", "horizon_days": 5, "classification": "Avoid Chase", "forward_abnormal_return": -0.05, "avoided_bad_trade": True, "false_negative": False, "is_matured": True},
    ]
    universe_cohort = [
        {"created_at": "2026-01-02", "horizon_days": 5, "ticker": "A", "forward_abnormal_return": 0.01},
        {"created_at": "2026-01-02", "horizon_days": 5, "ticker": "B", "forward_abnormal_return": -0.01},
        {"created_at": "2026-01-02", "horizon_days": 20, "ticker": "C", "forward_abnormal_return": 0.02},
    ]

    metrics = evaluate_signal_records(records, universe_cohort=universe_cohort)
    tradeable_group = next(group for group in metrics["groups"] if group["classification"] == "Tradeable")
    avoid_chase_group = next(group for group in metrics["groups"] if group["classification"] == "Avoid Chase")
    assert metrics["sample_positive_rate"] == 0.5
    assert "base_rate" not in metrics
    assert tradeable_group["trade_hit_rate"] == 0.5
    assert tradeable_group["base_rate"] == 0.5
    assert tradeable_group["excess_trade_hit_rate"] == 0
    assert avoid_chase_group["avoided_bad_trade_rate"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["avoid_chase_count"] == 1


def test_cohort_base_rate_uses_same_date_and_horizon():
    cohort = [
        {"created_at": "2026-01-02", "horizon_days": 5, "forward_abnormal_return": 0.02},
        {"created_at": "2026-01-02", "horizon_days": 5, "forward_abnormal_return": -0.01},
        {"created_at": "2026-01-03", "horizon_days": 5, "forward_abnormal_return": 0.04},
        {"created_at": "2026-01-02", "horizon_days": 20, "forward_abnormal_return": 0.04},
    ]

    assert compute_cohort_base_rate(cohort, created_dates={"2026-01-02"}, horizon_days=5) == 0.5
