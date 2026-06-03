from src.demo_validation_data import (
    DEPRECATED_OUTCOME_FIELD,
    EXPLICIT_OUTCOME_FIELDS,
    build_demo_price_data,
    build_demo_signal_inputs,
    build_demo_universe_cohort,
    build_demo_validation_database,
    prepare_validation_lab_data,
)

GROUP_KEYS = {
    "horizon_days",
    "classification",
    "sample_size",
    "sample_positive_rate",
    "base_rate",
    "trade_hit_rate",
    "excess_trade_hit_rate",
    "watch_followthrough_rate",
    "avoided_bad_trade_rate",
    "false_negatives",
    "average_forward_abnormal_return",
}


def test_build_demo_price_data_shares_one_business_day_index():
    data = build_demo_price_data()
    index = data["dates"]
    assert len(index) == 12
    assert {"NVDA", "UAL", "CRWD", "MSFT", "DAL"} <= set(data["prices_by_ticker"])
    assert {"SMH", "JETS", "HACK"} <= set(data["sector_prices"])
    for series in data["prices_by_ticker"].values():
        assert series.index.equals(index)
    assert data["benchmark_prices"].index.equals(index)


def test_build_demo_signal_inputs_have_required_fields():
    inputs = build_demo_signal_inputs()
    assert [item["ticker"] for item in inputs] == ["NVDA", "UAL", "CRWD"]
    for item in inputs:
        assert {"ticker", "theme", "sentiment", "metrics", "score"} <= set(item)


def test_build_demo_validation_database_runs_core_and_omits_deprecated_hit(tmp_path):
    db_path = tmp_path / "demo.db"
    built = build_demo_validation_database(db_path)

    assert set(built) == {"price_data", "evaluation"}
    evaluation = built["evaluation"]
    assert len(evaluation) == len(build_demo_signal_inputs())
    for record in evaluation:
        assert DEPRECATED_OUTCOME_FIELD not in record
        assert set(EXPLICIT_OUTCOME_FIELDS) <= set(record)
        assert "data_quality_flag" in record
        assert "beta_fallback_used" in record


def test_build_demo_universe_cohort_covers_full_universe():
    data = build_demo_price_data()
    cohort = build_demo_universe_cohort(
        created_at=data["dates"][0],
        horizon_days=5,
        prices_by_ticker=data["prices_by_ticker"],
        benchmark_prices=data["benchmark_prices"],
        sector_prices=data["sector_prices"],
    )
    assert {record["ticker"] for record in cohort} == set(data["prices_by_ticker"])
    for record in cohort:
        assert record["horizon_days"] == 5
        assert record["forward_abnormal_return"] is not None


def test_prepare_validation_lab_data_shape_with_explicit_db(tmp_path):
    lab = prepare_validation_lab_data(db_path=tmp_path / "lab.db")

    assert set(lab) == {"snapshots", "outcomes", "metrics", "cohort"}
    assert lab["snapshots"], "expected recorded snapshots"
    assert lab["outcomes"], "expected matured outcomes"

    metrics = lab["metrics"]
    assert metrics["sample_size"] == len(lab["outcomes"])
    assert isinstance(metrics["groups"], list) and metrics["groups"]
    for group in metrics["groups"]:
        assert GROUP_KEYS <= set(group)


def test_lab_data_uses_explicit_outcomes_not_deprecated_hit(tmp_path):
    lab = prepare_validation_lab_data(db_path=tmp_path / "lab.db")

    for record in lab["snapshots"] + lab["outcomes"]:
        assert DEPRECATED_OUTCOME_FIELD not in record
    for record in lab["outcomes"]:
        assert set(EXPLICIT_OUTCOME_FIELDS) <= set(record)


def test_prepare_validation_lab_data_defaults_to_throwaway_db():
    lab = prepare_validation_lab_data()
    assert lab["snapshots"]
    assert lab["metrics"]["sample_size"] >= 1


def test_tradeable_group_compares_hit_rate_to_cohort_base_rate(tmp_path):
    lab = prepare_validation_lab_data(db_path=tmp_path / "lab.db")
    tradeable = next(
        (group for group in lab["metrics"]["groups"] if group["classification"] == "Tradeable"),
        None,
    )
    assert tradeable is not None
    assert tradeable["base_rate"] is not None
    expected_excess = tradeable["trade_hit_rate"] - tradeable["base_rate"]
    assert tradeable["excess_trade_hit_rate"] == expected_excess
