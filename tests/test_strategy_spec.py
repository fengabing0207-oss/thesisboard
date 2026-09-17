import pytest

from src.strategy_spec import StrategySpec, audit_calculation_parity


def test_strategy_spec_id_is_canonical_across_universe_order_and_case():
    first = StrategySpec(
        model_name="logistic",
        model_version="tfidf-logit.v1",
        probability_threshold=0.7,
        holding_horizon_sessions=3,
        universe=("nvda", "AAPL"),
    )
    second = StrategySpec(
        model_name="logistic",
        model_version="tfidf-logit.v1",
        probability_threshold=0.7,
        holding_horizon_sessions=3,
        universe=("AAPL", "NVDA", "AAPL"),
    )

    assert first.spec_id == second.spec_id
    assert first.to_dict()["universe"] == ["AAPL", "NVDA"]


def test_strategy_spec_rejects_rules_the_engine_does_not_support():
    with pytest.raises(ValueError, match="unsupported weighting_rule"):
        StrategySpec(
            model_name="logistic",
            model_version="tfidf-logit.v1",
            probability_threshold=0.7,
            holding_horizon_sessions=1,
            universe=("NVDA",),
            weighting_rule="volatility_weighted",
        )


def test_strategy_spec_round_trips_from_recorded_dictionary():
    original = StrategySpec(
        model_name="logistic",
        model_version="tfidf-logit.v1",
        probability_threshold=0.65,
        holding_horizon_sessions=1,
        universe=("NVDA", "AAPL"),
    )

    restored = StrategySpec.from_dict(original.to_dict())

    assert restored == original
    assert restored.spec_id == original.spec_id
    with pytest.raises(ValueError, match="unknown StrategySpec fields"):
        StrategySpec.from_dict({**original.to_dict(), "future_rule": "unsafe"})


def test_calculation_parity_requires_same_spec_and_engine():
    base = {
        "status": "ok",
        "strategy_spec_id": "spec:123",
        "backtest_version": "engine.v1",
        "calculation_contract_version": "contract.v1",
    }

    verified = audit_calculation_parity({"validation": base, "test": dict(base)})
    changed = dict(base, strategy_spec_id="spec:456")
    mismatch = audit_calculation_parity({"validation": base, "test": changed})

    assert verified["status"] == "verified"
    assert mismatch["status"] == "mismatch"
    assert "strategy_spec_id differs across runs" in mismatch["mismatches"]
