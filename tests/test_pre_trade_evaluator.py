from dataclasses import fields as dataclass_fields

import pytest

from src.pre_trade_check import (
    ThesisDecision,
    VerdictType,
    derive_risk_flags,
    evaluate_pre_trade_risk,
)


def _decision(**overrides):
    base = dict(
        ticker="NVDA",
        theme="AI Infrastructure",
        event_type="pullback",  # neutral by default; tests opt into event risk
        planned_action="buy",
        instrument_type="stock",
        horizon_days=20,
        market_expectation="medium",
        entry_thesis="Demand keeps beating.",
        risk_thesis="Guidance disappoints.",
        max_loss=0.0,
        invalidation_rule="Exit below the prior range low.",
        confidence="medium",
        position_size=5.0,
        notes="",
    )
    base.update(overrides)
    return ThesisDecision(**base)


def test_high_runup_runtime_param_sets_flag():
    decision = _decision()
    flags_true, _ = evaluate_pre_trade_risk(decision, high_runup=True)
    flags_false, _ = evaluate_pre_trade_risk(decision, high_runup=False)
    assert flags_true.high_runup is True
    assert flags_false.high_runup is False


def test_high_runup_is_not_a_schema_field():
    names = {f.name for f in dataclass_fields(ThesisDecision)}
    assert "high_runup" not in names
    assert "high_runup" not in _decision().to_dict()


def _has_missing_invalidation_prompts(reasons):
    text = " ".join(reasons).lower()
    return (
        "invalidation rule" in text
        and "reflection" in text
        and "not financial advice and not a recommended stop" in text
    )


def test_missing_invalidation_floors_to_wait_alone():
    flags, verdict = evaluate_pre_trade_risk(_decision(invalidation_rule=""))
    assert flags.no_invalidation_rule is True
    assert verdict.verdict is VerdictType.WAIT
    assert _has_missing_invalidation_prompts(verdict.reasons)


@pytest.mark.parametrize("placeholder", ["idk", "tbd", "?", "not sure", "  IDK ", "n/a", "none"])
def test_placeholder_invalidation_floors_to_wait(placeholder):
    flags, verdict = evaluate_pre_trade_risk(_decision(invalidation_rule=placeholder))
    assert flags.no_invalidation_rule is True
    assert verdict.verdict is VerdictType.WAIT
    assert _has_missing_invalidation_prompts(verdict.reasons)


def test_missing_invalidation_with_chase_escalates_to_avoid_chase_and_keeps_prompts():
    flags, verdict = evaluate_pre_trade_risk(
        _decision(
            planned_action="buy",
            instrument_type="call",
            horizon_days=2,
            event_type="earnings",
            market_expectation="high",
            invalidation_rule="",
        ),
        high_runup=True,
    )
    assert flags.no_invalidation_rule is True
    # stricter floor wins (not pinned down to wait) ...
    assert verdict.verdict is VerdictType.AVOID_CHASE
    # ... but the missing-exit prompt must not be hidden by the escalation.
    assert _has_missing_invalidation_prompts(verdict.reasons)


def test_use_spread_floor_does_not_override_wait():
    # leveraged into a macro event -> use_spread_or_hedge on its own.
    _, spread_only = evaluate_pre_trade_risk(_decision(instrument_type="leveraged_etf", event_type="macro_event"))
    assert spread_only.verdict is VerdictType.USE_SPREAD_OR_HEDGE

    # add a missing invalidation rule -> WAIT floor; WAIT is stricter than
    # use_spread_or_hedge, so the verdict becomes WAIT (not use_spread).
    _, with_missing = evaluate_pre_trade_risk(
        _decision(instrument_type="leveraged_etf", event_type="macro_event", invalidation_rule="")
    )
    assert with_missing.verdict is VerdictType.WAIT
    assert _has_missing_invalidation_prompts(with_missing.reasons)


def test_real_invalidation_rule_is_not_flagged():
    flags, _ = evaluate_pre_trade_risk(_decision(invalidation_rule="Exit if it loses the 50-day MA."))
    assert flags.no_invalidation_rule is False


def test_max_loss_arithmetic_only_when_units_valid():
    decision = _decision(max_loss=2.0, position_size=10.0)

    _, default_verdict = evaluate_pre_trade_risk(decision)
    assert not any("adverse move" in reason for reason in default_verdict.reasons)
    assert any("define the exit basis" in reason.lower() for reason in default_verdict.reasons)

    _, pct_verdict = evaluate_pre_trade_risk(decision, max_loss_is_portfolio_pct=True)
    assert any("20% adverse move" in reason for reason in pct_verdict.reasons)
    assert not any("recommended stop" in reason and "not a recommended stop" not in reason for reason in pct_verdict.reasons)


def test_max_loss_pct_not_computed_without_position_size():
    # position_size 0 -> cannot divide -> no implied move, even if unit declared.
    decision = _decision(max_loss=2.0, position_size=0.0)
    _, verdict = evaluate_pre_trade_risk(decision, max_loss_is_portfolio_pct=True)
    assert not any("adverse move" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("instrument", ["call", "put"])
@pytest.mark.parametrize("event", ["earnings", "guidance"])
def test_naked_short_dated_option_into_event_triggers_flag(instrument, event):
    flags = derive_risk_flags(_decision(instrument_type=instrument, horizon_days=3, event_type=event))
    assert flags.short_dated_option is True
    assert flags.event_gamble is True


def test_long_dated_naked_option_does_not_trigger_short_dated():
    flags = derive_risk_flags(_decision(instrument_type="call", horizon_days=30, event_type="earnings"))
    assert flags.short_dated_option is False


@pytest.mark.parametrize("spread", ["call_spread", "put_spread"])
def test_spreads_are_not_treated_as_naked_short_dated(spread):
    naked = derive_risk_flags(_decision(instrument_type="call", horizon_days=3, event_type="earnings"))
    spread_flags = derive_risk_flags(_decision(instrument_type=spread, horizon_days=3, event_type="earnings"))
    assert naked.short_dated_option is True
    assert spread_flags.short_dated_option is False
    assert spread_flags.event_gamble is False


def test_entry_with_high_runup_escalates_to_avoid_chase():
    _, verdict = evaluate_pre_trade_risk(
        _decision(planned_action="buy", instrument_type="call", horizon_days=2, event_type="earnings", market_expectation="high"),
        high_runup=True,
    )
    assert verdict.verdict is VerdictType.AVOID_CHASE


@pytest.mark.parametrize("action", ["watch", "avoid", "trim"])
def test_non_entry_actions_are_not_treated_as_chase(action):
    flags, verdict = evaluate_pre_trade_risk(
        _decision(planned_action=action, instrument_type="call", horizon_days=2, event_type="earnings", market_expectation="high"),
        high_runup=True,
    )
    assert flags.high_runup is True  # flag stays descriptive
    assert verdict.verdict is not VerdictType.AVOID_CHASE


def test_position_too_large_floors_to_reduce_size():
    flags, verdict = evaluate_pre_trade_risk(_decision(position_size=25.0))
    assert flags.position_too_large is True
    assert verdict.verdict is VerdictType.REDUCE_SIZE


def test_clean_entry_proceeds():
    flags, verdict = evaluate_pre_trade_risk(_decision())
    assert flags.any_active() is False
    assert verdict.verdict is VerdictType.PROCEED


@pytest.mark.parametrize(
    "decision",
    [
        _decision(),
        _decision(invalidation_rule=""),
        _decision(planned_action="watch", instrument_type="call", horizon_days=2, event_type="earnings"),
        _decision(instrument_type="leveraged_etf", event_type="macro_event"),
        _decision(position_size=40.0, market_expectation="high"),
    ],
)
def test_every_verdict_carries_mandatory_disclaimers(decision):
    _, verdict = evaluate_pre_trade_risk(decision, high_runup=True)
    assert verdict.heuristic_only is True
    assert verdict.not_financial_advice is True
    data = verdict.to_dict()
    assert data["heuristic_only"] is True
    assert data["not_financial_advice"] is True
