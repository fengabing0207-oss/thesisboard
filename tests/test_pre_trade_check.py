import pytest

from src.pre_trade_check import (
    DecisionStatus,
    EventType,
    Level,
    RiskFlags,
    ThesisDecision,
    Verdict,
    VerdictType,
)


def _valid_kwargs(**overrides):
    base = dict(
        ticker="nvda",
        theme="AI Infrastructure",
        event_type="earnings",
        planned_action="buy",
        instrument_type="stock",
        horizon_days=20,
        market_expectation="medium",
        entry_thesis="Datacenter demand keeps beating estimates.",
        risk_thesis="Guidance disappoints; multiple compresses.",
        max_loss=1000.0,
        invalidation_rule="Exit if it closes below the pre-earnings range.",
        confidence="medium",
        position_size=5.0,
        notes="",
    )
    base.update(overrides)
    return base


def test_create_valid_decision():
    decision = ThesisDecision(**_valid_kwargs())

    assert decision.ticker == "NVDA"  # normalized to upper
    assert decision.event_type is EventType.EARNINGS
    assert decision.status is DecisionStatus.PENDING  # default
    assert decision.decision_id  # auto-generated
    assert decision.created_at  # auto-filled
    assert isinstance(decision.market_expectation, Level)


@pytest.mark.parametrize("missing", ["ticker", "theme", "entry_thesis"])
def test_missing_required_text_fields_raise(missing):
    with pytest.raises(ValueError, match=missing):
        ThesisDecision(**_valid_kwargs(**{missing: "   "}))


def test_missing_invalidation_rule_is_allowed():
    # An empty invalidation_rule is handled clearly: the record is accepted (it
    # does not crash). The no_invalidation_rule flag exists on RiskFlags so a
    # later check (PR #19) can surface it.
    decision = ThesisDecision(**_valid_kwargs(invalidation_rule=""))
    assert decision.invalidation_rule == ""
    assert "no_invalidation_rule" in RiskFlags().to_dict()


@pytest.mark.parametrize("field_name", ["confidence", "market_expectation"])
def test_level_fields_reject_invalid(field_name):
    with pytest.raises(ValueError, match="low, medium, high"):
        ThesisDecision(**_valid_kwargs(**{field_name: "extreme"}))


@pytest.mark.parametrize("field_name", ["confidence", "market_expectation"])
@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_level_fields_accept_valid(field_name, value):
    decision = ThesisDecision(**_valid_kwargs(**{field_name: value}))
    assert getattr(decision, field_name) is Level(value)


@pytest.mark.parametrize("bad", [-1.0, 100.1, 150.0])
def test_position_size_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="between 0 and 100"):
        ThesisDecision(**_valid_kwargs(position_size=bad))


@pytest.mark.parametrize("ok", [0.0, 50.0, 100.0])
def test_position_size_in_range_ok(ok):
    assert ThesisDecision(**_valid_kwargs(position_size=ok)).position_size == ok


def test_horizon_and_max_loss_validation():
    with pytest.raises(ValueError, match="horizon_days"):
        ThesisDecision(**_valid_kwargs(horizon_days=0))
    with pytest.raises(ValueError, match="max_loss"):
        ThesisDecision(**_valid_kwargs(max_loss=-10))
    assert ThesisDecision(**_valid_kwargs(max_loss=None)).max_loss is None


def test_invalid_enum_choices_raise():
    with pytest.raises(ValueError, match="event_type"):
        ThesisDecision(**_valid_kwargs(event_type="meme"))
    with pytest.raises(ValueError, match="instrument_type"):
        ThesisDecision(**_valid_kwargs(instrument_type="future"))
    with pytest.raises(ValueError, match="planned_action"):
        ThesisDecision(**_valid_kwargs(planned_action="yolo"))


def test_decision_serializes_and_round_trips():
    decision = ThesisDecision(**_valid_kwargs())
    data = decision.to_dict()

    assert data["ticker"] == "NVDA"
    assert data["event_type"] == "earnings"  # enum serialized to its value
    assert data["status"] == "pending"
    assert isinstance(data["position_size"], float)

    restored = ThesisDecision.from_dict(data)
    assert restored.to_dict() == data  # stable round-trip


def test_risk_flags_serialize_cleanly():
    flags = RiskFlags(earnings_imminent=True, short_dated_option=True)
    data = flags.to_dict()

    assert set(data) == {
        "earnings_imminent",
        "high_runup",
        "short_dated_option",
        "leveraged_product",
        "no_invalidation_rule",
        "position_too_large",
        "event_gamble",
        "priced_for_perfection_candidate",
    }
    assert all(isinstance(v, bool) for v in data.values())
    assert set(flags.active()) == {"earnings_imminent", "short_dated_option"}
    assert flags.any_active() is True
    assert RiskFlags.from_dict(data) == flags
    assert RiskFlags().any_active() is False


def test_verdict_serializes_with_mandatory_disclaimers():
    verdict = Verdict(verdict="wait", reasons=["define your exit"])
    data = verdict.to_dict()

    assert data["verdict"] == "wait"
    assert data["reasons"] == ["define your exit"]
    assert data["heuristic_only"] is True
    assert data["not_financial_advice"] is True
    assert Verdict.from_dict(data).to_dict() == data


def test_verdict_rejects_false_disclaimers():
    with pytest.raises(ValueError, match="heuristic_only"):
        Verdict(verdict="proceed", heuristic_only=False)
    with pytest.raises(ValueError, match="not_financial_advice"):
        Verdict(verdict="proceed", not_financial_advice=False)


def test_verdict_rejects_invalid_verdict_value():
    with pytest.raises(ValueError, match="verdict"):
        Verdict(verdict="moon")
    assert Verdict(verdict="avoid_chase").verdict is VerdictType.AVOID_CHASE
