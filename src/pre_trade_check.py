"""Pre-Trade Check data model (PR #17).

Behavioral-guardrail infrastructure for recording a trade idea BEFORE entry and
forcing structured pre-trade reasoning. This is NOT an alpha model and makes no
predictive or statistical claim. Verdict outputs are explicitly labeled
``heuristic_only`` and ``not_financial_advice``; the point is to enable
disciplined, structured pre-trade decisions and later prospective post-trade
validation.

Data structures:
  - ``ThesisDecision`` — the recorded pre-trade idea.
  - ``RiskFlags`` — eight behavioral risk flags.
  - ``Verdict`` — a heuristic verdict with reasons.

Expert-rule evaluator (PR #19):
  - ``evaluate_pre_trade_risk`` — maps a decision (plus a runtime ``high_runup``
    observation) to ``RiskFlags`` and a ``Verdict`` using transparent, explainable
    behavioral rules. These are NOT predictive signals and NOT financial advice;
    no validation-core methodology, no backtest, no market data is used.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ANALYST_DAY = "analyst_day"
    PRODUCT_LAUNCH = "product_launch"
    SYMPATHY_MOVE = "sympathy_move"
    PULLBACK = "pullback"
    BREAKOUT = "breakout"
    MACRO_EVENT = "macro_event"
    OTHER = "other"


class PlannedAction(str, Enum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    TRIM = "trim"
    AVOID = "avoid"
    WATCH = "watch"


class InstrumentType(str, Enum):
    STOCK = "stock"
    CALL = "call"
    PUT = "put"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"
    LEVERAGED_ETF = "leveraged_etf"
    OTHER = "other"


class Level(str, Enum):
    """Shared low/medium/high scale for market_expectation and confidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionStatus(str, Enum):
    PENDING = "pending"
    ENTERED = "entered"
    SKIPPED = "skipped"
    INVALIDATED = "invalidated"
    CLOSED = "closed"


class VerdictType(str, Enum):
    PROCEED = "proceed"
    WAIT = "wait"
    REDUCE_SIZE = "reduce_size"
    USE_SPREAD_OR_HEDGE = "use_spread_or_hedge"
    AVOID_CHASE = "avoid_chase"


# --- validation helpers ----------------------------------------------------


def _coerce_enum(value, enum_cls, field_name: str):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"{field_name} must be one of: {allowed} (got {value!r})") from None


def _require_present(value, field_name: str):
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field_name} is required")
    return value


def _require_text(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and cannot be empty")
    return value.strip()


def _require_positive_int(value, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive number of days (got {value!r})")
    return number


def _require_percent(value, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required")
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"{field_name} must be a percent of portfolio between 0 and 100 (got {value!r})")
    return number


def _optional_nonneg(value, field_name: str):
    if value is None:
        return None
    number = float(value)
    if number < 0:
        raise ValueError(f"{field_name} cannot be negative (got {value!r})")
    return number


# --- the recorded pre-trade idea ------------------------------------------


@dataclass
class ThesisDecision:
    """A trade idea recorded before entry.

    Required (raise on missing/blank): ``ticker``, ``theme``, ``entry_thesis``,
    the enum fields, ``horizon_days`` and ``position_size``. ``decision_id`` and
    ``created_at`` auto-fill when blank. ``invalidation_rule`` is recorded but
    allowed to be empty — an empty rule is handled clearly by accepting the
    record (a later check can surface the ``no_invalidation_rule`` flag) rather
    than by rejecting it.
    """

    decision_id: str = ""
    created_at: str = ""
    ticker: str = ""
    theme: str = ""
    event_type: "EventType | str | None" = None
    planned_action: "PlannedAction | str | None" = None
    instrument_type: "InstrumentType | str | None" = None
    horizon_days: "int | None" = None
    market_expectation: "Level | str | None" = None  # your read of how much the market already expects
    entry_thesis: str = ""
    risk_thesis: str = ""
    max_loss: "float | None" = None
    invalidation_rule: str = ""
    confidence: "Level | str | None" = None
    position_size: "float | None" = None  # percent of portfolio, 0-100
    notes: str = ""
    status: "DecisionStatus | str" = DecisionStatus.PENDING

    def __post_init__(self) -> None:
        self.decision_id = self.decision_id or uuid.uuid4().hex
        self.created_at = self.created_at or _utc_now_iso()

        self.ticker = _require_text(self.ticker, "ticker").upper()
        self.theme = _require_text(self.theme, "theme")
        self.entry_thesis = _require_text(self.entry_thesis, "entry_thesis")
        self.risk_thesis = (self.risk_thesis or "").strip()
        self.invalidation_rule = (self.invalidation_rule or "").strip()
        self.notes = (self.notes or "").strip()

        self.event_type = _coerce_enum(_require_present(self.event_type, "event_type"), EventType, "event_type")
        self.planned_action = _coerce_enum(_require_present(self.planned_action, "planned_action"), PlannedAction, "planned_action")
        self.instrument_type = _coerce_enum(_require_present(self.instrument_type, "instrument_type"), InstrumentType, "instrument_type")
        self.market_expectation = _coerce_enum(_require_present(self.market_expectation, "market_expectation"), Level, "market_expectation")
        self.confidence = _coerce_enum(_require_present(self.confidence, "confidence"), Level, "confidence")
        self.status = _coerce_enum(self.status or DecisionStatus.PENDING, DecisionStatus, "status")

        self.horizon_days = _require_positive_int(self.horizon_days, "horizon_days")
        self.position_size = _require_percent(self.position_size, "position_size")
        self.max_loss = _optional_nonneg(self.max_loss, "max_loss")

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "created_at": self.created_at,
            "ticker": self.ticker,
            "theme": self.theme,
            "event_type": self.event_type.value,
            "planned_action": self.planned_action.value,
            "instrument_type": self.instrument_type.value,
            "horizon_days": self.horizon_days,
            "market_expectation": self.market_expectation.value,
            "entry_thesis": self.entry_thesis,
            "risk_thesis": self.risk_thesis,
            "max_loss": self.max_loss,
            "invalidation_rule": self.invalidation_rule,
            "confidence": self.confidence.value,
            "position_size": self.position_size,
            "notes": self.notes,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThesisDecision":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


# --- risk flags ------------------------------------------------------------


@dataclass(frozen=True)
class RiskFlags:
    """Eight behavioral risk flags (all default False).

    These are self-audit prompts, not predictive signals. The rules that set them
    live in ``evaluate_pre_trade_risk`` below.
    """

    earnings_imminent: bool = False
    high_runup: bool = False
    short_dated_option: bool = False
    leveraged_product: bool = False
    no_invalidation_rule: bool = False
    position_too_large: bool = False
    event_gamble: bool = False
    priced_for_perfection_candidate: bool = False

    def to_dict(self) -> dict:
        return {key: bool(value) for key, value in asdict(self).items()}

    def active(self) -> list[str]:
        return [name for name, value in self.to_dict().items() if value]

    def any_active(self) -> bool:
        return any(self.to_dict().values())

    @classmethod
    def from_dict(cls, data: dict) -> "RiskFlags":
        known = {f.name for f in fields(cls)}
        return cls(**{key: bool(value) for key, value in data.items() if key in known})


# --- verdict ---------------------------------------------------------------


@dataclass
class Verdict:
    """A heuristic pre-trade verdict.

    ``heuristic_only`` and ``not_financial_advice`` are always True and validated
    as such — this output must never be presented as validated predictive advice.
    """

    verdict: "VerdictType | str" = VerdictType.PROCEED
    reasons: list = field(default_factory=list)
    heuristic_only: bool = True
    not_financial_advice: bool = True

    def __post_init__(self) -> None:
        self.verdict = _coerce_enum(self.verdict, VerdictType, "verdict")
        self.reasons = [str(reason) for reason in (self.reasons or [])]
        if self.heuristic_only is not True:
            raise ValueError("Verdict.heuristic_only must be True")
        if self.not_financial_advice is not True:
            raise ValueError("Verdict.not_financial_advice must be True")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "heuristic_only": True,
            "not_financial_advice": True,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Verdict":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


# --- expert-rule evaluator (PR #19) ---------------------------------------
#
# Transparent behavioral heuristics, NOT predictive signals and NOT advice.
# No market data, no backtest, no IV / 52-week logic, no persistence.

SHORT_DATED_OPTION_MAX_DAYS = 5
POSITION_SIZE_SOFT_LIMIT_PCT = 10.0

ENTRY_ACTIONS = frozenset({PlannedAction.BUY, PlannedAction.ADD, PlannedAction.HOLD})
_NAKED_OPTIONS = frozenset({InstrumentType.CALL, InstrumentType.PUT})
_SPREAD_OPTIONS = frozenset({InstrumentType.CALL_SPREAD, InstrumentType.PUT_SPREAD})
_BINARY_EVENT_TYPES = frozenset(
    {
        EventType.EARNINGS,
        EventType.GUIDANCE,
        EventType.ANALYST_DAY,
        EventType.PRODUCT_LAUNCH,
        EventType.MACRO_EVENT,
    }
)
_SHORT_OPTION_EVENTS = frozenset({EventType.EARNINGS, EventType.GUIDANCE})
_PLACEHOLDER_INVALIDATION = frozenset(
    {"", "idk", "tbd", "?", "??", "???", "na", "n/a", "none", "not sure", "no", "nope", "-", ".", "dunno", "unsure"}
)

# Severity order: proceed < reduce_size < use_spread_or_hedge < wait < avoid_chase.
_VERDICT_SEVERITY = {
    VerdictType.PROCEED: 0,
    VerdictType.REDUCE_SIZE: 1,
    VerdictType.USE_SPREAD_OR_HEDGE: 2,
    VerdictType.WAIT: 3,
    VerdictType.AVOID_CHASE: 4,
}
_SEVERITY_TO_VERDICT = {severity: verdict for verdict, severity in _VERDICT_SEVERITY.items()}


def _is_placeholder_invalidation(rule: str) -> bool:
    return (rule or "").strip().lower() in _PLACEHOLDER_INVALIDATION


def derive_risk_flags(decision: ThesisDecision, *, high_runup: bool = False) -> RiskFlags:
    """Derive the eight behavioral risk flags from a decision.

    Flags are purely descriptive of the inputs (they ignore ``planned_action`` —
    action gating only affects the verdict). ``high_runup`` is a runtime-only
    observation (a manual UI checkbox); it is NOT part of the schema.
    """
    short_dated_option = (
        decision.instrument_type in _NAKED_OPTIONS
        and decision.horizon_days <= SHORT_DATED_OPTION_MAX_DAYS
        and decision.event_type in _SHORT_OPTION_EVENTS
    )
    leveraged_product = decision.instrument_type == InstrumentType.LEVERAGED_ETF
    event_gamble = decision.event_type in _BINARY_EVENT_TYPES and (short_dated_option or leveraged_product)

    return RiskFlags(
        earnings_imminent=decision.event_type == EventType.EARNINGS,
        high_runup=bool(high_runup),
        short_dated_option=short_dated_option,
        leveraged_product=leveraged_product,
        no_invalidation_rule=_is_placeholder_invalidation(decision.invalidation_rule),
        position_too_large=decision.position_size > POSITION_SIZE_SOFT_LIMIT_PCT,
        event_gamble=event_gamble,
        priced_for_perfection_candidate=decision.market_expectation == Level.HIGH,
    )


def _max_loss_reasons(decision: ThesisDecision, max_loss_is_portfolio_pct: bool) -> list:
    """Arithmetic translation of the user's own max-loss / size inputs.

    Only computes an implied price move when the unit is unambiguously % of
    portfolio (and a position size exists). Never emits a recommended stop.
    """
    if decision.max_loss is None or decision.max_loss <= 0:
        return []
    if max_loss_is_portfolio_pct and decision.position_size and decision.position_size > 0:
        implied_pct = decision.max_loss / decision.position_size * 100
        return [
            f"Your inputs imply roughly a {implied_pct:.0f}% adverse move on this position "
            f"(max loss {decision.max_loss:g}% of portfolio / position {decision.position_size:g}% of portfolio). "
            "This is pure arithmetic from your own numbers, not a recommended stop."
        ]
    return [
        "Max loss recorded; define the exit basis. The unit is not specified as % of portfolio, "
        "so no implied price move is computed."
    ]


def decide_verdict(
    decision: ThesisDecision,
    flags: RiskFlags,
    *,
    max_loss_is_portfolio_pct: bool = False,
) -> Verdict:
    """Map flags to a heuristic verdict using severity floors and action gating.

    Risk escalation only applies to entry actions (buy / add / hold); for
    watch / avoid / trim, reasons are still surfaced but the verdict is not
    escalated (a non-entry action is never treated as a chase entry). A missing
    or placeholder invalidation rule is a hard floor of WAIT for entries — a
    stricter floor (e.g. avoid_chase) still wins via take-the-stricter, but the
    missing-exit reflection prompts are always surfaced regardless of verdict.
    """
    is_entry = decision.planned_action in ENTRY_ACTIONS
    reasons: list = []
    severity = _VERDICT_SEVERITY[VerdictType.PROCEED]

    def consider(active: bool, floor: VerdictType, reason: str) -> None:
        nonlocal severity
        if not active:
            return
        reasons.append(reason)
        if is_entry:
            severity = max(severity, _VERDICT_SEVERITY[floor])

    consider(flags.high_runup, VerdictType.AVOID_CHASE, "Recent large run-up flagged; entering here risks chasing.")
    consider(flags.event_gamble, VerdictType.USE_SPREAD_OR_HEDGE, "Binary event with a leveraged or naked short-dated instrument reads as an event gamble.")
    consider(flags.short_dated_option, VerdictType.USE_SPREAD_OR_HEDGE, "Naked short-dated option into a binary event; time-decay and gamma risk are high.")
    consider(flags.leveraged_product, VerdictType.USE_SPREAD_OR_HEDGE, "Leveraged product carries path and decay risk.")
    consider(flags.position_too_large, VerdictType.REDUCE_SIZE, f"Position size exceeds the {POSITION_SIZE_SOFT_LIMIT_PCT:.0f}% soft limit.")
    consider(flags.priced_for_perfection_candidate, VerdictType.WAIT, "You rate market expectation as high; the setup may be priced for perfection.")
    consider(flags.earnings_imminent, VerdictType.WAIT, "Earnings are imminent; the outcome is binary.")

    # A missing/placeholder invalidation rule is a hard floor of WAIT (not a pin):
    # take-the-stricter still lets a harsher floor like avoid_chase win. The
    # reflection prompts are surfaced regardless, so a stricter verdict never
    # hides the fact that no exit was defined.
    if is_entry and flags.no_invalidation_rule:
        reasons = [
            "No usable invalidation rule / stop basis was provided.",
            "Reflection: what specific price, level, or event would prove this thesis wrong?",
            "Reflection: define your exit basis before entry — where do you get out if it goes against you?",
            "These are reasoning prompts, not financial advice and not a recommended stop level.",
        ] + reasons
        severity = max(severity, _VERDICT_SEVERITY[VerdictType.WAIT])

    # Defined-risk spreads get a softer, non-escalating note (not naked-option severity).
    if (
        decision.instrument_type in _SPREAD_OPTIONS
        and decision.event_type in _BINARY_EVENT_TYPES
        and decision.market_expectation == Level.HIGH
    ):
        reasons.append("Defined-risk spread into a binary event with high expectations; risk is capped but still event-dependent.")

    reasons.extend(_max_loss_reasons(decision, max_loss_is_portfolio_pct))

    if not is_entry and flags.any_active():
        reasons.append(f"Planned action is '{decision.planned_action.value}' (not an entry); risk escalation suppressed.")

    if not reasons:
        reasons = ["No behavioral risk flags triggered by this heuristic check."]

    verdict = _SEVERITY_TO_VERDICT[severity]
    return Verdict(verdict=verdict, reasons=reasons)


def evaluate_pre_trade_risk(
    decision: ThesisDecision,
    *,
    high_runup: bool = False,
    max_loss_is_portfolio_pct: bool = False,
) -> "tuple[RiskFlags, Verdict]":
    """Heuristic behavioral pre-trade check. Not advice, not a predictive signal.

    ``high_runup`` is a runtime-only observation (manual checkbox), not a schema
    field. ``max_loss_is_portfolio_pct`` opts into the max-loss arithmetic only
    when the unit is unambiguously % of portfolio.
    """
    flags = derive_risk_flags(decision, high_runup=high_runup)
    verdict = decide_verdict(decision, flags, max_loss_is_portfolio_pct=max_loss_is_portfolio_pct)
    return flags, verdict
