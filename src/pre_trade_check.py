"""Pre-Trade Check data model (PR #17).

Behavioral-guardrail infrastructure for recording a trade idea BEFORE entry and
forcing structured pre-trade reasoning. This is NOT an alpha model and makes no
predictive or statistical claim. Verdict outputs are explicitly labeled
``heuristic_only`` and ``not_financial_advice``; the point is to enable
disciplined, structured pre-trade decisions and later prospective post-trade
validation.

This PR ships only the neutral data structures:
  - ``ThesisDecision`` — the recorded pre-trade idea.
  - ``RiskFlags`` — eight behavioral risk flags.
  - ``Verdict`` — a heuristic verdict with reasons.

The logic that derives flags/verdicts from a decision is intentionally NOT here;
those risk-judgement rules get their own design and review in PR #19. No
methodology from the validation core is used or changed in this module.
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

    These are self-audit prompts, not predictive signals. The logic that decides
    when each flag should be set lives in PR #19, not here.
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
