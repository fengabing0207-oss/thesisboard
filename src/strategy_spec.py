"""Immutable execution contract for News Signal Lab research strategies.

The spec intentionally supports one narrow strategy family.  Unsupported
choices fail validation instead of being recorded as if the engine implemented
them.  A stable hash makes it possible to prove that validation and held-out
calculations used the same rule set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json


STRATEGY_SPEC_VERSION = "news-signal-strategy.v1"
CALCULATION_CONTRACT_VERSION = "news-signal-calculation.v1"


@dataclass(frozen=True)
class StrategySpec:
    """Fully specified rule set supported by the current position-book engine."""

    model_name: str
    model_version: str
    probability_threshold: float
    holding_horizon_sessions: int
    universe: tuple[str, ...]
    benchmark: str = "SPY"
    one_way_cost_bps: float = 5.0
    availability_rule_version: str = "observed-to-regular-close.v1"
    target_definition: str = "beta_adjusted_market_abnormal_return"
    spec_version: str = STRATEGY_SPEC_VERSION
    signal_definition: str = "predicted_probability_gte_threshold"
    entry_rule: str = "eligible_signal_close"
    exit_rule: str = "recorded_horizon_end_close"
    rebalance_schedule: str = "each_benchmark_session_close"
    weighting_rule: str = "equal_weight_active_tickers"
    overlap_rule: str = "collapse_same_ticker_events"
    side: str = "long_only"
    max_gross_exposure: float = 1.0
    cash_return_assumption: str = "zero"

    def __post_init__(self) -> None:
        model = str(self.model_name).strip()
        model_version = str(self.model_version).strip()
        universe = tuple(
            sorted(
                {
                    str(value).strip().upper()
                    for value in self.universe
                    if str(value).strip()
                }
            )
        )
        benchmark = str(self.benchmark).strip().upper()
        object.__setattr__(self, "model_name", model)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "benchmark", benchmark)

        if not model:
            raise ValueError("model_name is required")
        if not model_version:
            raise ValueError("model_version is required")
        if not 0.0 <= float(self.probability_threshold) <= 1.0:
            raise ValueError("probability_threshold must be between 0 and 1")
        if int(self.holding_horizon_sessions) < 1:
            raise ValueError("holding_horizon_sessions must be positive")
        if not universe:
            raise ValueError("universe must contain at least one ticker")
        if not benchmark:
            raise ValueError("benchmark is required")
        if float(self.one_way_cost_bps) < 0.0:
            raise ValueError("one_way_cost_bps must be non-negative")
        supported = {
            "spec_version": STRATEGY_SPEC_VERSION,
            "availability_rule_version": "observed-to-regular-close.v1",
            "signal_definition": "predicted_probability_gte_threshold",
            "entry_rule": "eligible_signal_close",
            "exit_rule": "recorded_horizon_end_close",
            "rebalance_schedule": "each_benchmark_session_close",
            "weighting_rule": "equal_weight_active_tickers",
            "overlap_rule": "collapse_same_ticker_events",
            "side": "long_only",
            "cash_return_assumption": "zero",
        }
        for field, expected in supported.items():
            if getattr(self, field) != expected:
                raise ValueError(f"unsupported {field}: {getattr(self, field)!r}")
        if self.target_definition not in {
            "beta_adjusted_market_abnormal_return",
            "market_sector_blended_abnormal_return",
        }:
            raise ValueError(f"unsupported target_definition: {self.target_definition!r}")
        if float(self.max_gross_exposure) != 1.0:
            raise ValueError("the current engine supports max_gross_exposure=1.0 only")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["universe"] = list(self.universe)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict) -> "StrategySpec":
        if not isinstance(payload, dict):
            raise ValueError("StrategySpec payload must be a dictionary")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown StrategySpec fields: {unknown}")
        values = dict(payload)
        if "universe" in values:
            values["universe"] = tuple(values["universe"])
        return cls(**values)

    @property
    def spec_id(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"{self.spec_version}:{digest[:16]}"


def audit_calculation_parity(runs: dict[str, dict]) -> dict:
    """Verify that named runs share an identical spec and calculation engine.

    This proves rule/calculation parity only.  It does not claim that a research
    backtest reproduces broker fills or a live trading system.
    """

    if len(runs) < 2:
        return {
            "status": "insufficient_runs",
            "calculation_contract_version": CALCULATION_CONTRACT_VERSION,
            "runs": {},
            "mismatches": ["at least two runs are required"],
        }

    records: dict[str, dict] = {}
    mismatches: list[str] = []
    for name, run in runs.items():
        record = {
            "status": run.get("status"),
            "strategy_spec_id": run.get("strategy_spec_id"),
            "backtest_version": run.get("backtest_version"),
            "calculation_contract_version": run.get("calculation_contract_version"),
        }
        records[str(name)] = record
        missing = [key for key, value in record.items() if value is None]
        if missing:
            mismatches.append(f"{name} missing {', '.join(missing)}")
        if record["status"] != "ok":
            mismatches.append(f"{name} status is {record['status']!r}")

    for field in (
        "strategy_spec_id",
        "backtest_version",
        "calculation_contract_version",
    ):
        values = {record[field] for record in records.values()}
        if len(values) != 1:
            mismatches.append(f"{field} differs across runs")

    return {
        "status": "verified" if not mismatches else "mismatch",
        "calculation_contract_version": CALCULATION_CONTRACT_VERSION,
        "runs": records,
        "mismatches": mismatches,
    }
