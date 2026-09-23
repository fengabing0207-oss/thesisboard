"""Scheduled collection and validation-only calibration for News Signal Lab.

The cycle may propose a challenger but never promotes one.  Promotion is a
separate append-only human action so held-out diagnostics cannot silently feed
back into the active rule set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from .news_collector import collect_ticker_news
from .news_signal_backtest import select_and_evaluate_holdout_strategy
from .news_signal_lab import build_news_return_dataset, compare_walk_forward_models
from .news_store import (
    DB_PATH,
    append_research_event,
    get_research_event,
    list_news_items,
    list_research_events,
    storage_backend_info,
)
from .price_provider import CachingPriceProvider, YFinancePriceProvider
from .research_readiness import summarize_model_dataset
from .strategy_spec import StrategySpec


AUTOMATION_VERSION = "scheduled-research-loop.v2"
PRICE_CACHE = Path(__file__).resolve().parents[1] / "data" / "automation_price_cache"


@dataclass(frozen=True)
class ResearchCycleConfig:
    horizon_days: int = 1
    min_train_rows: int = 30
    min_train_sessions: int = 10
    validation_fraction: float = 0.60
    min_validation_sessions: int = 10
    min_test_sessions: int = 5
    min_validation_signals: int = 5
    max_validation_selection_rate: float = 0.50
    max_validation_average_daily_turnover: float = 1.0
    one_way_cost_bps: float = 5.0
    benchmark_symbol: str = "SPY"
    price_lookback_calendar_days: int = 240
    label_settlement_delay_minutes: int = 90

    def __post_init__(self) -> None:
        benchmark = str(self.benchmark_symbol).strip().upper()
        object.__setattr__(self, "benchmark_symbol", benchmark)
        if int(self.horizon_days) < 1:
            raise ValueError("horizon_days must be positive")
        if int(self.min_train_rows) < 1 or int(self.min_train_sessions) < 1:
            raise ValueError("minimum training requirements must be positive")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if int(self.min_validation_sessions) < 1 or int(self.min_test_sessions) < 1:
            raise ValueError("minimum validation and test sessions must be positive")
        if int(self.min_validation_signals) < 1:
            raise ValueError("min_validation_signals must be positive")
        if not 0.0 < float(self.max_validation_selection_rate) <= 1.0:
            raise ValueError("max_validation_selection_rate must be in (0, 1]")
        if not 0.0 < float(self.max_validation_average_daily_turnover) <= 2.0:
            raise ValueError("max_validation_average_daily_turnover must be in (0, 2]")
        if float(self.one_way_cost_bps) < 0.0:
            raise ValueError("one_way_cost_bps must be non-negative")
        if not benchmark:
            raise ValueError("benchmark_symbol is required")
        if int(self.price_lookback_calendar_days) < 30:
            raise ValueError("price_lookback_calendar_days must be at least 30")
        if int(self.label_settlement_delay_minutes) < 0:
            raise ValueError("label_settlement_delay_minutes must be non-negative")


def run_research_cycle(
    tickers,
    *,
    config: ResearchCycleConfig | None = None,
    db_path=DB_PATH,
    fetcher=None,
    price_provider=None,
    now=None,
    run_id: str | None = None,
) -> dict:
    """Collect data, evaluate matured history, and log a pending challenger."""

    settings = config or ResearchCycleConfig()
    symbols = tuple(
        dict.fromkeys(
            str(value).strip().upper() for value in tickers if str(value).strip()
        )
    )
    if not symbols:
        raise ValueError("at least one ticker is required")
    cycle_id = str(run_id or uuid4())
    timestamp = _utc_timestamp(now)
    backend = storage_backend_info(db_path)
    started_logged = False
    try:
        append_research_event(
            event_type="automation_run_started",
            run_id=cycle_id,
            payload={
                "automation_version": AUTOMATION_VERSION,
                "tickers": list(symbols),
                "config": settings.__dict__,
                "storage": backend,
            },
            event_time=timestamp,
            db_path=db_path,
        )
        started_logged = True

        collection_kwargs = {
            "observed_at": timestamp,
            "db_path": db_path,
        }
        if fetcher is not None:
            collection_kwargs["fetcher"] = fetcher
        collection = collect_ticker_news(symbols, **collection_kwargs)
        collection_payload = {
            key: value for key, value in collection.items() if key != "per_ticker"
        }
        collection_payload["errors"] = {
            ticker: _safe_error_message(message)
            for ticker, message in collection_payload.get("errors", {}).items()
        }
        append_research_event(
            event_type="collection_completed",
            run_id=cycle_id,
            subject_id=str(collection["run_id"]),
            payload=collection_payload,
            db_path=db_path,
        )

        if int(collection["failed_tickers"]) > 0:
            return _complete_without_candidate(
                cycle_id,
                status="collection_incomplete",
                collection=collection_payload,
                db_path=db_path,
                details={
                    "failed_tickers": int(collection["failed_tickers"]),
                    "errors": collection_payload["errors"],
                },
            )

        items = [
            item for item in list_news_items(db_path) if item["ticker"] in symbols
        ]
        if not items:
            return _complete_without_candidate(
                cycle_id,
                status="no_captured_headlines",
                collection=collection_payload,
                db_path=db_path,
            )

        provider = price_provider or CachingPriceProvider(
            YFinancePriceProvider(),
            PRICE_CACHE,
            snapshot_id=timestamp.date().isoformat(),
        )
        first_seen = min(pd.Timestamp(item["first_seen_at"]) for item in items)
        start = (
            first_seen.tz_convert("UTC").tz_localize(None).normalize()
            - pd.Timedelta(days=int(settings.price_lookback_calendar_days))
        )
        end = timestamp.tz_convert("UTC").tz_localize(None).normalize()
        bundle = provider.get_history(
            [*symbols, settings.benchmark_symbol],
            start,
            end,
        )
        price_vintage = _price_vintage(bundle)
        benchmark = bundle.prices.get(settings.benchmark_symbol)
        if benchmark is None or benchmark.empty:
            return _complete_without_candidate(
                cycle_id,
                status="missing_benchmark_prices",
                collection=collection_payload,
                db_path=db_path,
                details={
                    "missing_symbols": bundle.missing_symbols,
                    "price_vintage": price_vintage,
                },
            )

        dataset = build_news_return_dataset(
            items,
            prices_by_ticker={
                ticker: bundle.prices[ticker]
                for ticker in symbols
                if ticker in bundle.prices
            },
            benchmark_prices=benchmark,
            horizon_days=int(settings.horizon_days),
            labels_as_of=timestamp,
            label_settlement_delay_minutes=int(
                settings.label_settlement_delay_minutes
            ),
        )
        dataset_readiness = summarize_model_dataset(
            dataset,
            min_train_sessions=int(settings.min_train_sessions),
            min_validation_sessions=int(settings.min_validation_sessions),
            min_test_sessions=int(settings.min_test_sessions),
        )
        dataset_vintage = _dataset_vintage(dataset, settings, price_vintage)
        label_manifest = _settled_label_manifest(dataset)
        integrity_scope = {
            "universe": list(symbols),
            "horizon_days": int(settings.horizon_days),
            "benchmark_symbol": settings.benchmark_symbol,
            "label_settlement_delay_minutes": int(
                settings.label_settlement_delay_minutes
            ),
        }
        dataset_integrity = _check_dataset_integrity(
            db_path=db_path,
            dataset_readiness=dataset_readiness,
            label_manifest=label_manifest,
            integrity_scope=integrity_scope,
        )
        if dataset_integrity["status"] == "regressed":
            append_research_event(
                event_type="dataset_integrity_violation",
                run_id=cycle_id,
                payload={
                    **dataset_integrity,
                    "dataset_vintage": dataset_vintage,
                    "price_vintage": price_vintage,
                },
                db_path=db_path,
            )
            return _complete_without_candidate(
                cycle_id,
                status="dataset_integrity_regressed",
                collection=collection_payload,
                db_path=db_path,
                details={
                    "dataset_rows": int(len(dataset)),
                    "dataset_readiness": dataset_readiness,
                    "dataset_vintage": dataset_vintage,
                    "dataset_integrity": dataset_integrity,
                    "price_source": bundle.source,
                    "price_adjustment": bundle.adjustment,
                    "price_vintage": price_vintage,
                    "missing_symbols": bundle.missing_symbols,
                },
            )
        checkpoint_id = append_research_event(
            event_type="dataset_integrity_checkpoint",
            run_id=cycle_id,
            payload={
                "dataset_readiness": dataset_readiness,
                "integrity_scope": integrity_scope,
                "label_manifest": label_manifest,
                "label_manifest_hash": dataset_integrity["label_manifest_hash"],
                "dataset_vintage": dataset_vintage,
                "price_vintage": price_vintage,
            },
            db_path=db_path,
        )
        dataset_integrity["checkpoint_event_id"] = checkpoint_id
        comparison = compare_walk_forward_models(
            dataset,
            min_train_rows=int(settings.min_train_rows),
            min_train_sessions=int(settings.min_train_sessions),
        )
        if comparison["status"] != "ok":
            return _complete_without_candidate(
                cycle_id,
                status=comparison["status"],
                collection=collection_payload,
                db_path=db_path,
                details={
                    "dataset_rows": int(len(dataset)),
                    "dataset_readiness": dataset_readiness,
                    "dataset_vintage": dataset_vintage,
                    "dataset_integrity": dataset_integrity,
                    "price_source": bundle.source,
                    "price_adjustment": bundle.adjustment,
                    "price_vintage": price_vintage,
                    "missing_symbols": bundle.missing_symbols,
                },
            )

        policy = select_and_evaluate_holdout_strategy(
            comparison["common_predictions"],
            prices_by_ticker={
                ticker: bundle.prices[ticker]
                for ticker in symbols
                if ticker in bundle.prices
            },
            benchmark_prices=benchmark,
            validation_fraction=float(settings.validation_fraction),
            min_validation_sessions=int(settings.min_validation_sessions),
            min_test_sessions=int(settings.min_test_sessions),
            min_validation_signals=int(settings.min_validation_signals),
            max_validation_selection_rate=float(
                settings.max_validation_selection_rate
            ),
            max_validation_average_daily_turnover=float(
                settings.max_validation_average_daily_turnover
            ),
            one_way_cost_bps=float(settings.one_way_cost_bps),
            benchmark_symbol=settings.benchmark_symbol,
            universe=symbols,
        )
        if policy["status"] != "ok":
            return _complete_without_candidate(
                cycle_id,
                status=policy["status"],
                collection=collection_payload,
                db_path=db_path,
                details={
                    "dataset_rows": int(len(dataset)),
                    "dataset_readiness": dataset_readiness,
                    "dataset_vintage": dataset_vintage,
                    "dataset_integrity": dataset_integrity,
                    "price_source": bundle.source,
                    "price_adjustment": bundle.adjustment,
                    "price_vintage": price_vintage,
                    "chronological_windows": policy.get("chronological_windows", {}),
                    "validation_candidate_audit": _frame_records(
                        policy.get("candidates", pd.DataFrame())
                    ),
                },
            )

        champion = current_approved_strategy(db_path=db_path)
        candidate_payload = {
            "automation_version": AUTOMATION_VERSION,
            "strategy_spec_id": policy["strategy_spec_id"],
            "strategy_spec": policy["strategy_spec"],
            "dataset_vintage": dataset_vintage,
            "dataset_rows": int(len(dataset)),
            "dataset_readiness": dataset_readiness,
            "dataset_integrity": dataset_integrity,
            "selection_basis": "validation_only",
            "promotion_status": "pending_human_review",
            "auto_promoted": False,
            "current_champion_spec_id": (
                None if champion is None else champion["strategy_spec_id"]
            ),
            "chronological_windows": policy["chronological_windows"],
            "model_comparison_oos": _frame_records(comparison["comparison"]),
            "validation_candidate_audit": _frame_records(policy["candidates"]),
            "validation_metrics": policy["validation_backtest"]["metrics"],
            "held_out_diagnostics_not_used_for_selection": policy["test_backtest"][
                "metrics"
            ],
            "calculation_parity": policy["calculation_parity_audit"],
            "price_source": bundle.source,
            "price_adjustment": bundle.adjustment,
            "price_vintage": price_vintage,
            "missing_symbols": bundle.missing_symbols,
        }
        previous = list_research_events(
            db_path,
            event_type="strategy_candidate_proposed",
            limit=1,
        )
        unchanged = bool(
            previous
            and previous[0]["payload"].get("strategy_spec_id")
            == policy["strategy_spec_id"]
            and previous[0]["payload"].get("dataset_vintage") == dataset_vintage
        )
        candidate_event_type = (
            "strategy_candidate_unchanged"
            if unchanged
            else "strategy_candidate_proposed"
        )
        candidate_event_id = append_research_event(
            event_type=candidate_event_type,
            run_id=cycle_id,
            subject_id=policy["strategy_spec_id"],
            payload=candidate_payload,
            db_path=db_path,
        )
        completed = {
            "status": "candidate_unchanged" if unchanged else "candidate_proposed",
            "run_id": cycle_id,
            "candidate_event_id": candidate_event_id,
            "strategy_spec_id": policy["strategy_spec_id"],
            "collection": collection_payload,
            "dataset_vintage": dataset_vintage,
            "dataset_readiness": dataset_readiness,
            "dataset_integrity": dataset_integrity,
        }
        append_research_event(
            event_type="automation_run_completed",
            run_id=cycle_id,
            subject_id=policy["strategy_spec_id"],
            payload=completed,
            db_path=db_path,
        )
        return completed
    except Exception as exc:
        safe_message = _safe_error_message(exc)
        if started_logged:
            try:
                append_research_event(
                    event_type="automation_run_failed",
                    run_id=cycle_id,
                    payload={
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": safe_message,
                    },
                    db_path=db_path,
                )
            except Exception:
                pass
        if safe_message != str(exc):
            raise RuntimeError(safe_message) from None
        raise


def record_strategy_promotion(
    candidate_event_id: int,
    *,
    decision: str,
    decided_by: str,
    note: str = "",
    db_path=DB_PATH,
) -> dict:
    """Append a human approval/rejection; never mutate the candidate event."""

    action = str(decision).strip().lower()
    if action not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    reviewer = str(decided_by).strip()
    if not reviewer:
        raise ValueError("decided_by is required")
    candidate = get_research_event(int(candidate_event_id), db_path)
    if candidate is None or candidate["event_type"] != "strategy_candidate_proposed":
        raise ValueError("candidate_event_id must reference a proposed strategy")
    existing = list_research_events(db_path, limit=1_000)
    if any(
        event["event_type"].startswith("strategy_promotion_")
        and int(event["payload"].get("candidate_event_id", -1))
        == int(candidate_event_id)
        for event in existing
    ):
        raise ValueError("candidate already has a promotion decision")

    spec = StrategySpec.from_dict(candidate["payload"]["strategy_spec"])
    if spec.spec_id != candidate["payload"].get("strategy_spec_id"):
        raise ValueError("candidate StrategySpec hash does not match its recorded ID")
    run_id = str(uuid4())
    event_type = f"strategy_promotion_{action}"
    payload = {
        "candidate_event_id": int(candidate_event_id),
        "strategy_spec_id": spec.spec_id,
        "strategy_spec": spec.to_dict(),
        "decision": action,
        "decided_by": reviewer,
        "note": str(note).strip(),
        "automatic": False,
    }
    event_id = append_research_event(
        event_type=event_type,
        run_id=run_id,
        subject_id=spec.spec_id,
        promotion_candidate_event_id=int(candidate_event_id),
        payload=payload,
        db_path=db_path,
    )
    return {"event_id": event_id, "event_type": event_type, **payload}


def current_approved_strategy(*, db_path=DB_PATH) -> dict | None:
    approvals = list_research_events(
        db_path,
        event_type="strategy_promotion_approved",
        limit=1,
    )
    if not approvals:
        return None
    payload = approvals[0]["payload"]
    spec = StrategySpec.from_dict(payload["strategy_spec"])
    if spec.spec_id != payload.get("strategy_spec_id"):
        raise ValueError("approved StrategySpec hash does not match its recorded ID")
    return {
        "promotion_event_id": approvals[0]["id"],
        "strategy_spec_id": spec.spec_id,
        "strategy_spec": spec.to_dict(),
        "decided_by": payload["decided_by"],
        "decided_at": approvals[0]["event_time"],
    }


def _complete_without_candidate(
    run_id: str,
    *,
    status: str,
    collection: dict,
    db_path,
    details: dict | None = None,
) -> dict:
    payload = {
        "status": status,
        "run_id": run_id,
        "candidate_event_id": None,
        "strategy_spec_id": None,
        "collection": collection,
        **(details or {}),
    }
    append_research_event(
        event_type="automation_run_completed",
        run_id=run_id,
        payload=payload,
        db_path=db_path,
    )
    return payload


def _settled_label_manifest(dataset: pd.DataFrame) -> list[dict]:
    if dataset.empty or "usable_for_model" not in dataset:
        return []
    usable = dataset.loc[dataset["usable_for_model"].fillna(False)].copy()
    if usable.empty:
        return []
    records = []
    for row in usable.sort_values(["signal_session", "ticker"]).itertuples():
        records.append(
            {
                "ticker": str(row.ticker),
                "signal_session": pd.Timestamp(row.signal_session).date().isoformat(),
                "label_available_at": pd.Timestamp(row.label_available_at).isoformat(),
                "target_positive": int(row.target_positive),
                "target_abnormal_return": float(row.target_abnormal_return),
            }
        )
    return records


def _check_dataset_integrity(
    *,
    db_path,
    dataset_readiness: dict,
    label_manifest: list[dict],
    integrity_scope: dict,
) -> dict:
    canonical = json.dumps(label_manifest, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    checkpoints = list_research_events(
        db_path,
        event_type="dataset_integrity_checkpoint",
        limit=100,
    )
    previous = [
        event
        for event in checkpoints
        if event["payload"].get("integrity_scope") == integrity_scope
    ][:1]
    if not previous:
        return {
            "status": "baseline_created",
            "previous_checkpoint_event_id": None,
            "label_manifest_hash": manifest_hash,
            "checked_labels": len(label_manifest),
            "non_directional_return_revision_count": 0,
            "non_directional_return_revision_examples": [],
            "violations": [],
        }

    checkpoint = previous[0]
    payload = checkpoint["payload"]
    previous_readiness = payload.get("dataset_readiness", {})
    previous_manifest = payload.get("label_manifest", [])
    previous_by_key = {
        (str(row.get("ticker", "")), str(row.get("signal_session", ""))): row
        for row in previous_manifest
    }
    current_by_key = {
        (str(row.get("ticker", "")), str(row.get("signal_session", ""))): row
        for row in label_manifest
    }
    violations = []

    previous_sessions = int(previous_readiness.get("usable_sessions", 0))
    current_sessions = int(dataset_readiness.get("usable_sessions", 0))
    if current_sessions < previous_sessions:
        violations.append(
            {
                "type": "usable_sessions_decreased",
                "previous": previous_sessions,
                "current": current_sessions,
            }
        )

    removed = sorted(previous_by_key.keys() - current_by_key.keys())
    if removed:
        violations.append(
            {
                "type": "settled_labels_removed",
                "count": len(removed),
                "examples": [list(key) for key in removed[:10]],
            }
        )

    direction_flips = []
    return_revisions = []
    for key in sorted(previous_by_key.keys() & current_by_key.keys()):
        old = previous_by_key[key]
        new = current_by_key[key]
        if int(old["target_positive"]) != int(new["target_positive"]):
            direction_flips.append(
                {
                    "ticker": key[0],
                    "signal_session": key[1],
                    "previous": int(old["target_positive"]),
                    "current": int(new["target_positive"]),
                }
            )
            continue
        old_return = float(old["target_abnormal_return"])
        new_return = float(new["target_abnormal_return"])
        if abs(old_return - new_return) > 1e-12:
            return_revisions.append(
                {
                    "ticker": key[0],
                    "signal_session": key[1],
                    "previous": old_return,
                    "current": new_return,
                }
            )
    if direction_flips:
        violations.append(
            {
                "type": "settled_label_direction_changed",
                "count": len(direction_flips),
                "examples": direction_flips[:10],
            }
        )
    return {
        "status": "regressed" if violations else "ok",
        "previous_checkpoint_event_id": int(checkpoint["id"]),
        "label_manifest_hash": manifest_hash,
        "checked_labels": len(label_manifest),
        "non_directional_return_revision_count": len(return_revisions),
        "non_directional_return_revision_examples": return_revisions[:10],
        "violations": violations,
    }


def _dataset_vintage(
    dataset: pd.DataFrame,
    config: ResearchCycleConfig,
    price_vintage: str,
) -> str:
    rows = []
    for row in dataset.sort_values(["signal_session", "ticker"]).itertuples():
        rows.append(
            {
                "ticker": row.ticker,
                "signal_session": pd.Timestamp(row.signal_session).isoformat(),
                "data_vintage": row.data_vintage,
                "data_quality_flag": row.data_quality_flag,
                "target_abnormal_return": (
                    None
                    if pd.isna(row.target_abnormal_return)
                    else float(row.target_abnormal_return)
                ),
                "label_available_at": (
                    None
                    if pd.isna(row.label_available_at)
                    else pd.Timestamp(row.label_available_at).isoformat()
                ),
            }
        )
    payload = {
        "automation_version": AUTOMATION_VERSION,
        "config": config.__dict__,
        "price_vintage": price_vintage,
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _price_vintage(bundle) -> str:
    rows = []
    for symbol in sorted(bundle.prices):
        series = bundle.prices[symbol]
        for timestamp, value in series.sort_index().items():
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(timestamp).isoformat(),
                    "adjusted_close": float(value),
                }
            )
    payload = {
        "source": bundle.source,
        "adjustment": bundle.adjustment,
        "missing_symbols": sorted(bundle.missing_symbols),
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _safe_error_message(error) -> str:
    message = str(error)
    configured_url = os.getenv("THESISBOARD_DATABASE_URL", "").strip()
    if configured_url:
        message = message.replace(configured_url, "[REDACTED_DATABASE_URL]")
    message = re.sub(
        r"(postgres(?:ql)?://[^:/\s]+:)[^@\s]+@",
        r"\1[REDACTED]@",
        message,
        flags=re.IGNORECASE,
    )
    return message[:2_000]


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(timezone.utc))
    if timestamp.tzinfo is None:
        raise ValueError("now must include a timezone")
    return timestamp.tz_convert("UTC")
