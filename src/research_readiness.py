"""Operational and model-window readiness for the News Signal Lab.

Readiness is deliberately descriptive.  It never relaxes chronological model
requirements, fabricates missing sessions, or promotes a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_EXPECTED_TICKERS = ("NVDA", "AVGO", "MRVL", "MU", "VRT", "ORCL", "INTC")
BOOTSTRAP_STATUSES = {
    "insufficient_matured_data",
    "model_run_incomplete",
    "no_common_oos_predictions",
    "no_usable_matured_rows",
    "insufficient_chronological_training_history",
}


@dataclass(frozen=True)
class ReadinessConfig:
    expected_tickers: tuple[str, ...] = DEFAULT_EXPECTED_TICKERS
    stale_after_hours: float = 6.0
    min_train_sessions: int = 10
    min_validation_sessions: int = 10
    min_test_sessions: int = 5
    market_timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        symbols = tuple(
            dict.fromkeys(
                str(value).strip().upper()
                for value in self.expected_tickers
                if str(value).strip()
            )
        )
        object.__setattr__(self, "expected_tickers", symbols)
        if not symbols:
            raise ValueError("expected_tickers must not be empty")
        if float(self.stale_after_hours) <= 0:
            raise ValueError("stale_after_hours must be positive")
        if min(
            int(self.min_train_sessions),
            int(self.min_validation_sessions),
            int(self.min_test_sessions),
        ) < 1:
            raise ValueError("session requirements must be positive")

    @property
    def target_sessions(self) -> int:
        return (
            int(self.min_train_sessions)
            + int(self.min_validation_sessions)
            + int(self.min_test_sessions)
        )


def summarize_model_dataset(
    dataset: pd.DataFrame,
    *,
    min_train_sessions: int = 10,
    min_validation_sessions: int = 10,
    min_test_sessions: int = 5,
) -> dict:
    """Summarize exact matured model rows without claiming candidate eligibility."""

    requirements = {
        "training": int(min_train_sessions),
        "validation": int(min_validation_sessions),
        "test": int(min_test_sessions),
    }
    if min(requirements.values()) < 1:
        raise ValueError("session requirements must be positive")
    target_sessions = sum(requirements.values())
    frame = dataset.copy()
    if frame.empty:
        return {
            "phase": "collecting",
            "dataset_rows": 0,
            "matured_rows": 0,
            "usable_rows": 0,
            "signal_sessions": 0,
            "matured_sessions": 0,
            "usable_sessions": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "target_sessions": target_sessions,
            "remaining_proxy_sessions": target_sessions,
            "progress_fraction": 0.0,
            "quality_counts": {},
            "requirements": requirements,
        }

    signal_sessions = _session_count(frame)
    matured_mask = _bool_column(frame, "is_matured")
    usable_mask = _bool_column(frame, "usable_for_model")
    matured = frame.loc[matured_mask]
    usable = frame.loc[usable_mask]
    matured_sessions = _session_count(matured)
    usable_sessions = _session_count(usable)
    positive_labels = 0
    negative_labels = 0
    if "target_positive" in usable:
        labels = pd.to_numeric(usable["target_positive"], errors="coerce").dropna()
        positive_labels = int((labels == 1).sum())
        negative_labels = int((labels == 0).sum())

    if usable_sessions < requirements["training"]:
        phase = "building_training_window"
    elif usable_sessions < requirements["training"] + requirements["validation"]:
        phase = "building_validation_window"
    elif usable_sessions < target_sessions:
        phase = "building_test_window"
    else:
        phase = "minimum_window_reached"

    quality_counts = {}
    if "data_quality_flag" in frame:
        quality_counts = {
            str(key): int(value)
            for key, value in frame["data_quality_flag"]
            .fillna("missing")
            .value_counts()
            .sort_index()
            .items()
        }
    return {
        "phase": phase,
        "dataset_rows": int(len(frame)),
        "matured_rows": int(matured_mask.sum()),
        "usable_rows": int(usable_mask.sum()),
        "signal_sessions": signal_sessions,
        "matured_sessions": matured_sessions,
        "usable_sessions": usable_sessions,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "target_sessions": target_sessions,
        "remaining_proxy_sessions": max(0, target_sessions - usable_sessions),
        "progress_fraction": min(1.0, usable_sessions / target_sessions),
        "quality_counts": quality_counts,
        "requirements": requirements,
    }


def build_research_readiness(
    *,
    news_items: list[dict],
    collection_runs: list[dict],
    research_events: list[dict],
    storage: dict,
    now,
    config: ReadinessConfig | None = None,
) -> dict:
    """Build a fail-visible readiness report from append-only operational data."""

    settings = config or ReadinessConfig(
        expected_tickers=_expected_tickers_from_events(research_events)
    )
    timestamp = _aware_utc_timestamp(now, field="now")
    expected = set(settings.expected_tickers)
    captured = {
        str(item.get("ticker", "")).strip().upper()
        for item in news_items
        if str(item.get("ticker", "")).strip()
    }
    missing_tickers = sorted(expected - captured)
    checks: list[dict] = []

    durable = bool(storage.get("durable_for_scheduled_runs"))
    checks.append(
        _check(
            "Durable research store",
            "pass" if durable else "fail",
            "PostgreSQL is configured for scheduled runners."
            if durable
            else "Local SQLite cannot persist across GitHub-hosted scheduled runs.",
        )
    )

    latest_run = collection_runs[0] if collection_runs else None
    latest_completed_at = None
    freshness_hours = None
    if latest_run is None:
        checks.append(_check("Latest collection", "fail", "No collection run is recorded."))
        checks.append(_check("Scheduler freshness", "fail", "No completed run is available."))
    else:
        complete = (
            int(latest_run.get("requested_tickers", 0)) > 0
            and int(latest_run.get("successful_tickers", 0))
            == int(latest_run.get("requested_tickers", 0))
            and int(latest_run.get("failed_tickers", 0)) == 0
        )
        checks.append(
            _check(
                "Latest collection",
                "pass" if complete else "fail",
                (
                    f"{latest_run.get('successful_tickers', 0)}/"
                    f"{latest_run.get('requested_tickers', 0)} tickers succeeded."
                ),
            )
        )
        try:
            latest_completed_at = _aware_utc_timestamp(
                latest_run.get("completed_at"), field="completed_at"
            )
            freshness_hours = float((timestamp - latest_completed_at).total_seconds() / 3_600)
        except ValueError:
            freshness_hours = None
        fresh = freshness_hours is not None and 0 <= freshness_hours <= settings.stale_after_hours
        detail = (
            "Latest collection timestamp is invalid."
            if freshness_hours is None
            else f"Latest collection completed {freshness_hours:.1f} hours ago."
        )
        checks.append(_check("Scheduler freshness", "pass" if fresh else "fail", detail))

    checks.append(
        _check(
            "Expected universe captured",
            "pass" if not missing_tickers else "warn",
            (
                f"All {len(expected)} expected tickers have immutable captures."
                if not missing_tickers
                else "No stored headline yet for: " + ", ".join(missing_tickers)
            ),
        )
    )

    invalid_clock_rows = 0
    non_observed_rows = 0
    for item in news_items:
        try:
            _aware_utc_timestamp(item.get("first_seen_at"), field="first_seen_at")
        except ValueError:
            invalid_clock_rows += 1
        if item.get("availability_basis") != "observed_at":
            non_observed_rows += 1
    clock_ok = invalid_clock_rows == 0 and non_observed_rows == 0
    checks.append(
        _check(
            "Point-in-time captures",
            "pass" if clock_ok else "fail",
            (
                f"{len(news_items)} stored versions use timezone-aware observed_at clocks."
                if clock_ok
                else (
                    f"{invalid_clock_rows} invalid first_seen_at rows; "
                    f"{non_observed_rows} rows lack observed_at provenance."
                )
            ),
        )
    )

    complete_collection_weekdays = _complete_collection_weekdays(
        collection_runs,
        market_timezone=settings.market_timezone,
    )
    latest_cycle = next(
        (
            event
            for event in research_events
            if event.get("event_type") == "automation_run_completed"
        ),
        None,
    )
    latest_cycle_status = (
        None if latest_cycle is None else latest_cycle.get("payload", {}).get("status")
    )
    dataset_readiness = (
        None
        if latest_cycle is None
        else latest_cycle.get("payload", {}).get("dataset_readiness")
    )
    failed_checks = [row for row in checks if row["status"] == "fail"]
    if failed_checks:
        overall_status = "attention_required"
    elif dataset_readiness and dataset_readiness.get("phase") == "minimum_window_reached":
        overall_status = "minimum_window_reached"
    else:
        overall_status = "collecting"

    return {
        "status": overall_status,
        "checks": checks,
        "expected_tickers": sorted(expected),
        "captured_tickers": sorted(captured),
        "missing_tickers": missing_tickers,
        "headline_versions": int(len(news_items)),
        "complete_collection_weekdays": complete_collection_weekdays,
        "target_sessions": settings.target_sessions,
        "latest_collection_at": (
            None if latest_completed_at is None else latest_completed_at.isoformat()
        ),
        "freshness_hours": freshness_hours,
        "latest_cycle_status": latest_cycle_status,
        "latest_cycle_is_bootstrap": latest_cycle_status in BOOTSTRAP_STATUSES,
        "dataset_readiness": dataset_readiness,
    }


def _expected_tickers_from_events(events: list[dict]) -> tuple[str, ...]:
    for event in events:
        if event.get("event_type") != "automation_run_started":
            continue
        tickers = event.get("payload", {}).get("tickers", [])
        normalized = tuple(
            dict.fromkeys(
                str(value).strip().upper() for value in tickers if str(value).strip()
            )
        )
        if normalized:
            return normalized
    return DEFAULT_EXPECTED_TICKERS


def _complete_collection_weekdays(
    collection_runs: list[dict], *, market_timezone: str
) -> int:
    dates = set()
    timezone = ZoneInfo(market_timezone)
    for run in collection_runs:
        if (
            int(run.get("requested_tickers", 0)) <= 0
            or int(run.get("successful_tickers", 0))
            != int(run.get("requested_tickers", 0))
            or int(run.get("failed_tickers", 0)) != 0
        ):
            continue
        try:
            completed = _aware_utc_timestamp(run.get("completed_at"), field="completed_at")
        except ValueError:
            continue
        local_date = completed.tz_convert(timezone).date()
        if local_date.weekday() < 5:
            dates.add(local_date)
    return len(dates)


def _session_count(frame: pd.DataFrame) -> int:
    if frame.empty or "signal_session" not in frame:
        return 0
    sessions = pd.to_datetime(frame["signal_session"], errors="coerce")
    return int(sessions.dropna().dt.normalize().nunique())


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[name].fillna(False).astype(bool)


def _aware_utc_timestamp(value, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.tz_convert("UTC")


def _check(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}
