import pandas as pd
import pytest

from src.research_readiness import (
    ReadinessConfig,
    build_research_readiness,
    summarize_model_dataset,
)


def _dataset(session_count: int, *, usable_session_count: int) -> pd.DataFrame:
    sessions = pd.bdate_range("2026-01-02", periods=session_count)
    rows = []
    for index, session in enumerate(sessions):
        usable = index < usable_session_count
        rows.append(
            {
                "ticker": "AAA",
                "signal_session": session,
                "is_matured": usable,
                "usable_for_model": usable,
                "target_positive": index % 2 if usable else None,
                "data_quality_flag": "ok" if usable else "unmatured_horizon",
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("usable_sessions", "expected_phase"),
    [
        (0, "building_training_window"),
        (10, "building_validation_window"),
        (20, "building_test_window"),
        (25, "minimum_window_reached"),
    ],
)
def test_dataset_readiness_exposes_chronological_phase(usable_sessions, expected_phase):
    result = summarize_model_dataset(
        _dataset(25, usable_session_count=usable_sessions)
    )

    assert result["phase"] == expected_phase
    assert result["usable_sessions"] == usable_sessions
    assert result["target_sessions"] == 25
    assert result["remaining_proxy_sessions"] == 25 - usable_sessions
    assert result["positive_labels"] + result["negative_labels"] == usable_sessions


def test_empty_dataset_is_a_normal_collecting_state():
    result = summarize_model_dataset(pd.DataFrame())

    assert result["phase"] == "collecting"
    assert result["dataset_rows"] == 0
    assert result["remaining_proxy_sessions"] == 25


def test_operational_readiness_uses_latest_watchlist_and_exact_dataset_progress():
    items = [
        {
            "ticker": ticker,
            "first_seen_at": "2026-09-16T18:00:00Z",
            "availability_basis": "observed_at",
        }
        for ticker in ("AAA", "BBB")
    ]
    runs = [
        {
            "completed_at": "2026-09-17T18:00:00Z",
            "requested_tickers": 2,
            "successful_tickers": 2,
            "failed_tickers": 0,
        },
        {
            "completed_at": "2026-09-16T18:00:00Z",
            "requested_tickers": 2,
            "successful_tickers": 2,
            "failed_tickers": 0,
        },
    ]
    exact = summarize_model_dataset(_dataset(25, usable_session_count=12))
    events = [
        {
            "event_type": "automation_run_completed",
            "payload": {
                "status": "model_run_incomplete",
                "dataset_readiness": exact,
            },
        },
        {
            "event_type": "automation_run_started",
            "payload": {"tickers": ["aaa", "BBB"]},
        },
    ]

    report = build_research_readiness(
        news_items=items,
        collection_runs=runs,
        research_events=events,
        storage={"durable_for_scheduled_runs": True},
        now="2026-09-17T19:00:00Z",
    )

    assert report["status"] == "collecting"
    assert report["expected_tickers"] == ["AAA", "BBB"]
    assert report["complete_collection_weekdays"] == 2
    assert report["freshness_hours"] == 1.0
    assert report["latest_cycle_is_bootstrap"] is True
    assert report["dataset_readiness"]["usable_sessions"] == 12
    assert {check["status"] for check in report["checks"]} == {"pass"}


def test_operational_readiness_fails_visible_on_stale_partial_or_invalid_data():
    report = build_research_readiness(
        news_items=[
            {
                "ticker": "AAA",
                "first_seen_at": "2026-09-16 12:00:00",
                "availability_basis": "vendor_published_at",
            }
        ],
        collection_runs=[
            {
                "completed_at": "2026-09-16T12:00:00Z",
                "requested_tickers": 2,
                "successful_tickers": 1,
                "failed_tickers": 1,
            }
        ],
        research_events=[],
        storage={"durable_for_scheduled_runs": False},
        now="2026-09-17T19:00:00Z",
        config=ReadinessConfig(expected_tickers=("AAA", "BBB")),
    )

    assert report["status"] == "attention_required"
    assert report["missing_tickers"] == ["BBB"]
    failed = {row["check"] for row in report["checks"] if row["status"] == "fail"}
    assert failed == {
        "Durable research store",
        "Latest collection",
        "Scheduler freshness",
        "Point-in-time captures",
    }


def test_readiness_config_rejects_zero_thresholds():
    with pytest.raises(ValueError, match="stale_after_hours"):
        ReadinessConfig(stale_after_hours=0)
    with pytest.raises(ValueError, match="session requirements"):
        ReadinessConfig(min_test_sessions=0)
