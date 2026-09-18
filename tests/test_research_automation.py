import numpy as np
import pandas as pd
import pytest

import src.research_automation as automation
from src.news_signal_lab import session_close_utc
from src.news_store import ingest_news_items, list_research_events
from src.price_provider import DemoPriceProvider
from src.strategy_spec import StrategySpec


def _seed_store_and_prices(db_path, *, historical_price_bump=0.0):
    sessions = pd.bdate_range("2026-01-02", periods=60)
    benchmark_returns = np.array([0.002, -0.001, 0.0015, 0.0005, -0.0005] * 12)
    benchmark = pd.Series(100 * np.cumprod(1 + benchmark_returns), index=sessions)
    ticker_returns = 1.1 * benchmark_returns
    ticker_returns[46] += 0.04
    ticker = pd.Series(80 * np.cumprod(1 + ticker_returns), index=sessions)
    ticker.iloc[20] += float(historical_price_bump)
    observed = session_close_utc(sessions[45]) - pd.Timedelta(minutes=1)
    ingest_news_items(
        ticker="AAA",
        observed_at=observed,
        items=[{"title": "Strong demand and record growth", "provider_item_id": "seed-1"}],
        db_path=db_path,
    )
    return DemoPriceProvider({"AAA": ticker, "SPY": benchmark})


def _successful_policy():
    spec = StrategySpec(
        model_name="logistic",
        model_version="tfidf-logit.v1",
        probability_threshold=0.65,
        holding_horizon_sessions=1,
        universe=("AAA",),
    )
    return {
        "status": "ok",
        "strategy_spec_id": spec.spec_id,
        "strategy_spec": spec.to_dict(),
        "chronological_windows": {
            "validation": {"start_session": "2026-01-01", "end_session": "2026-02-01"},
            "test": {"start_session": "2026-02-02", "end_session": "2026-03-01"},
        },
        "validation_backtest": {"metrics": {"net_cumulative_return": 0.04}},
        "test_backtest": {"metrics": {"net_cumulative_return": 0.01}},
        "calculation_parity_audit": {"status": "verified"},
        "candidates": pd.DataFrame(
            [{"model": "logistic", "probability_threshold": 0.65, "eligible": True}]
        ),
    }


def _configure_success(monkeypatch):
    monkeypatch.setattr(
        automation,
        "compare_walk_forward_models",
        lambda dataset, **kwargs: {
            "status": "ok",
            "common_predictions": {},
            "comparison": pd.DataFrame(
                [{"model": "logistic", "directional_accuracy": 0.6}]
            ),
        },
    )
    monkeypatch.setattr(
        automation,
        "select_and_evaluate_holdout_strategy",
        lambda predictions, **kwargs: _successful_policy(),
    )


def test_cycle_proposes_without_promoting_and_deduplicates_identical_vintage(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "news.db"
    provider = _seed_store_and_prices(db_path)
    _configure_success(monkeypatch)
    kwargs = {
        "tickers": ["AAA"],
        "db_path": db_path,
        "fetcher": lambda ticker: [],
        "price_provider": provider,
        "now": "2026-09-17T16:00:00Z",
    }

    first = automation.run_research_cycle(run_id="cycle-1", **kwargs)
    second = automation.run_research_cycle(run_id="cycle-2", **kwargs)

    assert first["status"] == "candidate_proposed"
    assert second["status"] == "candidate_unchanged"
    candidate = automation.get_research_event(first["candidate_event_id"], db_path)
    assert candidate["payload"]["selection_basis"] == "validation_only"
    assert candidate["payload"]["auto_promoted"] is False
    assert candidate["payload"]["promotion_status"] == "pending_human_review"
    assert candidate["payload"]["held_out_diagnostics_not_used_for_selection"] == {
        "net_cumulative_return": 0.01
    }
    assert len(candidate["payload"]["price_vintage"]) == 64
    assert candidate["payload"]["model_comparison_oos"][0]["model"] == "logistic"
    assert candidate["payload"]["validation_candidate_audit"][0]["eligible"] is True
    assert candidate["payload"]["dataset_readiness"]["usable_sessions"] == 1
    assert first["dataset_readiness"]["phase"] == "building_training_window"
    assert list_research_events(
        db_path, event_type="strategy_promotion_approved"
    ) == []


def test_revised_price_input_creates_a_fresh_candidate_vintage(tmp_path, monkeypatch):
    db_path = tmp_path / "news.db"
    first_provider = _seed_store_and_prices(db_path)
    _configure_success(monkeypatch)
    first = automation.run_research_cycle(
        ["AAA"],
        db_path=db_path,
        fetcher=lambda ticker: [],
        price_provider=first_provider,
        now="2026-09-17T16:00:00Z",
        run_id="cycle-1",
    )
    revised_provider = _seed_store_and_prices(db_path, historical_price_bump=0.01)
    revised = automation.run_research_cycle(
        ["AAA"],
        db_path=db_path,
        fetcher=lambda ticker: [],
        price_provider=revised_provider,
        now="2026-09-17T16:00:00Z",
        run_id="cycle-2",
    )

    assert first["strategy_spec_id"] == revised["strategy_spec_id"]
    assert first["dataset_vintage"] != revised["dataset_vintage"]
    assert revised["status"] == "candidate_proposed"


def test_cycle_runs_real_walk_forward_selection_end_to_end(tmp_path):
    db_path = tmp_path / "news.db"
    sessions = pd.bdate_range("2025-01-02", periods=100)
    market_returns = np.asarray([0.002, -0.001, 0.0015, -0.0005, 0.001] * 20)
    ticker_returns = 1.1 * market_returns
    items = []
    for index in range(35, 90):
        positive = index % 2 == 1
        ticker_returns[index + 1] += 0.012 if positive else -0.012
        items.append(
            {
                "title": (
                    "Strong demand and profit growth"
                    if positive
                    else "Weak demand and loss warning"
                ),
                "provider_item_id": f"synthetic-{index}",
                "first_seen_at": session_close_utc(sessions[index])
                - pd.Timedelta(minutes=30),
            }
        )
    ingest_news_items(
        ticker="DEMO",
        items=items,
        observed_at="2025-01-02T16:00:00Z",
        db_path=db_path,
    )
    benchmark = pd.Series(100 * np.cumprod(1 + market_returns), index=sessions)
    ticker = pd.Series(80 * np.cumprod(1 + ticker_returns), index=sessions)

    result = automation.run_research_cycle(
        ["DEMO"],
        config=automation.ResearchCycleConfig(
            min_train_rows=20,
            min_train_sessions=20,
            min_validation_sessions=10,
            min_test_sessions=5,
            min_validation_signals=5,
            max_validation_selection_rate=0.6,
        ),
        db_path=db_path,
        fetcher=lambda ticker: [],
        price_provider=DemoPriceProvider({"DEMO": ticker, "SPY": benchmark}),
        now="2026-09-17T16:00:00Z",
        run_id="cycle-real-pipeline",
    )

    assert result["status"] == "candidate_proposed"
    event = automation.get_research_event(result["candidate_event_id"], db_path)
    assert event["payload"]["calculation_parity"]["status"] == "verified"
    assert event["payload"]["strategy_spec"]["universe"] == ["DEMO"]
    assert event["payload"]["auto_promoted"] is False


def test_promotion_is_manual_hash_checked_and_single_decision(tmp_path, monkeypatch):
    db_path = tmp_path / "news.db"
    provider = _seed_store_and_prices(db_path)
    _configure_success(monkeypatch)
    cycle = automation.run_research_cycle(
        ["AAA"],
        db_path=db_path,
        fetcher=lambda ticker: [],
        price_provider=provider,
        now="2026-09-17T16:00:00Z",
        run_id="cycle-1",
    )

    promotion = automation.record_strategy_promotion(
        cycle["candidate_event_id"],
        decision="approved",
        decided_by="research-owner",
        note="Reviewed validation and holdout diagnostics.",
        db_path=db_path,
    )

    assert promotion["automatic"] is False
    assert promotion["event_type"] == "strategy_promotion_approved"
    assert automation.current_approved_strategy(db_path=db_path)["decided_by"] == (
        "research-owner"
    )
    with pytest.raises(ValueError, match="already has a promotion decision"):
        automation.record_strategy_promotion(
            cycle["candidate_event_id"],
            decision="rejected",
            decided_by="second-reviewer",
            db_path=db_path,
        )


def test_promotion_rejects_tampered_strategy_hash(tmp_path):
    db_path = tmp_path / "news.db"
    spec = _successful_policy()["strategy_spec"]
    candidate_id = automation.append_research_event(
        event_type="strategy_candidate_proposed",
        run_id="cycle-tampered",
        subject_id="fake-id",
        payload={"strategy_spec": spec, "strategy_spec_id": "fake-id"},
        db_path=db_path,
    )

    with pytest.raises(ValueError, match="hash does not match"):
        automation.record_strategy_promotion(
            candidate_id,
            decision="approved",
            decided_by="research-owner",
            db_path=db_path,
        )


def test_cycle_stops_before_calibration_when_collection_is_incomplete(
    tmp_path, monkeypatch
):
    def fail_fetch(ticker):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        automation,
        "compare_walk_forward_models",
        lambda *args, **kwargs: pytest.fail("calibration must not run"),
    )
    result = automation.run_research_cycle(
        ["AAA"],
        db_path=tmp_path / "news.db",
        fetcher=fail_fetch,
        now="2026-09-17T16:00:00Z",
        run_id="cycle-failed-collection",
    )

    assert result["status"] == "collection_incomplete"
    assert result["failed_tickers"] == 1
    assert "RuntimeError" in result["errors"]["AAA"]
    assert list_research_events(
        tmp_path / "news.db", event_type="strategy_candidate_proposed"
    ) == []


def test_cycle_logs_empty_bootstrap_as_insufficient_matured_data(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "news.db"
    provider = _seed_store_and_prices(db_path)
    empty_dataset = pd.DataFrame(
        columns=[
            "ticker",
            "signal_session",
            "signal_close_at",
            "as_of_timestamp",
            "first_seen_at_max",
            "data_vintage",
            "availability_rule_version",
            "target_definition",
            "label_available_at",
            "usable_for_model",
        ]
    )
    monkeypatch.setattr(
        automation,
        "build_news_return_dataset",
        lambda *args, **kwargs: empty_dataset,
    )

    result = automation.run_research_cycle(
        ["AAA"],
        db_path=db_path,
        fetcher=lambda ticker: [],
        price_provider=provider,
        now="2026-09-17T16:00:00Z",
        run_id="cycle-bootstrap",
    )

    assert result["status"] == "insufficient_matured_data"
    assert result["dataset_rows"] == 0
    assert result["candidate_event_id"] is None
    completed = list_research_events(
        db_path,
        event_type="automation_run_completed",
        run_id="cycle-bootstrap",
    )
    assert completed[0]["payload"]["status"] == "insufficient_matured_data"


def test_cycle_logs_unhandled_failure_without_swallowing_it(tmp_path, monkeypatch):
    db_path = tmp_path / "news.db"
    _seed_store_and_prices(db_path)
    database_url = "postgresql://alice:top-secret@db.example/test"
    monkeypatch.setenv("THESISBOARD_DATABASE_URL", database_url)

    class FailingPriceProvider:
        def get_history(self, symbols, start, end):
            raise RuntimeError(f"price boundary failed for {database_url}")

    with pytest.raises(RuntimeError, match="REDACTED_DATABASE_URL"):
        automation.run_research_cycle(
            ["AAA"],
            db_path=db_path,
            fetcher=lambda ticker: [],
            price_provider=FailingPriceProvider(),
            now="2026-09-17T16:00:00Z",
            run_id="cycle-price-failure",
        )

    failures = list_research_events(
        db_path,
        event_type="automation_run_failed",
        run_id="cycle-price-failure",
    )
    assert failures[0]["payload"] == {
        "error": "price boundary failed for [REDACTED_DATABASE_URL]",
        "error_type": "RuntimeError",
        "status": "failed",
    }


def test_cycle_config_rejects_unsafe_ranges():
    with pytest.raises(ValueError, match="selection_rate"):
        automation.ResearchCycleConfig(max_validation_selection_rate=1.1)
    with pytest.raises(ValueError, match="one_way_cost_bps"):
        automation.ResearchCycleConfig(one_way_cost_bps=-1)
    assert automation.ResearchCycleConfig(benchmark_symbol=" spy ").benchmark_symbol == "SPY"
