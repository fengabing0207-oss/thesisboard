import pandas as pd
import pytest

from src.news_signal_backtest import (
    run_long_only_position_backtest,
    run_strategy_spec_backtest,
    select_and_evaluate_holdout_strategy,
)
from src.news_signal_lab import session_close_utc
from src.strategy_spec import StrategySpec


def _prediction(ticker, signal, horizon, probability=0.8):
    return {
        "ticker": ticker,
        "signal_session": pd.Timestamp(signal),
        "horizon_end": pd.Timestamp(horizon),
        "predicted_probability": probability,
    }


def test_position_backtest_charges_entry_and_exit_turnover():
    sessions = pd.bdate_range("2026-01-02", periods=4)
    predictions = pd.DataFrame([_prediction("AAA", sessions[0], sessions[2])])
    prices = pd.Series([100.0, 110.0, 121.0, 121.0], index=sessions)
    benchmark = pd.Series([100.0, 100.0, 100.0, 100.0], index=sessions)

    result = run_long_only_position_backtest(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=benchmark,
        probability_threshold=0.7,
        one_way_cost_bps=10,
    )

    assert result["status"] == "ok"
    metrics = result["metrics"]
    assert metrics["selected_event_count"] == 1
    assert metrics["active_return_sessions"] == 2
    assert metrics["total_turnover"] == pytest.approx(2.0)
    assert metrics["round_trip_equivalents"] == pytest.approx(1.0)
    assert metrics["total_transaction_cost"] == pytest.approx(0.002)
    assert result["daily"]["turnover"].tolist() == [1.0, 0.0, 1.0]


def test_overlapping_same_ticker_events_do_not_double_position_or_turnover():
    sessions = pd.bdate_range("2026-01-02", periods=5)
    predictions = pd.DataFrame(
        [
            _prediction("AAA", sessions[0], sessions[2]),
            _prediction("AAA", sessions[1], sessions[3]),
        ]
    )
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=sessions)

    result = run_long_only_position_backtest(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=prices,
        probability_threshold=0.7,
        one_way_cost_bps=5,
    )

    assert result["status"] == "ok"
    assert result["metrics"]["selected_event_count"] == 2
    assert result["metrics"]["total_turnover"] == pytest.approx(2.0)
    assert result["daily"]["gross_exposure"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_backtest_starts_flat_at_requested_holdout_boundary():
    sessions = pd.bdate_range("2026-01-02", periods=5)
    predictions = pd.DataFrame(
        [
            _prediction("AAA", sessions[0], sessions[2]),
            _prediction("AAA", sessions[2], sessions[4]),
        ]
    )
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=sessions)

    result = run_long_only_position_backtest(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=prices,
        probability_threshold=0.7,
        one_way_cost_bps=5,
        start_session=sessions[2],
    )

    assert result["status"] == "ok"
    assert result["selected_signals"]["signal_session"].min() == sessions[2]
    assert result["daily"].iloc[0]["turnover"] == pytest.approx(1.0)


def test_backtest_refuses_to_silently_bridge_missing_active_price():
    sessions = pd.bdate_range("2026-01-02", periods=4)
    predictions = pd.DataFrame([_prediction("AAA", sessions[0], sessions[2])])
    prices = pd.Series([100.0, 121.0], index=[sessions[0], sessions[2]])
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0], index=sessions)

    result = run_long_only_position_backtest(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=benchmark,
        probability_threshold=0.7,
        one_way_cost_bps=5,
    )

    assert result["status"] == "missing_active_prices"
    assert result["issues"]


def test_backtest_reports_no_selected_signals_without_inventing_performance():
    sessions = pd.bdate_range("2026-01-02", periods=3)
    predictions = pd.DataFrame([_prediction("AAA", sessions[0], sessions[1], probability=0.2)])

    result = run_long_only_position_backtest(
        predictions,
        prices_by_ticker={},
        benchmark_prices=pd.Series([100.0, 101.0, 102.0], index=sessions),
        probability_threshold=0.7,
        one_way_cost_bps=5,
    )

    assert result["status"] == "no_selected_signals"
    assert result["metrics"] == {}


def test_strategy_spec_backtest_rejects_prediction_contract_drift():
    sessions = pd.bdate_range("2026-01-02", periods=2)
    predictions = pd.DataFrame(
        [
            {
                **_prediction("AAA", sessions[0], sessions[1]),
                "model_name": "logistic",
                "model_version": "model.v2",
                "availability_rule_version": "observed-to-regular-close.v1",
                "target_definition": "beta_adjusted_market_abnormal_return",
            }
        ]
    )
    spec = StrategySpec(
        model_name="logistic",
        model_version="model.v1",
        probability_threshold=0.7,
        holding_horizon_sessions=1,
        universe=("AAA",),
    )

    with pytest.raises(ValueError, match="model_version does not match"):
        run_strategy_spec_backtest(
            predictions,
            prices_by_ticker={"AAA": pd.Series([100.0, 101.0], index=sessions)},
            benchmark_prices=pd.Series([100.0, 100.0], index=sessions),
            strategy_spec=spec,
        )


def _strategy_fixture():
    sessions = pd.bdate_range("2026-01-02", periods=21)
    returns = [0.02 if index % 2 else -0.02 for index in range(20)]
    prices = [100.0]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    price_series = pd.Series(prices, index=sessions)
    benchmark = pd.Series(100.0, index=sessions)
    frames = {}
    for model in ["logistic", "random_forest"]:
        rows = []
        for index, session in enumerate(sessions[:-1]):
            positive = index % 2 == 1
            probability = (0.8 if positive else 0.2) if model == "logistic" else (
                0.2 if positive else 0.8
            )
            test_close = session_close_utc(session)
            rows.append(
                {
                    "ticker": "AAA",
                    "signal_session": session,
                    "signal_close_at": test_close,
                    "as_of_timestamp": test_close,
                    "first_seen_at_max": test_close - pd.Timedelta(minutes=1),
                    "data_vintage": f"vintage-{index}",
                    "availability_rule_version": "observed-to-regular-close.v1",
                    "target_definition": "beta_adjusted_market_abnormal_return",
                    "model_name": model,
                    "model_version": f"{model}.v1",
                    "horizon_end": sessions[index + 1],
                    "horizon_days": 1,
                    "label_available_at": session_close_utc(sessions[index + 1]),
                    "train_session_start": sessions[0] - pd.Timedelta(days=2),
                    "train_session_end": session - pd.Timedelta(days=1),
                    "train_label_cutoff": test_close - pd.Timedelta(days=1),
                    "test_close_at": test_close,
                    "predicted_probability": probability,
                    "target_abnormal_return": returns[index],
                }
            )
        frames[model] = pd.DataFrame(rows)
    return sessions, frames, price_series, benchmark


def test_strategy_selection_uses_validation_pnl_and_ignores_test_prices():
    sessions, predictions, prices, benchmark = _strategy_fixture()
    result = select_and_evaluate_holdout_strategy(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=benchmark,
        thresholds=[0.5, 0.7],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.6,
        max_validation_average_daily_turnover=1.0,
        one_way_cost_bps=5,
        universe=["AAA", "BBB"],
    )

    assert result["status"] == "ok"
    assert result["chosen_model"] == "logistic"
    assert result["probability_threshold"] == 0.7
    assert result["validation_backtest"]["metrics"]["net_cumulative_return"] > 0
    assert result["strategy_spec"]["model_name"] == "logistic"
    assert result["strategy_spec"]["holding_horizon_sessions"] == 1
    assert result["strategy_spec"]["model_version"] == "logistic.v1"
    assert result["strategy_spec"]["universe"] == ["AAA", "BBB"]
    assert result["calculation_parity_audit"]["status"] == "verified"
    assert result["validation_backtest"]["strategy_spec_id"] == result[
        "test_backtest"
    ]["strategy_spec_id"]

    changed_prices = prices.copy()
    for index in range(12, len(sessions) - 1):
        changed_prices.iloc[index + 1] = changed_prices.iloc[index] * (
            0.8 if index % 2 else 1.2
        )
    changed = select_and_evaluate_holdout_strategy(
        predictions,
        prices_by_ticker={"AAA": changed_prices},
        benchmark_prices=benchmark,
        thresholds=[0.5, 0.7],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.6,
        max_validation_average_daily_turnover=1.0,
        one_way_cost_bps=5,
        universe=["AAA", "BBB"],
    )

    assert changed["chosen_model"] == result["chosen_model"]
    assert changed["probability_threshold"] == result["probability_threshold"]
    assert (
        changed["test_backtest"]["metrics"]["net_cumulative_return"]
        != result["test_backtest"]["metrics"]["net_cumulative_return"]
    )


def test_strategy_selection_enforces_validation_turnover_limit():
    _, predictions, prices, benchmark = _strategy_fixture()
    result = select_and_evaluate_holdout_strategy(
        predictions,
        prices_by_ticker={"AAA": prices},
        benchmark_prices=benchmark,
        thresholds=[0.7],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.6,
        max_validation_average_daily_turnover=0.1,
        one_way_cost_bps=5,
    )

    assert result["status"] == "no_eligible_validation_strategy"
    assert not result["candidates"]["turnover_eligible"].any()
