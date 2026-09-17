import pandas as pd
import pytest

from src.news_signal_lab import session_close_utc
from src.news_signal_policy import evaluate_event_policy, select_and_evaluate_holdout_policy


def _policy_predictions() -> dict[str, pd.DataFrame]:
    sessions = pd.bdate_range("2026-01-02", periods=21)
    frames = {}
    for model in ["logistic", "random_forest"]:
        rows = []
        for index, session in enumerate(sessions[:-1]):
            positive = index % 2 == 1
            probability = (0.80 if positive else 0.20) if model == "logistic" else (
                0.20 if positive else 0.80
            )
            label_session = sessions[index + 1]
            test_close = session_close_utc(session)
            rows.append(
                {
                    "ticker": "DEMO",
                    "signal_session": session,
                    "signal_close_at": test_close,
                    "as_of_timestamp": test_close,
                    "first_seen_at_max": test_close - pd.Timedelta(minutes=1),
                    "data_vintage": f"vintage-{index}",
                    "availability_rule_version": "observed-to-regular-close.v1",
                    "target_definition": "beta_adjusted_market_abnormal_return",
                    "model_name": model,
                    "model_version": f"{model}.v1",
                    "horizon_end": label_session,
                    "label_available_at": session_close_utc(label_session),
                    "train_session_start": sessions[0] - pd.Timedelta(days=1),
                    "train_session_end": session - pd.Timedelta(days=1),
                    "train_label_cutoff": test_close - pd.Timedelta(days=1),
                    "test_close_at": test_close,
                    "predicted_probability": probability,
                    "target_abnormal_return": 0.02 if positive else -0.02,
                }
            )
        frames[model] = pd.DataFrame(rows)
    return frames


def test_event_policy_subtracts_assumed_cost_once_per_selected_event():
    predictions = pd.DataFrame(
        {
            "predicted_probability": [0.8, 0.7, 0.2],
            "target_abnormal_return": [0.02, -0.01, 0.50],
        }
    )

    result = evaluate_event_policy(
        predictions,
        probability_threshold=0.6,
        round_trip_cost_bps=10,
    )

    assert result["selected_count"] == 2
    assert result["mean_gross_abnormal_return"] == pytest.approx(0.005)
    assert result["mean_net_abnormal_return"] == pytest.approx(0.004)


def test_holdout_policy_purges_boundary_label_and_never_selects_on_test():
    predictions = _policy_predictions()
    result = select_and_evaluate_holdout_policy(
        predictions,
        thresholds=[0.5, 0.7],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.6,
        round_trip_cost_bps=10,
    )

    assert result["status"] == "ok"
    assert result["chosen_model"] == "logistic"
    assert result["probability_threshold"] == 0.7
    assert result["purged_validation_rows"] == 1
    assert result["validation"]["mean_net_abnormal_return"] == pytest.approx(0.019)
    assert result["availability_audit"]["status"] == "ok"
    assert result["chronological_windows"]["validation"]["end_session"] < result[
        "chronological_windows"
    ]["test"]["start_session"]

    changed = {name: frame.copy() for name, frame in predictions.items()}
    test_sessions = sorted(changed["logistic"]["signal_session"].unique())[12:]
    for frame in changed.values():
        frame.loc[frame["signal_session"].isin(test_sessions), "target_abnormal_return"] *= -100
    changed_result = select_and_evaluate_holdout_policy(
        changed,
        thresholds=[0.5, 0.7],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.6,
        round_trip_cost_bps=10,
    )

    assert changed_result["chosen_model"] == result["chosen_model"]
    assert changed_result["probability_threshold"] == result["probability_threshold"]
    assert changed_result["test"] != result["test"]


def test_holdout_policy_reports_when_frequency_constraint_has_no_candidate():
    result = select_and_evaluate_holdout_policy(
        _policy_predictions(),
        thresholds=[0.5],
        validation_fraction=0.6,
        min_validation_sessions=8,
        min_test_sessions=4,
        min_validation_signals=3,
        max_validation_selection_rate=0.1,
    )

    assert result["status"] == "no_eligible_validation_policy"
    assert not result["candidates"].empty


def test_holdout_policy_fails_closed_when_train_cutoff_leaks():
    predictions = _policy_predictions()
    predictions["logistic"].loc[0, "train_label_cutoff"] = predictions["logistic"].loc[
        0, "test_close_at"
    ]

    result = select_and_evaluate_holdout_policy(predictions)

    assert result["status"] == "prediction_availability_audit_failed"
    assert result["availability_audit"]["status"] == "failed"


def test_holdout_policy_fails_closed_when_models_use_different_targets():
    predictions = _policy_predictions()
    predictions["random_forest"].loc[0, "target_abnormal_return"] = 0.99

    result = select_and_evaluate_holdout_policy(predictions)

    assert result["status"] == "prediction_availability_audit_failed"
    checks = result["availability_audit"]["checks"]
    mismatch = checks.loc[
        checks["check"] == "common_oos_data_contract", "violation_count"
    ].sum()
    assert mismatch == 1
