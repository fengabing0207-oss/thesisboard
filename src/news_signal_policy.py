"""Validation-only signal-policy selection for News Signal Lab.

This module evaluates selected headline events, not a daily position book. Its
cost-adjusted event return must not be described as portfolio P&L or turnover.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


POLICY_VERSION = "long-event-threshold.v1"
KEY_COLUMNS = ["ticker", "signal_session"]


def evaluate_event_policy(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float,
    round_trip_cost_bps: float,
) -> dict:
    """Evaluate a fixed long-event threshold with one assumed cost per event."""

    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    if float(round_trip_cost_bps) < 0.0:
        raise ValueError("round_trip_cost_bps must be non-negative")
    required = {"predicted_probability", "target_abnormal_return"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")

    frame = predictions.dropna(subset=list(required)).copy()
    selected = frame.loc[
        frame["predicted_probability"].astype(float) >= float(probability_threshold)
    ]
    cost = float(round_trip_cost_bps) / 10_000.0
    result = {
        "row_count": int(len(frame)),
        "selected_count": int(len(selected)),
        "selection_rate": float(len(selected) / len(frame)) if len(frame) else 0.0,
        "positive_event_rate": None,
        "mean_gross_abnormal_return": None,
        "mean_net_abnormal_return": None,
    }
    if selected.empty:
        return result
    returns = selected["target_abnormal_return"].astype(float)
    result.update(
        {
            "positive_event_rate": float((returns > 0.0).mean()),
            "mean_gross_abnormal_return": float(returns.mean()),
            "mean_net_abnormal_return": float((returns - cost).mean()),
        }
    )
    return result


def select_and_evaluate_holdout_policy(
    model_predictions: Mapping[str, pd.DataFrame],
    *,
    thresholds: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    validation_fraction: float = 0.60,
    min_validation_sessions: int = 10,
    min_test_sessions: int = 5,
    min_validation_signals: int = 5,
    max_validation_selection_rate: float = 0.50,
    round_trip_cost_bps: float = 10.0,
) -> dict:
    """Select model/threshold on purged validation data, then lock for test.

    All supplied models are reduced to the same ticker/session rows. The split
    is chronological. Validation rows whose labels were not known before the
    first test close are purged before candidate selection.
    """

    if not model_predictions:
        return _empty_policy("no_model_predictions")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < float(max_validation_selection_rate) <= 1.0:
        raise ValueError("max_validation_selection_rate must be in (0, 1]")
    if int(min_validation_sessions) < 1 or int(min_test_sessions) < 1:
        raise ValueError("minimum session counts must be positive")

    threshold_grid = sorted({float(value) for value in thresholds})
    if not threshold_grid or any(value < 0.0 or value > 1.0 for value in threshold_grid):
        raise ValueError("thresholds must contain values between 0 and 1")

    normalized: dict[str, pd.DataFrame] = {}
    common_keys: set[tuple] | None = None
    required = {
        *KEY_COLUMNS,
        "label_available_at",
        "test_close_at",
        "predicted_probability",
        "target_abnormal_return",
    }
    for name, source in model_predictions.items():
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing required columns: {missing}")
        frame = source.dropna(subset=list(required)).copy()
        frame["signal_session"] = pd.to_datetime(frame["signal_session"]).dt.normalize()
        frame["label_available_at"] = pd.to_datetime(frame["label_available_at"], utc=True)
        frame["test_close_at"] = pd.to_datetime(frame["test_close_at"], utc=True)
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} predictions contain duplicate ticker/session keys")
        keys = set(frame[KEY_COLUMNS].itertuples(index=False, name=None))
        common_keys = keys if common_keys is None else common_keys & keys
        normalized[str(name)] = frame

    if not common_keys:
        return _empty_policy("no_common_oos_predictions")
    for name, frame in normalized.items():
        keys = list(frame[KEY_COLUMNS].itertuples(index=False, name=None))
        normalized[name] = frame.loc[[key in common_keys for key in keys]].sort_values(
            ["signal_session", "ticker"]
        )

    reference = next(iter(normalized.values()))
    sessions = reference["signal_session"].drop_duplicates().sort_values().tolist()
    split_index = int(len(sessions) * float(validation_fraction))
    if split_index < int(min_validation_sessions) or len(sessions) - split_index < int(min_test_sessions):
        result = _empty_policy("insufficient_holdout_sessions")
        result["common_session_count"] = int(len(sessions))
        return result

    validation_sessions = set(sessions[:split_index])
    test_sessions = set(sessions[split_index:])
    first_test_session = pd.Timestamp(sessions[split_index])
    first_test_rows = reference.loc[reference["signal_session"] == first_test_session]
    first_test_close = first_test_rows["test_close_at"].min()

    candidates: list[dict] = []
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, frame in normalized.items():
        validation = frame.loc[
            frame["signal_session"].isin(validation_sessions)
            & (frame["label_available_at"] < first_test_close)
        ]
        test = frame.loc[frame["signal_session"].isin(test_sessions)]
        frames[name] = (validation, test)
        for threshold in threshold_grid:
            metrics = evaluate_event_policy(
                validation,
                probability_threshold=threshold,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            eligible = (
                metrics["selected_count"] >= int(min_validation_signals)
                and metrics["selection_rate"] <= float(max_validation_selection_rate)
                and metrics["mean_net_abnormal_return"] is not None
            )
            candidates.append(
                {
                    "model": name,
                    "probability_threshold": threshold,
                    "eligible": bool(eligible),
                    **metrics,
                }
            )

    candidate_table = pd.DataFrame(candidates)
    eligible = candidate_table.loc[candidate_table["eligible"]].copy()
    if eligible.empty:
        result = _empty_policy("no_eligible_validation_policy")
        result.update(
            {
                "candidates": candidate_table,
                "validation_session_count": int(len(validation_sessions)),
                "test_session_count": int(len(test_sessions)),
            }
        )
        return result

    chosen = eligible.sort_values(
        ["mean_net_abnormal_return", "selected_count", "probability_threshold", "model"],
        ascending=[False, False, False, True],
    ).iloc[0]
    chosen_model = str(chosen["model"])
    threshold = float(chosen["probability_threshold"])
    validation, test = frames[chosen_model]
    test_metrics = evaluate_event_policy(
        test,
        probability_threshold=threshold,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    return {
        "status": "ok",
        "policy_version": POLICY_VERSION,
        "chosen_model": chosen_model,
        "probability_threshold": threshold,
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "first_test_session": first_test_session,
        "first_test_close": first_test_close,
        "validation_session_count": int(len(validation_sessions)),
        "test_session_count": int(len(test_sessions)),
        "purged_validation_rows": int(
            len(reference.loc[reference["signal_session"].isin(validation_sessions)])
            - len(validation)
        ),
        "validation": evaluate_event_policy(
            validation,
            probability_threshold=threshold,
            round_trip_cost_bps=round_trip_cost_bps,
        ),
        "test": test_metrics,
        "candidates": candidate_table,
    }


def _empty_policy(status: str) -> dict:
    return {
        "status": status,
        "policy_version": POLICY_VERSION,
        "candidates": pd.DataFrame(),
    }
