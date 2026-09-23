import numpy as np
import pandas as pd

from src.news_signal_lab import (
    MODEL_VERSION,
    NUMERIC_FEATURES,
    TREE_MODEL_VERSION,
    availability_session,
    build_news_return_dataset,
    compare_walk_forward_models,
    extract_headline_features,
    session_close_utc,
    walk_forward_baseline,
    walk_forward_tree_challenger,
)
from src.news_signal_validation import audit_feature_availability


def test_availability_session_uses_first_seen_and_regular_close():
    sessions = pd.to_datetime(["2026-09-17", "2026-09-18", "2026-09-21"])

    assert availability_session("2026-09-18T15:59:00-04:00", sessions) == pd.Timestamp("2026-09-18")
    assert availability_session("2026-09-18T16:00:00-04:00", sessions) == pd.Timestamp("2026-09-21")
    assert availability_session("2026-09-19T10:00:00-04:00", sessions) == pd.Timestamp("2026-09-21")


def test_headline_features_separate_positive_and_negative_language():
    positive = extract_headline_features(["Company reports strong profit growth and record demand"])
    negative = extract_headline_features(["Company warns of fraud losses and bankruptcy risk"])

    assert positive["vader_compound_mean"] > negative["vader_compound_mean"]
    assert positive["news_count"] == 1
    assert negative["vader_negative_share"] == 1.0


def _price_fixture():
    sessions = pd.bdate_range("2026-01-02", periods=55)
    pattern = np.array([0.004, -0.002, 0.003, 0.001, -0.001] * 11)
    benchmark = pd.Series(100 * np.cumprod(1 + pattern), index=sessions)
    ticker_returns = 1.2 * pattern + np.array([0.0005, -0.0003, 0.0002, 0.0001, -0.0002] * 11)
    ticker = pd.Series(80 * np.cumprod(1 + ticker_returns), index=sessions)
    ticker.iloc[41:] = ticker.iloc[41:] * 1.05
    return sessions, ticker, benchmark


def test_dataset_uses_first_seen_not_vendor_timestamp_and_builds_forward_label():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    news = [
        {
            "ticker": "NVDA",
            "title": "Demand expands",
            "published_at": "2025-01-01T12:00:00Z",
            "first_seen_at": session_close_utc(signal_day) - pd.Timedelta(minutes=1),
        }
    ]

    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    )

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["signal_session"] == signal_day
    assert row["as_of_timestamp"] == session_close_utc(signal_day)
    assert len(row["data_vintage"]) == 64
    assert row["horizon_end"] == sessions[41]
    assert row["label_available_at"] == session_close_utc(sessions[41])
    assert row["beta_estimation_end"] == signal_day.isoformat()
    assert row["data_quality_flag"] == "ok"
    assert bool(row["usable_for_model"]) is True
    assert row["target_abnormal_return"] > 0


def test_dataset_moves_after_close_capture_to_next_session():
    sessions, ticker, benchmark = _price_fixture()
    observed_day = sessions[39]
    news = [
        {
            "ticker": "NVDA",
            "title": "After-close update",
            "first_seen_at": session_close_utc(observed_day),
        }
    ]
    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    )
    assert dataset.iloc[0]["signal_session"] == sessions[40]


def test_dataset_refuses_shifted_label_when_ticker_session_is_missing():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    news = [
        {
            "ticker": "NVDA",
            "title": "Demand expands",
            "first_seen_at": session_close_utc(signal_day) - pd.Timedelta(minutes=1),
        }
    ]
    ticker = ticker.drop(sessions[41])
    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    )
    assert dataset.iloc[0]["data_quality_flag"] == "missing_ticker_session_price"
    assert bool(dataset.iloc[0]["usable_for_model"]) is False


def test_dataset_does_not_mature_a_label_before_the_close_is_settled():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    horizon_end = sessions[41]
    news = [
        {
            "ticker": "NVDA",
            "title": "Demand expands",
            "first_seen_at": session_close_utc(signal_day) - pd.Timedelta(minutes=1),
        }
    ]

    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
        labels_as_of=session_close_utc(horizon_end) + pd.Timedelta(minutes=30),
        label_settlement_delay_minutes=90,
    )

    row = dataset.iloc[0]
    assert row["data_quality_flag"] == "label_not_settled"
    assert row["label_available_at"] == session_close_utc(horizon_end)
    assert bool(row["is_matured"]) is False
    assert bool(row["usable_for_model"]) is False
    assert pd.isna(row["target_positive"])


def test_dataset_matures_a_label_after_the_settlement_delay():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    horizon_end = sessions[41]
    news = [
        {
            "ticker": "NVDA",
            "title": "Demand expands",
            "first_seen_at": session_close_utc(signal_day) - pd.Timedelta(minutes=1),
        }
    ]

    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
        labels_as_of=session_close_utc(horizon_end) + pd.Timedelta(minutes=90),
        label_settlement_delay_minutes=90,
    )

    assert dataset.iloc[0]["data_quality_flag"] == "ok"
    assert bool(dataset.iloc[0]["usable_for_model"]) is True


def test_dataset_uses_latest_article_version_within_a_session():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    close = session_close_utc(signal_day)
    news = [
        {
            "ticker": "NVDA",
            "item_key": "wire:id:123",
            "title": "Initial headline",
            "first_seen_at": close - pd.Timedelta(minutes=30),
        },
        {
            "ticker": "NVDA",
            "item_key": "wire:id:123",
            "title": "Corrected headline",
            "first_seen_at": close - pd.Timedelta(minutes=15),
        },
    ]
    dataset = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    )
    assert dataset.iloc[0]["news_count"] == 1
    assert dataset.iloc[0]["document"] == "Corrected headline"


def test_dataset_vintage_and_document_are_stable_across_input_order():
    sessions, ticker, benchmark = _price_fixture()
    signal_day = sessions[40]
    close = session_close_utc(signal_day)
    news = [
        {
            "ticker": "NVDA",
            "item_key": "wire:2",
            "title": "Second captured headline",
            "first_seen_at": close - pd.Timedelta(minutes=10),
        },
        {
            "ticker": "NVDA",
            "item_key": "wire:1",
            "title": "First captured headline",
            "first_seen_at": close - pd.Timedelta(minutes=20),
        },
    ]

    first = build_news_return_dataset(
        news,
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    ).iloc[0]
    second = build_news_return_dataset(
        list(reversed(news)),
        prices_by_ticker={"NVDA": ticker},
        benchmark_prices=benchmark,
        horizon_days=1,
    ).iloc[0]

    assert first["data_vintage"] == second["data_vintage"]
    assert first["document"] == second["document"]


def _walk_forward_frame() -> pd.DataFrame:
    sessions = pd.bdate_range("2026-01-02", periods=23)
    rows = []
    for index, session in enumerate(sessions[:-1]):
        positive = index % 2
        signal_close = session_close_utc(session)
        row = {
            "ticker": "NVDA" if index % 3 else "AAPL",
            "signal_session": session,
            "signal_close_at": signal_close,
            "as_of_timestamp": signal_close,
            "first_seen_at_max": signal_close - pd.Timedelta(minutes=1),
            "data_vintage": f"vintage-{index}",
            "availability_rule_version": "observed-to-regular-close.v1",
            "target_definition": "beta_adjusted_market_abnormal_return",
            "horizon_days": 1,
            "horizon_end": sessions[index + 1],
            "label_available_at": session_close_utc(sessions[index + 1]),
            "document": "profit growth strong" if positive else "loss warning weak",
            "target_positive": positive,
            "target_abnormal_return": 0.02 if positive else -0.02,
            "usable_for_model": True,
        }
        row.update({name: 0.0 for name in NUMERIC_FEATURES})
        row["news_count"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_fits_only_on_labels_available_before_test_close():
    result = walk_forward_baseline(
        _walk_forward_frame(),
        min_train_rows=6,
        min_train_sessions=6,
    )

    assert result["status"] == "ok"
    assert result["model_version"] == MODEL_VERSION
    predictions = result["predictions"]
    assert not predictions.empty
    assert (predictions["train_label_cutoff"] < predictions["test_close_at"]).all()
    assert (predictions["first_seen_at_max"] < predictions["as_of_timestamp"]).all()
    assert result["metrics"]["directional_accuracy"] >= result["metrics"]["historical_rate_accuracy"]


def test_walk_forward_refuses_unusable_rows():
    frame = _walk_forward_frame()
    frame["usable_for_model"] = False
    result = walk_forward_baseline(frame, min_train_rows=2, min_train_sessions=2)
    assert result["status"] == "no_usable_matured_rows"
    assert result["predictions"].empty


def test_tree_challenger_uses_same_chronological_guards():
    result = walk_forward_tree_challenger(
        _walk_forward_frame(),
        min_train_rows=6,
        min_train_sessions=6,
    )

    assert result["status"] == "ok"
    assert result["model_version"] == TREE_MODEL_VERSION
    predictions = result["predictions"]
    assert not predictions.empty
    assert (predictions["train_label_cutoff"] < predictions["test_close_at"]).all()


def test_model_comparison_uses_identical_oos_rows():
    result = compare_walk_forward_models(
        _walk_forward_frame(),
        min_train_rows=6,
        min_train_sessions=6,
    )

    assert result["status"] == "ok"
    logistic_keys = set(
        result["common_predictions"]["logistic"][["ticker", "signal_session"]].itertuples(
            index=False, name=None
        )
    )
    tree_keys = set(
        result["common_predictions"]["random_forest"][["ticker", "signal_session"]].itertuples(
            index=False, name=None
        )
    )
    assert logistic_keys == tree_keys
    assert set(result["comparison"]["model"]) == {"logistic", "random_forest"}
    assert result["feature_availability_audit"]["status"] == "ok"
    assert result["prediction_availability_audit"]["status"] == "ok"


def test_model_comparison_classifies_empty_bootstrap_as_insufficient_data():
    result = compare_walk_forward_models(
        _walk_forward_frame().iloc[0:0],
        min_train_rows=6,
        min_train_sessions=6,
    )

    assert result["status"] == "insufficient_matured_data"
    assert result["feature_availability_audit"]["status"] == "no_rows"
    assert result["models"] == {}


def test_model_comparison_does_not_treat_missing_schema_as_bootstrap():
    result = compare_walk_forward_models(pd.DataFrame())

    assert result["status"] == "feature_availability_audit_failed"
    assert result["feature_availability_audit"]["status"] == (
        "missing_required_columns"
    )


def test_model_comparison_keeps_real_availability_failure_fail_closed():
    frame = _walk_forward_frame()
    frame.loc[0, "first_seen_at_max"] = frame.loc[0, "as_of_timestamp"]

    result = compare_walk_forward_models(
        frame,
        min_train_rows=6,
        min_train_sessions=6,
    )

    assert result["status"] == "feature_availability_audit_failed"
    assert result["feature_availability_audit"]["status"] == "failed"
    assert result["models"] == {}


def test_feature_availability_audit_rejects_post_as_of_observation():
    frame = _walk_forward_frame()
    frame.loc[0, "first_seen_at_max"] = frame.loc[0, "as_of_timestamp"]

    audit = audit_feature_availability(frame)

    assert audit["status"] == "failed"
    check = audit["checks"].set_index("check")
    assert check.loc["feature_observed_at_or_after_as_of", "violation_count"] == 1
