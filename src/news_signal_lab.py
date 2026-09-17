"""Leakage-aware research helpers for News Signal Lab.

This module deliberately separates four clocks:

1. vendor publication time (metadata only),
2. ``first_seen_at`` (when ThesisBoard could first use the headline),
3. the market-close session at which a daily signal becomes eligible, and
4. ``label_available_at`` (when the forward-return target is actually known).

The walk-forward baseline trains only on rows whose labels were available
before the test-session close. That rule is stricter than a random train/test
split and prevents overlapping forward horizons from leaking into training.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .abnormal_returns import abnormal_return_summary
from .forward_tracker import trading_session_horizon_end


MARKET_TIMEZONE = "America/New_York"
REGULAR_CLOSE = time(16, 0)
AVAILABILITY_RULE_VERSION = "observed-to-regular-close.v1"
MODEL_VERSION = "tfidf-vader-logit.v1"
NUMERIC_FEATURES = [
    "news_count",
    "title_char_mean",
    "vader_compound_mean",
    "vader_compound_max",
    "vader_compound_min",
    "vader_positive_share",
    "vader_negative_share",
    "vader_neutral_share",
]


def availability_session(
    first_seen_at,
    benchmark_sessions: pd.Index,
    *,
    market_timezone: str = MARKET_TIMEZONE,
    regular_close: time = REGULAR_CLOSE,
) -> pd.Timestamp | None:
    """Map an observation timestamp to the earliest eligible EOD session.

    A headline observed before the regular close can enter the same close-to-
    close signal. At or after the close, on weekends, or on market holidays it
    moves to the next benchmark session. ``first_seen_at`` must be timezone-
    aware; silently guessing a timezone would corrupt point-in-time ordering.
    """

    seen = pd.Timestamp(first_seen_at)
    if seen.tzinfo is None:
        raise ValueError("first_seen_at must include a timezone")
    local = seen.tz_convert(ZoneInfo(market_timezone))
    sessions = _session_dates(benchmark_sessions)
    if len(sessions) == 0:
        return None

    local_day = pd.Timestamp(local.date())
    same_day = local_day in sessions
    if same_day and local.time().replace(tzinfo=None) < regular_close:
        return local_day
    later = sessions[sessions > local_day]
    return None if len(later) == 0 else pd.Timestamp(later[0])


def session_close_utc(
    session,
    *,
    market_timezone: str = MARKET_TIMEZONE,
    regular_close: time = REGULAR_CLOSE,
) -> pd.Timestamp:
    day = pd.Timestamp(session).date()
    local = pd.Timestamp.combine(day, regular_close).tz_localize(ZoneInfo(market_timezone))
    return local.tz_convert("UTC")


def extract_headline_features(headlines: list[str], *, analyzer=None) -> dict:
    titles = [" ".join(str(value).split()).strip() for value in headlines or []]
    titles = [value for value in titles if value]
    if analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    compounds = [float(analyzer.polarity_scores(title)["compound"]) for title in titles]
    if not compounds:
        return {
            "news_count": 0,
            "title_char_mean": 0.0,
            "vader_compound_mean": 0.0,
            "vader_compound_max": 0.0,
            "vader_compound_min": 0.0,
            "vader_positive_share": 0.0,
            "vader_negative_share": 0.0,
            "vader_neutral_share": 0.0,
        }
    scores = np.asarray(compounds, dtype=float)
    positive = scores >= 0.05
    negative = scores <= -0.05
    neutral = ~(positive | negative)
    return {
        "news_count": int(len(titles)),
        "title_char_mean": float(np.mean([len(value) for value in titles])),
        "vader_compound_mean": float(scores.mean()),
        "vader_compound_max": float(scores.max()),
        "vader_compound_min": float(scores.min()),
        "vader_positive_share": float(positive.mean()),
        "vader_negative_share": float(negative.mean()),
        "vader_neutral_share": float(neutral.mean()),
    }


def build_news_return_dataset(
    news_items: list[dict],
    *,
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    horizon_days: int = 1,
    sector_prices_by_ticker: dict[str, pd.Series] | None = None,
    sector_proxy_by_ticker: dict[str, str] | None = None,
    analyzer=None,
) -> pd.DataFrame:
    """Aggregate captured headlines by ticker/session and attach forward labels."""

    benchmark = _clean_prices(benchmark_prices)
    if benchmark.empty:
        raise ValueError("benchmark price history is required")
    sector_prices_by_ticker = {
        str(key).strip().upper(): _clean_prices(value)
        for key, value in (sector_prices_by_ticker or {}).items()
    }
    sector_proxy_by_ticker = {
        str(key).strip().upper(): str(value).strip().upper()
        for key, value in (sector_proxy_by_ticker or {}).items()
    }
    grouped: dict[tuple[str, pd.Timestamp], dict[str, dict]] = {}
    for item in news_items or []:
        if not isinstance(item, dict) or not item.get("title") or not item.get("first_seen_at"):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        signal_session = availability_session(item["first_seen_at"], benchmark.index)
        if signal_session is None:
            continue
        identity = str(item.get("item_key") or item.get("content_hash") or item["title"])
        session_items = grouped.setdefault((ticker, signal_session), {})
        previous = session_items.get(identity)
        if previous is None or pd.Timestamp(item["first_seen_at"]) > pd.Timestamp(previous["first_seen_at"]):
            session_items[identity] = item

    rows: list[dict] = []
    latest_benchmark_session = pd.Timestamp(benchmark.index.max())
    normalized_prices = {
        str(symbol).strip().upper(): _clean_prices(series)
        for symbol, series in prices_by_ticker.items()
    }

    for (ticker, signal_session), item_versions in sorted(grouped.items()):
        items = list(item_versions.values())
        titles = [str(item["title"]).strip() for item in items if str(item.get("title", "")).strip()]
        features = extract_headline_features(titles, analyzer=analyzer)
        row = {
            "ticker": ticker,
            "signal_session": signal_session,
            "signal_close_at": session_close_utc(signal_session),
            "first_seen_at_min": min(pd.Timestamp(item["first_seen_at"]) for item in items),
            "first_seen_at_max": max(pd.Timestamp(item["first_seen_at"]) for item in items),
            "document": " [SEP] ".join(titles),
            "horizon_days": int(horizon_days),
            "availability_rule_version": AVAILABILITY_RULE_VERSION,
            **features,
        }

        ticker_prices = normalized_prices.get(ticker)
        if ticker_prices is None or ticker_prices.empty:
            rows.append(_unlabeled_row(row, "missing_ticker_prices"))
            continue
        try:
            horizon_end = trading_session_horizon_end(signal_session, horizon_days, benchmark.index)
        except ValueError:
            rows.append(_unlabeled_row(row, "unmatured_horizon"))
            continue
        if latest_benchmark_session < horizon_end:
            rows.append(_unlabeled_row(row, "unmatured_horizon", horizon_end=horizon_end))
            continue
        if signal_session not in ticker_prices.index or horizon_end not in ticker_prices.index:
            rows.append(_unlabeled_row(row, "missing_ticker_session_price", horizon_end=horizon_end))
            continue

        sector_prices = sector_prices_by_ticker.get(ticker)
        sector_proxy = sector_proxy_by_ticker.get(ticker)
        if sector_proxy and (
            sector_prices is None
            or signal_session not in sector_prices.index
            or horizon_end not in sector_prices.index
        ):
            rows.append(_unlabeled_row(row, "missing_sector_session_price", horizon_end=horizon_end))
            continue
        summary = abnormal_return_summary(
            ticker_prices=ticker_prices,
            benchmark_prices=benchmark,
            start_date=signal_session,
            end_date=horizon_end,
            sector_prices=sector_prices,
            sector_proxy=sector_proxy,
            beta_estimation_end=signal_session,
        )
        target = summary["combined_abnormal_return"]
        quality = summary["data_quality_flag"]
        row.update(
            {
                "horizon_end": horizon_end,
                "label_available_at": session_close_utc(horizon_end),
                "target_raw_return": summary["raw_return"],
                "target_abnormal_return": target,
                "target_positive": None if target is None else int(target > 0),
                "beta": summary["beta"],
                "beta_estimation_end": summary["beta_estimation_end"],
                "sector_proxy": sector_proxy,
                "data_quality_flag": quality,
                "is_matured": True,
                "usable_for_model": target is not None and quality == "ok",
            }
        )
        rows.append(row)

    columns = [
        "ticker",
        "signal_session",
        "signal_close_at",
        "first_seen_at_min",
        "first_seen_at_max",
        "document",
        "horizon_days",
        "availability_rule_version",
        *NUMERIC_FEATURES,
        "horizon_end",
        "label_available_at",
        "target_raw_return",
        "target_abnormal_return",
        "target_positive",
        "beta",
        "beta_estimation_end",
        "sector_proxy",
        "data_quality_flag",
        "is_matured",
        "usable_for_model",
    ]
    return pd.DataFrame(rows, columns=columns)


def walk_forward_baseline(
    dataset: pd.DataFrame,
    *,
    min_train_rows: int = 30,
    min_train_sessions: int = 10,
    random_state: int = 42,
) -> dict:
    """Run an expanding-window TF-IDF + VADER logistic baseline.

    For each test session, the model is refit using only labels available before
    that session's close. Vectorization, scaling, and classification are all fit
    inside the training window.
    """

    required = {
        "signal_session",
        "label_available_at",
        "document",
        "target_positive",
        "target_abnormal_return",
        "usable_for_model",
        *NUMERIC_FEATURES,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    frame = dataset.loc[dataset["usable_for_model"].fillna(False)].copy()
    frame = frame.dropna(subset=["label_available_at", "target_positive", "target_abnormal_return"])
    if frame.empty:
        return _empty_walk_forward("no_usable_matured_rows")
    frame["signal_session"] = pd.to_datetime(frame["signal_session"]).dt.normalize()
    frame["label_available_at"] = pd.to_datetime(frame["label_available_at"], utc=True)
    frame["target_positive"] = frame["target_positive"].astype(int)
    frame = frame.sort_values(["signal_session", "ticker"]).reset_index(drop=True)

    predictions: list[dict] = []
    skipped_sessions: list[str] = []
    for test_session in frame["signal_session"].drop_duplicates().sort_values():
        test_close = session_close_utc(test_session)
        train = frame.loc[frame["label_available_at"] < test_close]
        test = frame.loc[frame["signal_session"] == test_session]
        if (
            len(train) < int(min_train_rows)
            or train["signal_session"].nunique() < int(min_train_sessions)
            or train["target_positive"].nunique() < 2
        ):
            skipped_sessions.append(pd.Timestamp(test_session).date().isoformat())
            continue

        pipeline = _baseline_pipeline(random_state=random_state)
        try:
            pipeline.fit(train, train["target_positive"])
            probabilities = pipeline.predict_proba(test)[:, 1]
        except ValueError:
            skipped_sessions.append(pd.Timestamp(test_session).date().isoformat())
            continue
        train_cutoff = train["label_available_at"].max()
        train_positive_rate = float(train["target_positive"].mean())
        for (_, source), probability in zip(test.iterrows(), probabilities):
            predictions.append(
                {
                    "ticker": source["ticker"],
                    "signal_session": source["signal_session"],
                    "target_positive": int(source["target_positive"]),
                    "target_abnormal_return": float(source["target_abnormal_return"]),
                    "predicted_probability": float(probability),
                    "predicted_positive": int(probability >= 0.5),
                    "historical_positive_rate": train_positive_rate,
                    "historical_rate_prediction": int(train_positive_rate >= 0.5),
                    "train_rows": int(len(train)),
                    "train_sessions": int(train["signal_session"].nunique()),
                    "train_label_cutoff": train_cutoff,
                    "test_close_at": test_close,
                    "model_version": MODEL_VERSION,
                }
            )

    result = pd.DataFrame(predictions)
    if result.empty:
        empty = _empty_walk_forward("insufficient_chronological_training_history")
        empty["skipped_sessions"] = skipped_sessions
        return empty
    metrics = _prediction_metrics(result)
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "predictions": result,
        "metrics": metrics,
        "skipped_sessions": skipped_sessions,
    }


def _baseline_pipeline(*, random_state: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5_000,
                    sublinear_tf=True,
                ),
                "document",
            ),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _prediction_metrics(predictions: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    actual = predictions["target_positive"].astype(int)
    probability = predictions["predicted_probability"].astype(float)
    predicted = predictions["predicted_positive"].astype(int)
    baseline_probability = predictions["historical_positive_rate"].astype(float)
    baseline_prediction = predictions["historical_rate_prediction"].astype(int)
    accuracy = float(accuracy_score(actual, predicted))
    metrics = {
        "prediction_count": int(len(predictions)),
        "test_session_count": int(predictions["signal_session"].nunique()),
        "directional_accuracy": accuracy,
        "historical_rate_accuracy": float(accuracy_score(actual, baseline_prediction)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "historical_rate_brier": float(brier_score_loss(actual, baseline_probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "roc_auc": None,
        "spearman_ic": None,
    }
    if actual.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(actual, probability))
    ic = probability.corr(predictions["target_abnormal_return"].astype(float), method="spearman")
    if not pd.isna(ic):
        metrics["spearman_ic"] = float(ic)
    return metrics


def _unlabeled_row(row: dict, flag: str, *, horizon_end=None) -> dict:
    return {
        **row,
        "horizon_end": horizon_end,
        "label_available_at": None,
        "target_raw_return": None,
        "target_abnormal_return": None,
        "target_positive": None,
        "beta": None,
        "beta_estimation_end": None,
        "sector_proxy": None,
        "data_quality_flag": flag,
        "is_matured": False,
        "usable_for_model": False,
    }


def _empty_walk_forward(status: str) -> dict:
    return {
        "status": status,
        "model_version": MODEL_VERSION,
        "predictions": pd.DataFrame(),
        "metrics": {},
        "skipped_sessions": [],
    }


def _session_dates(index: pd.Index) -> pd.DatetimeIndex:
    values = pd.to_datetime(pd.Index(index), errors="coerce", utc=True)
    values = values[~values.isna()]
    if getattr(values, "tz", None) is not None:
        values = values.tz_convert(None)
    return pd.DatetimeIndex(sorted(set(pd.Timestamp(value).normalize() for value in values)))


def _clean_prices(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    result = pd.Series(series).dropna()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    result.index = result.index.normalize()
    result = result.sort_index()
    return result[~result.index.duplicated(keep="last")]
