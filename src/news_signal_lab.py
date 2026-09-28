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
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .abnormal_returns import abnormal_return_summary
from .forward_tracker import trading_session_horizon_end
from .news_signal_validation import (
    audit_feature_availability,
    audit_prediction_availability,
)


MARKET_TIMEZONE = "America/New_York"
REGULAR_CLOSE = time(16, 0)
AVAILABILITY_RULE_VERSION = "observed-to-regular-close.v1"
LOGISTIC_MODEL_VERSION = "tfidf-vader-logit.v1"
TREE_MODEL_VERSION = "tfidf-vader-random-forest.v1"
# Backward-compatible name for callers of the original baseline.
MODEL_VERSION = LOGISTIC_MODEL_VERSION
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
    labels_as_of=None,
    label_settlement_delay_minutes: int = 0,
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
    label_cutoff = None
    if labels_as_of is not None:
        label_cutoff = pd.Timestamp(labels_as_of)
        if label_cutoff.tzinfo is None:
            raise ValueError("labels_as_of must include a timezone")
        label_cutoff = label_cutoff.tz_convert("UTC")
    if int(label_settlement_delay_minutes) < 0:
        raise ValueError("label_settlement_delay_minutes must be non-negative")
    settlement_delay = pd.Timedelta(minutes=int(label_settlement_delay_minutes))
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
        items = sorted(item_versions.values(), key=_item_sort_key)
        titles = [str(item["title"]).strip() for item in items if str(item.get("title", "")).strip()]
        features = extract_headline_features(titles, analyzer=analyzer)
        row = {
            "ticker": ticker,
            "signal_session": signal_session,
            "signal_close_at": session_close_utc(signal_session),
            "as_of_timestamp": session_close_utc(signal_session),
            "first_seen_at_min": min(pd.Timestamp(item["first_seen_at"]) for item in items),
            "first_seen_at_max": max(pd.Timestamp(item["first_seen_at"]) for item in items),
            "data_vintage": _data_vintage(items),
            "document": " [SEP] ".join(titles),
            "horizon_days": int(horizon_days),
            "availability_rule_version": AVAILABILITY_RULE_VERSION,
            "target_definition": (
                "market_sector_blended_abnormal_return"
                if sector_proxy_by_ticker.get(ticker)
                else "beta_adjusted_market_abnormal_return"
            ),
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
        label_available_at = session_close_utc(horizon_end)
        label_settled_at = label_available_at + settlement_delay
        if label_cutoff is not None and label_cutoff < label_settled_at:
            rows.append(
                _unlabeled_row(
                    row,
                    "label_not_settled",
                    horizon_end=horizon_end,
                    label_available_at=label_available_at,
                )
            )
            continue
        if signal_session not in ticker_prices.index or horizon_end not in ticker_prices.index:
            rows.append(
                _unlabeled_row(
                    row,
                    "missing_ticker_session_price",
                    horizon_end=horizon_end,
                    label_available_at=label_available_at,
                )
            )
            continue

        sector_prices = sector_prices_by_ticker.get(ticker)
        sector_proxy = sector_proxy_by_ticker.get(ticker)
        if sector_proxy and (
            sector_prices is None
            or signal_session not in sector_prices.index
            or horizon_end not in sector_prices.index
        ):
            rows.append(
                _unlabeled_row(
                    row,
                    "missing_sector_session_price",
                    horizon_end=horizon_end,
                    label_available_at=label_available_at,
                )
            )
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
                "label_available_at": label_available_at,
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
        "as_of_timestamp",
        "first_seen_at_min",
        "first_seen_at_max",
        "data_vintage",
        "document",
        "horizon_days",
        "availability_rule_version",
        "target_definition",
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

    return _walk_forward_model(
        dataset,
        pipeline_factory=_logistic_pipeline,
        model_version=LOGISTIC_MODEL_VERSION,
        min_train_rows=min_train_rows,
        min_train_sessions=min_train_sessions,
        random_state=random_state,
    )


def walk_forward_tree_challenger(
    dataset: pd.DataFrame,
    *,
    min_train_rows: int = 30,
    min_train_sessions: int = 10,
    random_state: int = 42,
) -> dict:
    """Run a constrained random-forest challenger on the same feature set."""

    return _walk_forward_model(
        dataset,
        pipeline_factory=_tree_pipeline,
        model_version=TREE_MODEL_VERSION,
        min_train_rows=min_train_rows,
        min_train_sessions=min_train_sessions,
        random_state=random_state,
    )


def compare_walk_forward_models(
    dataset: pd.DataFrame,
    *,
    min_train_rows: int = 30,
    min_train_sessions: int = 10,
    random_state: int = 42,
) -> dict:
    """Compare the linear baseline and tree challenger on identical OOS rows."""

    feature_audit = audit_feature_availability(dataset)
    if dataset.empty and feature_audit["status"] == "no_rows":
        return {
            "status": "insufficient_matured_data",
            "feature_availability_audit": feature_audit,
            "prediction_availability_audit": {},
            "models": {},
            "common_predictions": {},
            "comparison": pd.DataFrame(),
        }
    if feature_audit["status"] != "ok":
        return {
            "status": "feature_availability_audit_failed",
            "feature_availability_audit": feature_audit,
            "prediction_availability_audit": {},
            "models": {},
            "common_predictions": {},
            "comparison": pd.DataFrame(),
        }

    results = {
        "logistic": walk_forward_baseline(
            dataset,
            min_train_rows=min_train_rows,
            min_train_sessions=min_train_sessions,
            random_state=random_state,
        ),
        "random_forest": walk_forward_tree_challenger(
            dataset,
            min_train_rows=min_train_rows,
            min_train_sessions=min_train_sessions,
            random_state=random_state,
        ),
    }
    successful = {
        name: result for name, result in results.items() if result["status"] == "ok"
    }
    if len(successful) != len(results):
        return {
            "status": "model_run_incomplete",
            "feature_availability_audit": feature_audit,
            "prediction_availability_audit": {},
            "models": results,
            "common_predictions": {},
            "comparison": pd.DataFrame(),
        }

    key_columns = ["ticker", "signal_session"]
    common_keys: set[tuple] | None = None
    for result in successful.values():
        keys = set(
            result["predictions"][key_columns]
            .itertuples(index=False, name=None)
        )
        common_keys = keys if common_keys is None else common_keys & keys
    if not common_keys:
        return {
            "status": "no_common_oos_predictions",
            "feature_availability_audit": feature_audit,
            "prediction_availability_audit": {},
            "models": results,
            "common_predictions": {},
            "comparison": pd.DataFrame(),
        }

    common_predictions: dict[str, pd.DataFrame] = {}
    comparison_rows: list[dict] = []
    for name, result in successful.items():
        predictions = result["predictions"].copy()
        keys = list(predictions[key_columns].itertuples(index=False, name=None))
        predictions = predictions.loc[[key in common_keys for key in keys]].reset_index(drop=True)
        predictions["model_name"] = name
        common_predictions[name] = predictions
        metrics = _prediction_metrics(predictions)
        comparison_rows.append(
            {
                "model": name,
                "model_version": result["model_version"],
                **metrics,
            }
        )
    prediction_audit = audit_prediction_availability(common_predictions)
    if prediction_audit["status"] != "ok":
        return {
            "status": "prediction_availability_audit_failed",
            "feature_availability_audit": feature_audit,
            "prediction_availability_audit": prediction_audit,
            "models": results,
            "common_predictions": common_predictions,
            "comparison": pd.DataFrame(comparison_rows),
        }
    return {
        "status": "ok",
        "feature_availability_audit": feature_audit,
        "prediction_availability_audit": prediction_audit,
        "models": results,
        "common_predictions": common_predictions,
        "comparison": pd.DataFrame(comparison_rows),
    }


def _walk_forward_model(
    dataset: pd.DataFrame,
    *,
    pipeline_factory,
    model_version: str,
    min_train_rows: int,
    min_train_sessions: int,
    random_state: int,
) -> dict:
    feature_audit = audit_feature_availability(dataset)
    if feature_audit["status"] != "ok":
        result = _empty_walk_forward(
            "feature_availability_audit_failed", model_version=model_version
        )
        result["feature_availability_audit"] = feature_audit
        return result
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
        return _empty_walk_forward("no_usable_matured_rows", model_version=model_version)
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

        pipeline = pipeline_factory(random_state=random_state)
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
                    "signal_close_at": source["signal_close_at"],
                    "as_of_timestamp": source["as_of_timestamp"],
                    "first_seen_at_max": source["first_seen_at_max"],
                    "data_vintage": source["data_vintage"],
                    "availability_rule_version": source["availability_rule_version"],
                    "target_definition": source["target_definition"],
                    "target_positive": int(source["target_positive"]),
                    "target_raw_return": source.get("target_raw_return"),
                    "target_abnormal_return": float(source["target_abnormal_return"]),
                    "horizon_end": source.get("horizon_end"),
                    "horizon_days": source.get("horizon_days"),
                    "label_available_at": source["label_available_at"],
                    "predicted_probability": float(probability),
                    "predicted_positive": int(probability >= 0.5),
                    "historical_positive_rate": train_positive_rate,
                    "historical_rate_prediction": int(train_positive_rate >= 0.5),
                    "train_rows": int(len(train)),
                    "train_sessions": int(train["signal_session"].nunique()),
                    "train_session_start": train["signal_session"].min(),
                    "train_session_end": train["signal_session"].max(),
                    "train_label_cutoff": train_cutoff,
                    "test_close_at": test_close,
                    "model_version": model_version,
                }
            )

    result = pd.DataFrame(predictions)
    if result.empty:
        empty = _empty_walk_forward(
            "insufficient_chronological_training_history", model_version=model_version
        )
        empty["skipped_sessions"] = skipped_sessions
        return empty
    metrics = _prediction_metrics(result)
    return {
        "status": "ok",
        "model_version": model_version,
        "predictions": result,
        "metrics": metrics,
        "feature_availability_audit": feature_audit,
        "skipped_sessions": skipped_sessions,
    }


def _feature_pipeline(model):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
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
    return Pipeline([("features", features), ("model", model)])


def _logistic_pipeline(*, random_state: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
        solver="liblinear",
    )
    return _feature_pipeline(model)


def _tree_pipeline(*, random_state: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    return _feature_pipeline(model)


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


def _unlabeled_row(
    row: dict,
    flag: str,
    *,
    horizon_end=None,
    label_available_at=None,
) -> dict:
    return {
        **row,
        "horizon_end": horizon_end,
        "label_available_at": label_available_at,
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


def _empty_walk_forward(status: str, *, model_version: str = MODEL_VERSION) -> dict:
    return {
        "status": status,
        "model_version": model_version,
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


def _data_vintage(items: list[dict]) -> str:
    """Hash the exact immutable headline versions used in one feature row."""

    payload = []
    for item in items:
        payload.append(
            {
                "content_hash": str(item.get("content_hash") or ""),
                "item_key": str(item.get("item_key") or ""),
                "title": " ".join(str(item.get("title") or "").split()),
                "first_seen_at": pd.Timestamp(item["first_seen_at"]).isoformat(),
            }
        )
    payload = sorted(
        payload,
        key=lambda row: (
            row["first_seen_at"],
            row["content_hash"],
            row["item_key"],
            row["title"],
        ),
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _item_sort_key(item: dict) -> tuple:
    return (
        pd.Timestamp(item["first_seen_at"]).isoformat(),
        str(item.get("content_hash") or ""),
        str(item.get("item_key") or ""),
        " ".join(str(item.get("title") or "").split()),
    )
