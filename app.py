from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import market_news
from src.demo_validation_data import EXPLICIT_OUTCOME_FIELDS, prepare_validation_lab_data
from src.journal import append_record, load_records
from src.news_signal_backtest import select_and_evaluate_holdout_strategy
from src.news_signal_lab import build_news_return_dataset, compare_walk_forward_models
from src.news_store import (
    ingest_news_items,
    list_collection_runs,
    list_news_items,
    list_research_events,
    news_capture_coverage,
    news_store_summary,
    storage_backend_info,
)
from src.pre_trade_check import (
    EventType,
    InstrumentType,
    Level,
    PlannedAction,
    ThesisDecision,
    evaluate_pre_trade_risk,
)
from src.price_provider import CachingPriceProvider, YFinancePriceProvider
from src.research_automation import current_approved_strategy


ROOT = Path(__file__).resolve().parent
NEWS_PRICE_CACHE = ROOT / "data" / "price_cache"


SNAPSHOT_COLUMNS = [
    "ticker",
    "theme",
    "classification",
    "score",
    "horizon_days",
    "benchmark",
    "sector_proxy",
    "price_at_creation",
    "rule_version",
    "created_at",
]

OUTCOME_COLUMNS = [
    "ticker",
    "classification",
    "horizon_days",
    "raw_return",
    "market_return",
    "beta",
    "beta_fallback_used",
    "combined_abnormal_return",
    "trade_hit",
    "watch_followthrough",
    "avoided_bad_trade",
    "false_negative",
    "max_drawdown",
    "max_runup",
    "data_quality_flag",
]

GROUP_COLUMNS = [
    "horizon_days",
    "classification",
    "sample_size",
    "sample_positive_rate",
    "base_rate",
    "trade_hit_rate",
    "excess_trade_hit_rate",
    "watch_followthrough_rate",
    "avoided_bad_trade_rate",
    "false_negatives",
    "average_forward_abnormal_return",
]


def main() -> None:
    st.set_page_config(page_title="ThesisBoard", layout="wide")
    st.title("ThesisBoard")
    st.caption("User-controlled thematic trading research. Guarded automation, human promotion.")

    page = st.sidebar.radio(
        "Pages",
        [
            "Home",
            "Pre-Trade Check",
            "Market News",
            "News Signal Lab",
            "Validation Lab",
            "Methodology",
            "Roadmap",
        ],
    )

    if page == "Home":
        render_home()
    elif page == "Pre-Trade Check":
        render_pre_trade_check()
    elif page == "Market News":
        render_market_news()
    elif page == "News Signal Lab":
        render_news_signal_lab()
    elif page == "Validation Lab":
        render_validation_lab()
    elif page == "Methodology":
        render_methodology()
    elif page == "Roadmap":
        render_roadmap()


def render_home() -> None:
    st.header("Home")
    st.write(
        "ThesisBoard is a user-controlled thematic trading research dashboard. "
        "The core idea is to test whether the market validates a thematic relationship."
    )
    st.write(
        "This deployable review branch focuses on the validation spine: horizons, abnormal returns, "
        "forward-return tracking, hit-rate/base-rate comparison, and reproducible rule-based validation."
    )
    st.info(
        "Current branch focus: point-in-time validation and a guarded research loop. "
        "Automation may propose a challenger but cannot promote one."
    )
    st.warning("ThesisBoard is not financial advice and does not execute trades.")


def render_pre_trade_check() -> None:
    st.header("Pre-Trade Check")
    st.caption(
        "A behavioral risk-control checklist to run before entering a trade — not investment advice. "
        "Fill the core fields in under a minute; everything else is optional."
    )

    action_options = [member.value for member in PlannedAction]
    instrument_options = [member.value for member in InstrumentType]
    event_options = [member.value for member in EventType]
    level_options = [member.value for member in Level]

    with st.form("pre_trade_check_form"):
        st.subheader("1) Trade setup")
        col_a, col_b, col_c = st.columns(3)
        ticker = col_a.text_input("Ticker", placeholder="e.g. NVDA")
        planned_action = col_b.selectbox("Planned action", action_options, index=action_options.index("buy"))
        instrument_type = col_c.selectbox("Instrument", instrument_options, index=instrument_options.index("stock"))
        col_d, col_e, col_f = st.columns(3)
        event_type = col_d.selectbox("Event type", event_options, index=event_options.index("earnings"))
        horizon_days = col_e.number_input("Horizon (trading days)", min_value=1, value=5, step=1)
        market_expectation = col_f.selectbox(
            "Market expectation", level_options, index=level_options.index("medium")
        )
        high_runup = st.checkbox("Recent large run-up / possible chase?")

        st.subheader("2) Thesis and risk")
        entry_thesis = st.text_area("Entry thesis", placeholder="Why enter now?")
        risk_thesis = st.text_area("Risk thesis", placeholder="What's the bear case / what could go wrong?")

        st.subheader("3) Sizing and discipline")
        col_g, col_h = st.columns(2)
        max_loss = col_g.number_input("Max loss you'll accept", min_value=0.0, value=0.0, step=50.0)
        position_size = col_h.number_input(
            "Position size (% of portfolio)", min_value=0.0, max_value=100.0, value=0.0, step=0.5
        )
        invalidation_rule = st.text_input(
            "Invalidation rule", placeholder="What would prove this wrong / force an exit?"
        )

        with st.expander("Advanced / optional context"):
            theme = st.text_input("Theme", placeholder="(optional) e.g. AI Infrastructure")
            st.caption("Left blank, theme is recorded as 'unspecified'.")
            confidence = st.selectbox("Confidence", level_options, index=level_options.index("medium"))
            notes = st.text_area("Notes", placeholder="(optional)")

        submitted = st.form_submit_button("Build pre-trade check")

    if submitted:
        try:
            decision = ThesisDecision(
                ticker=ticker,
                theme=theme.strip() or "unspecified",
                event_type=event_type,
                planned_action=planned_action,
                instrument_type=instrument_type,
                horizon_days=int(horizon_days),
                market_expectation=market_expectation,
                entry_thesis=entry_thesis,
                risk_thesis=risk_thesis,
                max_loss=float(max_loss),
                invalidation_rule=invalidation_rule,
                confidence=confidence,
                position_size=float(position_size),
                notes=notes,
            )
        except ValueError as exc:
            st.error(f"Could not build the pre-trade check: {exc}")
            st.session_state.pop("pretrade_last_result", None)
        else:
            # max_loss in this form is an unlabeled amount, so the % -> implied-move
            # arithmetic stays off; the checkbox feeds the runtime high_runup signal.
            flags, verdict = evaluate_pre_trade_risk(decision, high_runup=bool(high_runup))
            st.session_state["pretrade_last_result"] = {
                "decision": decision.to_dict(),
                "evaluation": {
                    "verdict": verdict.verdict.value,
                    "active_risk_flags": flags.active(),
                    "reasons": list(verdict.reasons),
                    "heuristic_only": True,
                    "not_financial_advice": True,
                },
            }

    # Persisted across reruns so the separate Save button (a fresh run) still has
    # the last-built result to write.
    result = st.session_state.get("pretrade_last_result")
    if result is None:
        st.caption("A heuristic risk check appears after you build the decision.")
    else:
        _render_built_result(result)

    _render_saved_journal()


def _render_built_result(result: dict) -> None:
    st.success("Pre-trade check built for this session.")
    _render_decision_summary(result["decision"])
    _render_risk_output(result["evaluation"])
    st.subheader("Serialized decision (JSON preview)")
    st.json(result["decision"])
    if st.button("Save this record"):
        append_record(decision=result["decision"], evaluation=result["evaluation"])
        st.success("Saved to journal.")


def _render_saved_journal() -> None:
    st.subheader("Saved records")
    records = load_records()
    if not records:
        st.caption("No records yet. Build a check and click Save to start your journal.")
        return
    rows = [
        {
            "saved_at": record.get("saved_at", ""),
            "ticker": record.get("decision", {}).get("ticker", ""),
            "verdict": record.get("evaluation", {}).get("verdict", ""),
        }
        for record in records
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_risk_output(evaluation: dict) -> None:
    st.subheader("Heuristic risk check")
    st.caption("Heuristic behavioral check only — not financial advice and not a predictive signal.")
    st.write(f"**Verdict:** `{evaluation['verdict']}`")
    active = evaluation.get("active_risk_flags", [])
    st.write("**Risk flags:** " + (", ".join(active) if active else "none"))
    st.write("**Reasons:**")
    for reason in evaluation.get("reasons", []):
        st.markdown(f"- {reason}")


def _render_decision_summary(data: dict) -> None:
    st.subheader("Summary")
    rows = [
        ("Ticker", data["ticker"]),
        ("Planned action", data["planned_action"]),
        ("Instrument", data["instrument_type"]),
        ("Event type", data["event_type"]),
        ("Horizon (days)", data["horizon_days"]),
        ("Max loss", data["max_loss"]),
        ("Invalidation rule", data["invalidation_rule"] or "(none)"),
        ("Position size (%)", data["position_size"]),
        ("Confidence", data["confidence"]),
        ("Market expectation", data["market_expectation"]),
        ("Theme", data["theme"]),
        ("Status", data["status"]),
    ]
    # Stringify the value column so the mixed-type table serializes cleanly.
    table = pd.DataFrame([(field, "" if value is None else str(value)) for field, value in rows], columns=["Field", "Value"])
    st.table(table)


def render_market_news() -> None:
    st.header("Market News")
    st.caption("Market context only, not a buy/sell signal.")

    ticker = st.text_input("Ticker", placeholder="e.g. NVDA").strip().upper()

    _render_market_news_snapshot(ticker)
    normalized = _render_market_news_headlines(ticker)
    _render_market_news_capture(ticker, normalized)
    _render_market_news_ai_summary(normalized)


def _render_market_news_snapshot(ticker: str) -> None:
    st.subheader("Market snapshot")
    symbols = list(market_news.MARKET_SYMBOLS) + ([ticker] if ticker else [])
    try:
        prices = market_news.fetch_market_prices(symbols)
    except Exception:
        st.info("Market snapshot unavailable: could not load price data from yfinance.")
        return

    snapshot = market_news.build_market_snapshot(prices, ticker=ticker or None)

    columns = st.columns(len(market_news.MARKET_SYMBOLS))
    for column, symbol in zip(columns, market_news.MARKET_SYMBOLS):
        info = snapshot["market"][symbol]
        label = market_news.MARKET_LABELS.get(symbol, symbol)
        if not info.get("available"):
            column.metric(label, "n/a")
            column.caption("Unavailable")
            continue
        column.metric(label, f"{info['latest']:.2f}")
        column.line_chart(info["series"].tail(63), height=120)

    if not ticker:
        st.caption("Enter a ticker above for ticker-specific context.")
        return

    metrics = snapshot["ticker"]
    if not metrics or not metrics.get("available"):
        st.info(f"No price data available for {ticker} from yfinance.")
        return

    st.markdown(f"**{ticker}**")
    st.line_chart(metrics["series"].tail(63), height=180)
    row = st.columns(4)
    row[0].metric("5D return", _fmt_pct(metrics["return_5d"]))
    row[1].metric("20D return", _fmt_pct(metrics["return_20d"]))
    row[2].metric("60D return", _fmt_pct(metrics["return_60d"]))
    row[3].metric("From recent high", _fmt_pct(metrics["distance_from_high"]))

    _render_next_earnings(ticker)


def _render_next_earnings(ticker: str) -> None:
    try:
        earnings_date = market_news.fetch_next_earnings_date(ticker)
    except Exception:
        earnings_date = None
    if earnings_date is None:
        st.caption("Next earnings: unavailable.")
        return
    days = market_news.days_until(earnings_date)
    when = "" if days is None else f" (~{days}d)"
    st.caption(f"Next earnings: {str(earnings_date.date())}{when}")


def _render_market_news_headlines(ticker: str) -> list:
    st.subheader("News headlines")
    if not ticker:
        st.caption("Enter a ticker above to load recent headlines.")
        return []
    try:
        raw = market_news.fetch_ticker_news(ticker)
    except Exception:
        raw = []
    normalized = market_news.normalize_news(raw)
    if not normalized:
        st.write("No recent news available from yfinance.")
        return []

    for item in normalized:
        title = item.get("title") or "(no title)"
        meta_parts = [part for part in (item.get("publisher"), item.get("timestamp")) if part]
        meta = " · ".join(meta_parts)
        st.markdown(f"**{title}**")
        if meta:
            st.caption(meta)
        if item.get("link"):
            st.link_button("Open original", item["link"])
    return normalized


def _render_market_news_ai_summary(normalized_news: list) -> None:
    st.subheader("AI topic summary")
    st.caption(
        "Topic summary of public headlines only — not sentiment prediction, not investment advice. "
        "Headlines can mislead; read the originals."
    )
    headlines = market_news.headlines_from_news(normalized_news)
    if not headlines:
        st.info("AI topic summary unavailable: no headlines to summarize.")
        return
    if not market_news.anthropic_api_key_available():
        st.info("AI topic summary unavailable: no local Anthropic API key is configured.")
        return
    try:
        client = market_news.get_anthropic_client()
        summary = market_news.summarize_headlines(headlines, client=client)
    except Exception:
        st.info("AI topic summary unavailable: could not reach the local Anthropic API.")
        return
    st.write(summary)


def _render_market_news_capture(ticker: str, normalized_news: list) -> None:
    st.subheader("Research capture")
    st.caption(
        "Save the headlines exactly as ThesisBoard sees them now. The first-seen timestamp, "
        "not the vendor publication time, controls future point-in-time research."
    )
    if not ticker or not normalized_news:
        st.info("Research capture unavailable until a ticker has headlines.")
        return
    if st.button("Capture current headlines", key="capture_market_news"):
        result = ingest_news_items(ticker=ticker, items=normalized_news, provider="yfinance")
        st.success(
            f"Captured {result['inserted']} new headline versions; "
            f"{result['existing']} were already stored and {result['skipped']} were skipped."
        )


def _render_research_automation_status() -> None:
    st.subheader("Scheduled research loop")
    backend = storage_backend_info()
    events = list_research_events(limit=100)
    completed = next(
        (event for event in events if event["event_type"] == "automation_run_completed"),
        None,
    )
    proposed = next(
        (event for event in events if event["event_type"] == "strategy_candidate_proposed"),
        None,
    )
    decided_candidate_ids = {
        int(event["payload"]["candidate_event_id"])
        for event in events
        if event["event_type"].startswith("strategy_promotion_")
        and event["payload"].get("candidate_event_id") is not None
    }
    pending = proposed is not None and int(proposed["id"]) not in decided_candidate_ids
    champion = current_approved_strategy()

    row = st.columns(4)
    row[0].metric("Research store", backend["backend"])
    row[1].metric(
        "Latest cycle",
        "never run" if completed is None else completed["payload"].get("status", "unknown"),
    )
    row[2].metric("Latest challenger", "pending review" if pending else "none pending")
    row[3].metric(
        "Approved champion",
        "none" if champion is None else champion["strategy_spec_id"].split(":")[-1],
    )
    if backend["durable_for_scheduled_runs"]:
        st.success(
            "Durable storage is configured. The scheduler can collect and validate when its repository "
            "enable switch is on. Candidate promotion always remains a separate human action."
        )
    else:
        st.warning(
            "This process is using local SQLite. Manual capture works, but scheduled runners refuse this "
            "ephemeral backend; configure THESISBOARD_DATABASE_URL before enabling automation."
        )
    if pending:
        st.info(
            f"Candidate event #{proposed['id']} is awaiting explicit approval or rejection. "
            "Its held-out metrics are diagnostic and were not used for automated selection."
        )
    if events:
        with st.expander("Automation event audit"):
            audit = pd.DataFrame(events)[
                ["id", "event_time", "event_type", "run_id", "subject_id"]
            ]
            st.dataframe(audit, width="stretch", hide_index=True)
            st.caption(
                "Events are append-only. Held-out results are diagnostics, not inputs to automatic "
                "selection, and no event path performs automatic promotion."
            )


def render_news_signal_lab() -> None:
    st.header("News Signal Lab")
    st.caption(
        "Point-in-time headline research with close-to-close labels and chronological evaluation — "
        "not a trading recommendation."
    )
    st.warning(
        "The lab never backfills availability from a vendor publication timestamp. A headline becomes "
        "eligible only when ThesisBoard first captured it, so a new installation needs time to build history."
    )

    summary = news_store_summary()
    metrics = st.columns(3)
    metrics[0].metric("Captured headline versions", int(summary["item_count"] or 0))
    metrics[1].metric("Tickers", int(summary["ticker_count"] or 0))
    coverage = "n/a"
    if summary.get("first_seen_at") and summary.get("last_seen_at"):
        first = pd.Timestamp(summary["first_seen_at"]).date()
        last = pd.Timestamp(summary["last_seen_at"]).date()
        coverage = f"{first} → {last}"
    metrics[2].metric("Observed coverage", coverage)

    _render_research_automation_status()

    ticker_coverage = pd.DataFrame(news_capture_coverage())
    collection_runs = list_collection_runs(limit=10)
    if not ticker_coverage.empty:
        st.subheader("Capture readiness")
        st.caption(
            "Independent observed dates matter more than headline count. Repeated headlines on one day do not "
            "create a defensible chronological sample."
        )
        st.dataframe(ticker_coverage, width="stretch", hide_index=True)
    if collection_runs:
        with st.expander("Collection run audit"):
            runs = pd.DataFrame(collection_runs).drop(columns=["errors"], errors="ignore")
            st.dataframe(runs, width="stretch", hide_index=True)

    items = list_news_items()
    if not items:
        st.info(
            "No captured headlines yet. Use Market News → Capture current headlines, "
            "or enable the scheduled research loop, to start the dataset."
        )
        return

    latest = pd.DataFrame(items[:100])
    st.subheader("Latest immutable captures")
    st.dataframe(
        latest[["first_seen_at", "ticker", "title", "publisher", "published_at", "provider"]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Chronological model comparison")
    st.caption(
        "A logistic baseline and constrained random-forest challenger use the same TF-IDF/VADER features "
        "and identical out-of-sample rows. Each refit sees only labels known before the test close."
    )
    config = st.columns(5)
    horizon_days = config[0].selectbox("Forward horizon (sessions)", [1, 3], index=0)
    min_train_rows = int(
        config[1].number_input("Minimum chronological training rows", min_value=10, value=30, step=5)
    )
    round_trip_cost_bps = float(
        config[2].number_input("Assumed round-trip cost (bps)", min_value=0, value=10, step=1)
    )
    max_selection_rate = float(
        config[3].number_input("Max validation selection (%)", min_value=5, max_value=100, value=50, step=5)
    ) / 100.0
    max_daily_turnover = float(
        config[4].number_input("Max validation daily turnover (%)", min_value=5, max_value=200, value=100, step=5)
    ) / 100.0
    if not st.button("Run locked evaluation", key="run_news_signal_evaluation"):
        st.caption("No model is trained automatically. Run only when you want a dated research snapshot.")
        return

    tickers = sorted({item["ticker"] for item in items})
    first_seen = min(pd.Timestamp(item["first_seen_at"]) for item in items)
    start = first_seen.tz_convert("UTC").tz_localize(None).normalize() - pd.Timedelta(days=240)
    end = pd.Timestamp(date.today())
    provider = CachingPriceProvider(
        YFinancePriceProvider(),
        NEWS_PRICE_CACHE,
        snapshot_id=date.today().isoformat(),
    )
    try:
        bundle = provider.get_history([*tickers, "SPY"], start, end)
    except Exception as exc:
        st.error(f"Could not build the research dataset from yfinance: {exc}")
        return
    benchmark = bundle.prices.get("SPY")
    if benchmark is None or benchmark.empty:
        st.error("SPY benchmark history is unavailable; the lab will not fall back to raw returns.")
        return

    dataset = build_news_return_dataset(
        items,
        prices_by_ticker={ticker: bundle.prices[ticker] for ticker in tickers if ticker in bundle.prices},
        benchmark_prices=benchmark,
        horizon_days=int(horizon_days),
    )
    st.subheader("Dataset audit")
    quality = (
        dataset["data_quality_flag"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")
        if not dataset.empty
        else pd.DataFrame(columns=["status", "rows"])
    )
    st.dataframe(quality, width="stretch", hide_index=True)
    if bundle.missing_symbols:
        st.warning("Missing price histories: " + ", ".join(bundle.missing_symbols))

    comparison = compare_walk_forward_models(dataset, min_train_rows=min_train_rows)
    if comparison["status"] != "ok":
        model_statuses = ", ".join(
            f"{name}: {result['status'].replace('_', ' ')}"
            for name, result in comparison["models"].items()
        )
        st.info(
            "No fair two-model out-of-sample comparison yet. "
            f"{model_statuses}. Keep collecting headlines and let their horizons mature."
        )
        return

    feature_audit = comparison["feature_availability_audit"]
    prediction_audit = comparison["prediction_availability_audit"]
    st.subheader("Point-in-time leakage audit")
    audit_row = st.columns(4)
    audit_row[0].metric("Feature audit", feature_audit["status"])
    audit_row[1].metric("Feature rows", feature_audit["row_count"])
    audit_row[2].metric("Prediction audit", prediction_audit["status"])
    audit_row[3].metric("Leakage violations", prediction_audit["violation_count"])
    st.caption(
        "Every feature row is tied to an as-of close and immutable data-vintage hash. OOS predictions "
        "must use features observed before that close, training labels known before the test close, and "
        "targets that mature strictly afterward. A failed audit stops policy selection."
    )
    with st.expander("Availability audit details"):
        st.write("Feature snapshot checks")
        st.dataframe(feature_audit["checks"], width="stretch", hide_index=True)
        st.write("Walk-forward prediction checks")
        st.dataframe(prediction_audit["checks"], width="stretch", hide_index=True)

    st.subheader("Common-window OOS diagnostics")
    display = comparison["comparison"][
        [
            "model",
            "prediction_count",
            "directional_accuracy",
            "historical_rate_accuracy",
            "brier_score",
            "roc_auc",
            "spearman_ic",
        ]
    ].copy()
    display.columns = [
        "model",
        "OOS rows",
        "directional accuracy",
        "historical-rate accuracy",
        "Brier score",
        "ROC AUC",
        "Spearman IC",
    ]
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "These full common-window diagnostics compare model behavior; they are not used to select the "
        "model or probability threshold in the holdout audit below."
    )

    st.subheader("Validation-selected strategy policy")
    policy = select_and_evaluate_holdout_strategy(
        comparison["common_predictions"],
        prices_by_ticker={ticker: bundle.prices[ticker] for ticker in tickers if ticker in bundle.prices},
        benchmark_prices=benchmark,
        max_validation_selection_rate=max_selection_rate,
        max_validation_average_daily_turnover=max_daily_turnover,
        one_way_cost_bps=round_trip_cost_bps / 2.0,
        universe=tickers,
    )
    if policy["status"] != "ok":
        st.info(
            "No defensible held-out policy result yet: "
            f"{policy['status'].replace('_', ' ')}. This is expected with short capture history."
        )
        return

    values = policy["test_event"]
    row = st.columns(5)
    row[0].metric("Validation-selected model", policy["chosen_model"])
    row[1].metric("Locked threshold", f"{policy['probability_threshold']:.2f}")
    row[2].metric("Held-out selected events", values["selected_count"])
    row[3].metric("Held-out mean net event return", _fmt_pct(values["mean_net_abnormal_return"]))
    row[4].metric("Rule parity", policy["calculation_parity_audit"]["status"])
    windows = policy["chronological_windows"]
    validation_window = windows["validation"]
    test_window = windows["test"]
    st.caption(
        f"Selection used {policy['validation_session_count']} validation sessions; evaluation used "
        f"{policy['test_session_count']} later sessions. {policy['purged_validation_rows']} boundary rows "
        f"were purged because their labels were unavailable at the first test close. Candidate policies "
        f"had to stay below {_fmt_rate(max_selection_rate)} selection and "
        f"{_fmt_rate(max_daily_turnover)} average daily turnover. The validation window was "
        f"{pd.Timestamp(validation_window['start_session']).date()}–"
        f"{pd.Timestamp(validation_window['end_session']).date()}; the untouched test window began "
        f"{pd.Timestamp(test_window['start_session']).date()}."
    )
    st.warning(
        "Model and threshold maximize validation-period net return relative to exposure-matched SPY, subject "
        "to the frequency and turnover limits. Re-running after viewing the held-out result makes that test "
        "exploratory; a production claim needs a preregistered policy and a new untouched time period."
    )
    with st.expander("Locked StrategySpec and calculation parity"):
        st.code(policy["strategy_spec_id"])
        st.json(policy["strategy_spec"])
        st.caption(
            "Validation and held-out books share this exact spec ID, backtest version, and calculation "
            "contract. This is research/holdout calculation parity, not proof of live execution parity."
        )
    with st.expander("Validation candidate audit"):
        st.dataframe(policy["candidates"], width="stretch", hide_index=True)

    st.subheader("Locked holdout position book")
    backtest = policy["test_backtest"]
    strategy = backtest["metrics"]
    row = st.columns(5)
    row[0].metric("Net cumulative return", _fmt_pct(strategy["net_cumulative_return"]))
    row[1].metric(
        "Exposure-matched SPY",
        _fmt_pct(strategy["exposure_matched_market_return"]),
    )
    row[2].metric(
        "Net minus matched SPY",
        _fmt_pct(strategy["net_minus_exposure_matched_market"]),
    )
    row[3].metric("Total turnover", f"{strategy['total_turnover']:.2f}×")
    row[4].metric("Max drawdown", _fmt_pct(strategy["max_drawdown"]))
    st.caption(
        f"The test book starts flat on {pd.Timestamp(policy['first_test_session']).date()}, equal-weights "
        f"{strategy['unique_ticker_count']} selected ticker(s), collapses overlapping same-ticker signals, "
        f"and charges {round_trip_cost_bps / 2.0:.1f} bps per one-way weight change. "
        f"Average gross exposure was {_fmt_rate(strategy['average_gross_exposure'])}."
    )
    st.warning(
        "This adjusted-close audit assumes the signal can be acted on at its eligible close and holds cash "
        "at zero return. It is a reproducible research backtest, not executable live performance, and the "
        "short held-out window cannot establish durable alpha."
    )
    st.dataframe(
        backtest["daily"][
            [
                "session",
                "period_end",
                "active_tickers",
                "gross_exposure",
                "turnover",
                "transaction_cost",
                "gross_return",
                "net_return",
                "exposure_matched_market_return",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_validation_lab() -> None:
    st.header("Validation Lab")
    st.caption(
        "Live output from the validation core (event study -> rule-based classification -> "
        "forward-return tracking -> cohort-relative evaluation), run on demo data."
    )

    st.warning(
        "Demo data only. The synthetic universe cohort validates the *shape* of the workflow, "
        "not predictive power. ThesisBoard is not investment advice."
    )
    st.info(
        "How to read this: a hit rate is meaningful only against the cohort base rate for the same "
        "universe and horizon — focus on excess hit rate, not the raw hit rate. Daily data can show "
        "abnormal returns followed an event but cannot establish intraday causality."
    )

    lab = prepare_validation_lab_data()

    snapshots = _frame(lab["snapshots"], SNAPSHOT_COLUMNS)
    outcomes = _frame(lab["outcomes"], OUTCOME_COLUMNS)
    metrics = lab["metrics"]
    groups = _frame(metrics["groups"], GROUP_COLUMNS)

    st.subheader("Signal snapshots (recorded at creation)")
    st.caption("What the rule-based validator classified and stored before any forward return was known.")
    st.dataframe(snapshots, width="stretch", hide_index=True)

    st.subheader("Matured signal outcomes")
    st.caption(
        "Forward and abnormal returns measured at the horizon, with explicit outcome semantics and "
        "abnormal-return data-quality flags (watch for beta_fallback_used / missing_sector_proxy)."
    )
    st.dataframe(outcomes, width="stretch", hide_index=True)

    if outcomes["beta_fallback_used"].any():
        flagged = ", ".join(sorted(outcomes.loc[outcomes["beta_fallback_used"], "ticker"].unique()))
        st.warning(
            f"beta_fallback_used: insufficient return history to estimate beta for {flagged}; "
            "beta defaulted to 1.0. Treat the abnormal return as lower-confidence."
        )

    st.subheader("Metrics by horizon and classification")
    st.caption("Hit-rate semantics are classification-specific; the deprecated generic hit field is not shown.")
    st.dataframe(groups, width="stretch", hide_index=True)

    _render_summary_metrics(metrics)
    _render_excess_hit_rate_chart(groups)


def _render_summary_metrics(metrics: dict) -> None:
    st.subheader("Validation summary")

    tradeable = _first_group(metrics["groups"], "Tradeable")
    row_one = st.columns(3)
    row_one[0].metric("Matured signals", metrics["sample_size"])
    row_one[1].metric("Tradeable hit rate", _fmt_rate(_group_value(tradeable, "trade_hit_rate")))
    row_one[2].metric("Cohort base rate", _fmt_rate(_group_value(tradeable, "base_rate")))

    row_two = st.columns(3)
    row_two[0].metric("Excess hit rate vs base", _fmt_rate(_group_value(tradeable, "excess_trade_hit_rate")))
    row_two[1].metric(
        "Watch follow-through",
        _fmt_rate(_group_value(_first_group(metrics["groups"], "Watch"), "watch_followthrough_rate")),
    )
    row_two[2].metric(
        "Avoided bad trade",
        _fmt_rate(_group_value(_first_group(metrics["groups"], "Avoid Chase"), "avoided_bad_trade_rate")),
    )

    row_three = st.columns(4)
    row_three[0].metric("False positives", metrics["false_positives"])
    row_three[1].metric("False negatives", metrics["false_negatives"])
    row_three[2].metric("Avg forward abnormal return", _fmt_pct(metrics["average_forward_abnormal_return"]))
    row_three[3].metric("Median forward abnormal return", _fmt_pct(metrics["median_forward_abnormal_return"]))


def _render_excess_hit_rate_chart(groups: pd.DataFrame) -> None:
    chart_df = groups.dropna(subset=["excess_trade_hit_rate"]).copy()
    if chart_df.empty:
        return
    chart_df["excess_hit_rate_pct"] = chart_df["excess_trade_hit_rate"] * 100
    fig = px.bar(
        chart_df,
        x="classification",
        y="excess_hit_rate_pct",
        color="classification",
        labels={
            "excess_hit_rate_pct": "Excess hit rate vs base rate (pp)",
            "classification": "Classification",
        },
    )
    st.plotly_chart(fig, width="stretch")


def render_methodology() -> None:
    st.header("Methodology")
    st.subheader("Why Raw Return Is Not Enough")
    st.write(
        "A ticker can rise because the whole market rose, because its sector moved, or because beta amplified "
        "a benchmark move. Raw return alone is not enough to validate a thesis."
    )
    st.subheader("Why Abnormal Return Is Needed")
    st.write(
        "Abnormal return adjusts ticker performance against a market benchmark and, when available, a sector "
        "or theme proxy. That helps separate ticker-specific validation from broad market drift."
    )
    st.subheader("Why Forward-Return Tracking Matters")
    st.write(
        "A signal is only useful if it can be recorded at creation time and later evaluated against future returns "
        "at the intended horizon."
    )
    st.subheader("Why Hit Rate Needs Base Rate")
    st.write(
        "A 56% hit rate means little if the same universe has a 54% base rate. ThesisBoard compares hit rate "
        "with base rate and focuses on excess hit rate."
    )
    st.subheader("Daily Data Causality Limit")
    st.write(
        "Daily data can show that abnormal returns followed an event date, but it cannot prove precise intraday "
        "causal ordering."
    )


def render_roadmap() -> None:
    st.header("Roadmap")
    roadmap = pd.DataFrame(
        [
            {"version": "V1", "focus": "validation spine"},
            {"version": "V1.5", "focus": "point-in-time news evidence and chronological baseline"},
            {"version": "V2", "focus": "agentic theme classification and semantic expansion"},
            {"version": "V3", "focus": "production-grade data providers and portfolio risk"},
        ]
    )
    st.table(roadmap)


def _frame(records: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=columns)
    present = [column for column in columns if column in df.columns]
    return df[present]


def _first_group(groups: list[dict], classification: str) -> dict | None:
    return next((group for group in groups if group["classification"] == classification), None)


def _group_value(group: dict | None, key: str):
    return None if group is None else group.get(key)


def _fmt_rate(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


if __name__ == "__main__":
    main()
