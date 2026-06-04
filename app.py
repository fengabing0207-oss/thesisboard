from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import market_news
from src.demo_validation_data import EXPLICIT_OUTCOME_FIELDS, prepare_validation_lab_data
from src.journal import append_record, load_records
from src.pre_trade_check import (
    EventType,
    InstrumentType,
    Level,
    PlannedAction,
    ThesisDecision,
    evaluate_pre_trade_risk,
)


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
    st.caption("User-controlled thematic trading research. Validation first, automation later.")

    page = st.sidebar.radio(
        "Pages",
        ["Home", "Pre-Trade Check", "Market News", "Validation Lab", "Methodology", "Roadmap"],
    )

    if page == "Home":
        render_home()
    elif page == "Pre-Trade Check":
        render_pre_trade_check()
    elif page == "Market News":
        render_market_news()
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
    st.info("Current branch focus: validation spine, not full AI automation yet.")
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
            {"version": "V1.5", "focus": "price/news evidence layer"},
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
