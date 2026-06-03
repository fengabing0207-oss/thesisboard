from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.demo_validation_data import EXPLICIT_OUTCOME_FIELDS, prepare_validation_lab_data


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

    page = st.sidebar.radio("Pages", ["Home", "Validation Lab", "Methodology", "Roadmap"])

    if page == "Home":
        render_home()
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
