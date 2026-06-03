from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


DEMO_SIGNALS = [
    {
        "ticker": "NVDA",
        "theme": "AI Infrastructure",
        "classification": "Tradeable",
        "horizon_days": 5,
        "forward_abnormal_return": 0.034,
        "base_rate": 0.51,
    },
    {
        "ticker": "UAL",
        "theme": "June Airline Bull Case",
        "classification": "Watch",
        "horizon_days": 5,
        "forward_abnormal_return": 0.012,
        "base_rate": 0.48,
    },
    {
        "ticker": "SMH",
        "theme": "AI Infrastructure",
        "classification": "Avoid Chase",
        "horizon_days": 5,
        "forward_abnormal_return": -0.018,
        "base_rate": 0.51,
    },
    {
        "ticker": "JETS",
        "theme": "June Airline Bull Case",
        "classification": "Wait for Confirmation",
        "horizon_days": 20,
        "forward_abnormal_return": -0.006,
        "base_rate": 0.46,
    },
]


def positive_abnormal_return_hit(forward_abnormal_return: float, bullish_signal: bool = True) -> bool:
    if bullish_signal:
        return forward_abnormal_return > 0
    return forward_abnormal_return < 0


def calculate_excess_hit_rate(hit_rate: float, base_rate: float) -> float:
    return hit_rate - base_rate


def build_demo_validation_table() -> pd.DataFrame:
    df = pd.DataFrame(DEMO_SIGNALS)
    df["hit"] = df.apply(
        lambda row: positive_abnormal_return_hit(row["forward_abnormal_return"])
        if row["classification"] != "Avoid Chase"
        else None,
        axis=1,
    )
    df["hit_rate"] = df["hit"].map({True: 1.0, False: 0.0})
    df["excess_hit_rate"] = df.apply(
        lambda row: calculate_excess_hit_rate(row["hit_rate"], row["base_rate"])
        if pd.notna(row["hit_rate"])
        else None,
        axis=1,
    )
    return df[
        [
            "ticker",
            "theme",
            "classification",
            "horizon_days",
            "forward_abnormal_return",
            "hit",
            "base_rate",
            "excess_hit_rate",
        ]
    ]


def split_tradeable_and_avoid_chase(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tradeable = df[df["classification"].isin(["Tradeable", "Watch", "Wait for Confirmation"])]
    avoid_chase = df[df["classification"] == "Avoid Chase"]
    return tradeable, avoid_chase


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
    st.caption("Demo data only. This page exists to validate the workflow before adding real signal history.")

    df = build_demo_validation_table()
    st.dataframe(df, width="stretch", hide_index=True)

    chart_df = df.dropna(subset=["excess_hit_rate"]).copy()
    chart_df["excess_hit_rate_pct"] = chart_df["excess_hit_rate"] * 100
    fig = px.bar(
        chart_df,
        x="ticker",
        y="excess_hit_rate_pct",
        color="classification",
        labels={"excess_hit_rate_pct": "Excess hit rate vs base rate (pp)", "ticker": "Ticker"},
    )
    st.plotly_chart(fig, width="stretch")

    tradeable, avoid_chase = split_tradeable_and_avoid_chase(df)
    left, right = st.columns(2)
    left.metric("Bullish/setup signals", len(tradeable))
    right.metric("Avoid Chase tracked separately", len(avoid_chase))


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


if __name__ == "__main__":
    main()
