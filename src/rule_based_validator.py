from __future__ import annotations

from dataclasses import dataclass

from .event_study import (
    BAD_NEWS_ABSORBED,
    INCONCLUSIVE,
    NEGATIVE_ABNORMAL_REACTION,
    NO_MEANINGFUL_ABNORMAL_REACTION,
    POSITIVE_ABNORMAL_REACTION,
    SELL_THE_NEWS,
)


RULE_VERSION = "rules.v1"


@dataclass(frozen=True)
class ValidationDecision:
    classification: str
    score: float
    rule_version: str
    reason: str


def classify_signal(
    *,
    catalyst_sentiment: str,
    event_reaction: str,
    technically_overextended: bool = False,
    setup_score: float = 50,
    volume_confirmation: bool = False,
) -> ValidationDecision:
    sentiment = _normalize_sentiment(catalyst_sentiment)
    score = float(setup_score) + (8 if volume_confirmation else 0)

    if sentiment == "positive" and event_reaction == POSITIVE_ABNORMAL_REACTION and technically_overextended:
        return ValidationDecision(
            classification="Avoid Chase",
            score=score,
            rule_version=RULE_VERSION,
            reason="Positive catalyst and reaction are real, but the setup is technically overextended.",
        )

    if event_reaction in {SELL_THE_NEWS, NEGATIVE_ABNORMAL_REACTION}:
        return ValidationDecision("Avoid", score, RULE_VERSION, "Abnormal reaction argues against the bullish thesis.")

    if sentiment == "negative" and event_reaction == BAD_NEWS_ABSORBED:
        return ValidationDecision("Watch", score, RULE_VERSION, "Bad news was absorbed, but needs forward validation.")

    if event_reaction in {INCONCLUSIVE, NO_MEANINGFUL_ABNORMAL_REACTION}:
        if sentiment == "positive":
            return ValidationDecision(
                "Wait for Confirmation",
                score,
                RULE_VERSION,
                "Positive catalyst has not produced a meaningful abnormal reaction.",
            )
        return ValidationDecision("Avoid", score, RULE_VERSION, "No testable positive validation is present.")

    if sentiment == "positive" and event_reaction == POSITIVE_ABNORMAL_REACTION and score >= 70:
        return ValidationDecision("Tradeable", score, RULE_VERSION, "Catalyst, abnormal reaction, and setup score align.")

    if event_reaction == POSITIVE_ABNORMAL_REACTION:
        return ValidationDecision("Watch", score, RULE_VERSION, "Positive abnormal reaction exists, but setup quality is not high enough.")

    return ValidationDecision("Avoid", score, RULE_VERSION, "Rules did not find a validated setup.")


def is_technically_overextended(metrics: dict) -> bool:
    if metrics.get("technically_overextended"):
        return True
    rsi = metrics.get("rsi")
    if rsi is not None and rsi >= 75:
        return True
    return_20d = metrics.get("return_20d")
    if return_20d is not None and return_20d >= 0.25:
        return True
    return False


def _normalize_sentiment(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"positive", "bullish", "good"}:
        return "positive"
    if lowered in {"negative", "bearish", "bad"}:
        return "negative"
    return "neutral"
