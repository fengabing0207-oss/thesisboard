from __future__ import annotations

from dataclasses import dataclass


POSITIVE_ABNORMAL_REACTION = "Positive Abnormal Reaction"
NEGATIVE_ABNORMAL_REACTION = "Negative Abnormal Reaction"
NO_MEANINGFUL_ABNORMAL_REACTION = "No Meaningful Abnormal Reaction"
SELL_THE_NEWS = "Sell the News"
BAD_NEWS_ABSORBED = "Bad News Absorbed"
INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True)
class EventReaction:
    label: str
    abnormal_return: float | None
    causal_claim: str
    reason: str


def classify_event_reaction(
    news_sentiment: str,
    abnormal_return: float | None,
    *,
    timestamp_present: bool = True,
    data_granularity: str = "daily",
    material_threshold: float = 0.01,
    has_explicit_basket_evidence: bool = False,
) -> EventReaction:
    """Classify event reaction from abnormal return, not raw return."""
    if abnormal_return is None or not timestamp_present:
        return EventReaction(
            label=INCONCLUSIVE,
            abnormal_return=abnormal_return,
            causal_claim="Do not make a strong causal claim.",
            reason="Missing abnormal return or event timestamp.",
        )

    sentiment = _normalize_sentiment(news_sentiment)
    causal_claim = (
        "Daily-only evidence; use as association, not proof of intraday causality."
        if data_granularity.lower() in {"daily", "daily-only", "end_of_day"}
        else "Timestamped event evidence; still validate with forward tracking."
    )

    if sentiment == "neutral" and not has_explicit_basket_evidence:
        return EventReaction(
            label=INCONCLUSIVE,
            abnormal_return=abnormal_return,
            causal_claim=causal_claim,
            reason="Neutral or unknown catalyst without explicit basket/theme evidence.",
        )

    if abs(abnormal_return) < material_threshold:
        return EventReaction(
            label=NO_MEANINGFUL_ABNORMAL_REACTION,
            abnormal_return=abnormal_return,
            causal_claim=causal_claim,
            reason="Abnormal return did not clear the materiality threshold.",
        )

    if sentiment == "positive" and abnormal_return > 0:
        label = POSITIVE_ABNORMAL_REACTION
        reason = "Positive catalyst aligned with positive abnormal return."
    elif sentiment == "positive" and abnormal_return < 0:
        label = SELL_THE_NEWS
        reason = "Positive catalyst was met with negative abnormal return."
    elif sentiment == "negative" and abnormal_return >= 0:
        label = BAD_NEWS_ABSORBED
        reason = "Negative catalyst was absorbed without negative abnormal return."
    elif sentiment == "negative" and abnormal_return < 0:
        label = NEGATIVE_ABNORMAL_REACTION
        reason = "Negative catalyst aligned with negative abnormal return."
    elif abnormal_return > 0:
        label = POSITIVE_ABNORMAL_REACTION
        reason = "Abnormal return was positive with explicit basket/theme evidence."
    else:
        label = NEGATIVE_ABNORMAL_REACTION
        reason = "Abnormal return was negative with explicit basket/theme evidence."

    return EventReaction(label=label, abnormal_return=abnormal_return, causal_claim=causal_claim, reason=reason)


def _normalize_sentiment(news_sentiment: str) -> str:
    value = (news_sentiment or "").strip().lower()
    if value in {"positive", "bullish", "good", "upbeat"}:
        return "positive"
    if value in {"negative", "bearish", "bad", "downbeat"}:
        return "negative"
    return "neutral"
