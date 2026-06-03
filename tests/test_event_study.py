from src.event_study import (
    BAD_NEWS_ABSORBED,
    INCONCLUSIVE,
    POSITIVE_ABNORMAL_REACTION,
    SELL_THE_NEWS,
    classify_event_reaction,
)


def test_positive_news_positive_abnormal_return_is_positive_reaction():
    reaction = classify_event_reaction("positive", 0.03)
    assert reaction.label == POSITIVE_ABNORMAL_REACTION
    assert "Daily-only" in reaction.causal_claim


def test_positive_news_negative_abnormal_return_is_sell_the_news():
    reaction = classify_event_reaction("positive", -0.02)
    assert reaction.label == SELL_THE_NEWS


def test_negative_news_non_negative_abnormal_return_is_bad_news_absorbed():
    reaction = classify_event_reaction("negative", 0.015)
    assert reaction.label == BAD_NEWS_ABSORBED


def test_missing_timestamp_is_inconclusive():
    reaction = classify_event_reaction("positive", 0.04, timestamp_present=False)
    assert reaction.label == INCONCLUSIVE
    assert "Do not make a strong causal claim" in reaction.causal_claim


def test_neutral_or_unknown_catalyst_is_inconclusive_without_basket_evidence():
    reaction = classify_event_reaction("unknown", 0.05)
    assert reaction.label == INCONCLUSIVE


def test_neutral_catalyst_can_classify_with_explicit_basket_evidence():
    reaction = classify_event_reaction("neutral", 0.05, has_explicit_basket_evidence=True)
    assert reaction.label == POSITIVE_ABNORMAL_REACTION
