from src.event_study import POSITIVE_ABNORMAL_REACTION, SELL_THE_NEWS
from src.rule_based_validator import classify_signal, is_technically_overextended


def test_positive_validated_setup_can_be_tradeable():
    decision = classify_signal(
        catalyst_sentiment="positive",
        event_reaction=POSITIVE_ABNORMAL_REACTION,
        technically_overextended=False,
        setup_score=75,
    )
    assert decision.classification == "Tradeable"


def test_avoid_chase_overrides_tradeable():
    decision = classify_signal(
        catalyst_sentiment="positive",
        event_reaction=POSITIVE_ABNORMAL_REACTION,
        technically_overextended=True,
        setup_score=90,
    )
    assert decision.classification == "Avoid Chase"


def test_sell_the_news_is_avoid():
    decision = classify_signal(catalyst_sentiment="positive", event_reaction=SELL_THE_NEWS, setup_score=85)
    assert decision.classification == "Avoid"


def test_overextended_detection_from_metrics():
    assert is_technically_overextended({"rsi": 78})
    assert is_technically_overextended({"return_20d": 0.28})
    assert not is_technically_overextended({"rsi": 62, "return_20d": 0.1})
