from types import SimpleNamespace

import pandas as pd

from src.market_news import (
    anthropic_api_key_available,
    build_market_snapshot,
    compute_ticker_metrics,
    days_until,
    headlines_from_news,
    normalize_news,
    summarize_headlines,
)


def _series(values, start="2026-01-01"):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype="float64")


# --- 1 & 2: market snapshot helper -----------------------------------------


def test_market_snapshot_handles_valid_mocked_data():
    ticker_values = [float(100 + i) for i in range(70)]  # enough for 60D return
    prices = {
        "SPY": _series([400.0, 401.0, 402.0]),
        "QQQ": _series([350.0, 351.0]),
        "SOXX": _series([220.0, 221.0, 222.0]),
        "^VIX": _series([18.0, 17.5, 19.0]),
        "NVDA": _series(ticker_values),
    }
    snapshot = build_market_snapshot(prices, ticker="NVDA")

    assert snapshot["market"]["SPY"]["available"] is True
    assert snapshot["market"]["SPY"]["latest"] == 402.0
    assert snapshot["market"]["^VIX"]["available"] is True

    ticker = snapshot["ticker"]
    assert ticker["available"] is True
    assert ticker["latest"] == ticker_values[-1]
    assert ticker["return_5d"] is not None
    assert ticker["return_20d"] is not None
    assert ticker["return_60d"] is not None
    assert ticker["distance_from_high"] is not None


def test_market_snapshot_handles_missing_and_empty_data():
    snapshot = build_market_snapshot({}, ticker="NVDA")
    for symbol in ("SPY", "QQQ", "SOXX", "^VIX"):
        assert snapshot["market"][symbol]["available"] is False
    assert snapshot["ticker"]["available"] is False

    # empty series and a too-short series degrade gracefully
    short = build_market_snapshot({"SPY": pd.Series([], dtype="float64"), "NVDA": _series([10.0, 11.0])}, ticker="NVDA")
    assert short["market"]["SPY"]["available"] is False
    metrics = short["ticker"]
    assert metrics["available"] is True
    assert metrics["return_60d"] is None  # not enough history
    assert metrics["return_5d"] is None


def test_compute_ticker_metrics_none_series():
    assert compute_ticker_metrics(None) == {"available": False}


def test_days_until_is_none_safe():
    assert days_until(None) is None
    assert days_until("2026-06-10", as_of="2026-06-04") == 6


# --- 3, 4, 5: news normalizer ----------------------------------------------


def test_news_normalizer_handles_normal_records():
    raw = [
        {
            "title": "Chipmaker beats estimates",
            "publisher": "Reuters",
            "link": "https://example.com/a",
            "providerPublishTime": 1_760_000_000,
        },
        {  # newer nested "content" shape
            "content": {
                "title": "Datacenter demand surges",
                "provider": {"displayName": "Bloomberg"},
                "canonicalUrl": {"url": "https://example.com/b"},
                "pubDate": "2026-06-01T12:00:00Z",
            }
        },
    ]
    items = normalize_news(raw)
    assert len(items) == 2
    assert items[0]["title"] == "Chipmaker beats estimates"
    assert items[0]["publisher"] == "Reuters"
    assert items[0]["link"] == "https://example.com/a"
    assert items[0]["timestamp"] is not None
    assert items[1]["title"] == "Datacenter demand surges"
    assert items[1]["publisher"] == "Bloomberg"
    assert items[1]["link"] == "https://example.com/b"
    assert items[1]["timestamp"] == "2026-06-01T12:00:00Z"


def test_news_normalizer_handles_empty():
    assert normalize_news([]) == []
    assert normalize_news(None) == []
    assert normalize_news("unexpected") == []


def test_news_normalizer_handles_missing_fields_without_crashing():
    raw = [
        {},  # everything missing
        {"title": "Only a title"},
        {"publisher": "NoTitleWire"},
        "not a dict",  # unexpected shape -> skipped
        {"title": "Bad ts", "providerPublishTime": "oops"},
    ]
    items = normalize_news(raw)
    assert len(items) == 4  # the non-dict entry is skipped
    assert items[0] == {"title": None, "publisher": None, "link": None, "timestamp": None}
    assert items[1]["title"] == "Only a title"
    assert items[1]["link"] is None
    assert items[2]["title"] is None
    assert items[2]["publisher"] == "NoTitleWire"
    assert items[3]["title"] == "Bad ts"
    assert items[3]["timestamp"] == "oops" or items[3]["timestamp"] is not None


def test_headlines_from_news_drops_titleless():
    items = [{"title": "A"}, {"title": None}, {"title": "B"}]
    assert headlines_from_news(items) == ["A", "B"]


# --- 6 & 7: AI summary path -------------------------------------------------


def test_ai_summary_skipped_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_api_key_available() is False


def test_ai_summary_detected_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert anthropic_api_key_available() is True


class _MockMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="topic summary", type="text")])


class _MockAnthropic:
    def __init__(self):
        self.messages = _MockMessages()


def test_summarize_headlines_receives_only_normalized_headlines():
    normalized = normalize_news(
        [
            {"title": "Chipmaker beats estimates", "publisher": "Reuters"},
            {"title": "Regulatory probe opened", "publisher": "WSJ"},
            {"publisher": "NoTitle"},  # title-less -> excluded from headlines
        ]
    )
    headlines = headlines_from_news(normalized)
    client = _MockAnthropic()

    result = summarize_headlines(headlines, client=client)

    assert result == "topic summary"
    assert len(client.messages.calls) == 1
    sent = client.messages.calls[0]["messages"][0]["content"]
    # only the (two) normalized headlines were sent
    assert "Chipmaker beats estimates" in sent
    assert "Regulatory probe opened" in sent
    assert "NoTitle" not in sent
    # no price/return data leaked into the prompt
    assert "return_5d" not in sent
