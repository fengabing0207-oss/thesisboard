"""Independent market-context reference sources (link-out only).

A small, configurable list of external sites grouped by category, used to check
macro regime, sector/theme, ticker research, event risk, and options/volatility
BEFORE a trade. This is a reference layer only: NO scraping, NO requests, NO data
fetching — just links the user opens manually. Ticker-specific sources use a
``url_template`` with ``{ticker}``; the rest are static.

These are context references, not buy/sell signals (see ``PANEL_CAUTIONS``).
"""

from __future__ import annotations

from dataclasses import dataclass

MACRO_REGIME = "macro_regime"
SECTOR_THEME = "sector_theme"
TICKER_RESEARCH = "ticker_research"
EVENT_RISK = "event_risk"
OPTIONS_VOLATILITY = "options_volatility"

CATEGORY_ORDER = [MACRO_REGIME, SECTOR_THEME, TICKER_RESEARCH, EVENT_RISK, OPTIONS_VOLATILITY]
CATEGORY_LABELS = {
    MACRO_REGIME: "Macro regime",
    SECTOR_THEME: "Sector / theme",
    TICKER_RESEARCH: "Ticker research",
    EVENT_RISK: "Event risk",
    OPTIONS_VOLATILITY: "Options / volatility",
}

# Manual snapshot note fields (all optional). source_checked_at is auto-filled on save.
SNAPSHOT_NOTE_FIELDS = [
    "market_regime_note",
    "sector_theme_note",
    "ticker_context_note",
    "event_risk_note",
    "options_flow_note",
]

PANEL_CAUTIONS = [
    "External market sources provide context only. They are not buy/sell signals.",
    "Options flow / unusual volume is not smart-money confirmation without thesis validation.",
    "Setup rating is a discipline/risk-control rating, not a return forecast.",
]


@dataclass(frozen=True)
class MarketSource:
    id: str
    name: str
    category: str
    description: str
    best_for: str
    caution: str
    priority: int
    url: str | None = None
    url_template: str | None = None

    def is_ticker_specific(self) -> bool:
        return self.url_template is not None

    def resolve_url(self, ticker: str | None = None) -> str:
        """Return the openable URL, filling {ticker} (uppercased) for templates."""
        if self.url_template is not None:
            return self.url_template.format(ticker=(ticker or "").strip().upper())
        return self.url or ""


MARKET_SOURCES = [
    # --- macro regime ---
    MarketSource(
        id="finviz_overview",
        name="Finviz Market Overview",
        category=MACRO_REGIME,
        description="Broad market dashboard: indices, breadth, movers.",
        best_for="Quick read of the overall tape.",
        caution="Snapshot only; not a regime model.",
        priority=1,
        url="https://finviz.com/",
    ),
    MarketSource(
        id="cboe_vix",
        name="Cboe VIX",
        category=MACRO_REGIME,
        description="Implied-volatility index for the S&P 500.",
        best_for="Gauging market fear / risk appetite.",
        caution="VIX level is not a timing signal.",
        priority=2,
        url="https://www.cboe.com/tradable_products/vix/",
    ),
    MarketSource(
        id="cme_fedwatch",
        name="CME FedWatch",
        category=MACRO_REGIME,
        description="Market-implied probabilities for Fed rate moves.",
        best_for="Rate-path expectations around FOMC.",
        caution="Probabilities shift fast; not a forecast.",
        priority=3,
        url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
    ),
    MarketSource(
        id="aaii_sentiment",
        name="AAII Sentiment Survey",
        category=MACRO_REGIME,
        description="Weekly retail bull/bear sentiment survey.",
        best_for="Contrarian sentiment context.",
        caution="Noisy; a sentiment reading, not a trigger.",
        priority=4,
        url="https://www.aaii.com/sentimentsurvey",
    ),
    # --- sector / theme ---
    MarketSource(
        id="tradingview_heatmap",
        name="TradingView Stock Heatmap",
        category=SECTOR_THEME,
        description="Visual heatmap of sectors and constituents.",
        best_for="Seeing where money is rotating today.",
        caution="One-day color is not a trend.",
        priority=1,
        url="https://www.tradingview.com/heatmap/stock/",
    ),
    MarketSource(
        id="yahoo_sectors",
        name="Yahoo Finance Sectors",
        category=SECTOR_THEME,
        description="Sector-level performance breakdown.",
        best_for="Relative sector strength.",
        caution="Lagging, end-of-day oriented.",
        priority=2,
        url="https://finance.yahoo.com/sectors/",
    ),
    MarketSource(
        id="finviz_map",
        name="Finviz Map",
        category=SECTOR_THEME,
        description="Treemap of the market by sector and size.",
        best_for="Fast visual sector/theme scan.",
        caution="Snapshot; no historical context.",
        priority=3,
        url="https://finviz.com/map.ashx",
    ),
    # --- ticker research ---
    MarketSource(
        id="finviz_quote",
        name="Finviz Ticker Page",
        category=TICKER_RESEARCH,
        description="Per-ticker fundamentals, technicals, and news.",
        best_for="One-screen ticker overview.",
        caution="Ratios can be stale or vendor-adjusted.",
        priority=1,
        url_template="https://finviz.com/quote.ashx?t={ticker}",
    ),
    MarketSource(
        id="yahoo_quote",
        name="Yahoo Finance Quote",
        category=TICKER_RESEARCH,
        description="Per-ticker quote, chart, statistics, news.",
        best_for="General ticker research.",
        caution="Data quality varies; verify key numbers.",
        priority=2,
        url_template="https://finance.yahoo.com/quote/{ticker}",
    ),
    MarketSource(
        id="stockanalysis",
        name="StockAnalysis",
        category=TICKER_RESEARCH,
        description="Clean financial statements and metrics.",
        best_for="Fast fundamentals check.",
        caution="Not a substitute for primary filings.",
        priority=3,
        url_template="https://stockanalysis.com/stocks/{ticker}",
    ),
    MarketSource(
        id="sec_edgar",
        name="SEC EDGAR Filings",
        category=TICKER_RESEARCH,
        description="Primary-source company filings by ticker.",
        best_for="Reading the actual 10-K/10-Q/8-K.",
        caution="Primary source; you do the reading.",
        priority=4,
        url_template="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={ticker}&type=&dateb=&owner=include&count=40",
    ),
    # --- event risk ---
    MarketSource(
        id="yahoo_earnings",
        name="Yahoo Earnings Calendar",
        category=EVENT_RISK,
        description="Upcoming earnings dates for the ticker.",
        best_for="Confirming earnings timing.",
        caution="Dates are estimates until confirmed.",
        priority=1,
        url_template="https://finance.yahoo.com/calendar/earnings?symbol={ticker}",
    ),
    MarketSource(
        id="nasdaq_earnings",
        name="Nasdaq Earnings Calendar",
        category=EVENT_RISK,
        description="Earnings schedule and history for the ticker.",
        best_for="Cross-checking the earnings date.",
        caution="Estimated dates can move.",
        priority=2,
        url_template="https://www.nasdaq.com/market-activity/stocks/{ticker}/earnings",
    ),
    MarketSource(
        id="marketchameleon_earnings",
        name="Market Chameleon Earnings",
        category=EVENT_RISK,
        description="Earnings detail incl. historical move stats.",
        best_for="Gauging typical post-earnings moves.",
        caution="Past move size is not predictive.",
        priority=3,
        url_template="https://marketchameleon.com/Overview/{ticker}/Earnings/",
    ),
    # --- options / volatility ---
    MarketSource(
        id="barchart_unusual_options",
        name="Barchart Unusual Options Activity",
        category=OPTIONS_VOLATILITY,
        description="Unusual options volume for the ticker.",
        best_for="Spotting notable options activity.",
        caution="Unusual volume is not smart-money proof.",
        priority=1,
        url_template="https://www.barchart.com/stocks/quotes/{ticker}/options-activity",
    ),
    MarketSource(
        id="marketchameleon_ivrank",
        name="Market Chameleon IV Rank",
        category=OPTIONS_VOLATILITY,
        description="IV rank/percentile context for the ticker.",
        best_for="Is option premium rich or cheap right now.",
        caution="IV rank is context, not a directional call.",
        priority=2,
        url_template="https://marketchameleon.com/Overview/{ticker}/IVRank/",
    ),
]


def sources_for_category(category: str) -> list:
    return sorted(
        (source for source in MARKET_SOURCES if source.category == category),
        key=lambda source: source.priority,
    )
