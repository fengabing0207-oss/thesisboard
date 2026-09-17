# ThesisBoard

ThesisBoard is a user-controlled thematic trading research dashboard for testing whether the market validates a thematic relationship.

## Current Status

This repository is in public-review MVP mode. The current `main` branch demonstrates a validation-first research workflow: record a thesis before its outcome is known, apply reproducible risk checks, and compare forward outcomes with market- and sector-adjusted benchmarks.

The application has seven pages:

- **Home** — product scope and research constraints.
- **Pre-Trade Check** — structured thesis capture, heuristic risk checks, and an append-only local journal.
- **Market News** — a yfinance market snapshot, raw ticker headlines, and an optional Anthropic topic summary.
- **News Signal Lab** — immutable first-seen headline capture, close-to-close targets, common-window linear/tree comparison, and a validation-selected event-policy audit.
- **Validation Lab** — horizon-specific signal outcomes, abnormal returns, hit rates, and cohort base-rate comparisons.
- **Methodology** — the reasoning behind the validation design.
- **Roadmap** — planned evidence and automation layers.

Market News shows the raw source headlines before any optional AI summary. The summary is limited to headline topics and does not provide an overall bullish/bearish verdict, predict price direction, or recommend a trade.

See [News Signal Methodology](docs/news_signal_methodology.md) for the availability clock, immutable capture contract, target construction, and walk-forward evaluation rules.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The optional headline summary requires an `ANTHROPIC_API_KEY` in the local environment. The rest of the application runs without it.

For an offline check of the News Signal Lab pipeline, run `python scripts/run_news_signal_demo.py`. Its data and relationship are synthetic and demonstrate plumbing only, not predictive performance.

## Deploy To Streamlit Community Cloud

- Repository: `fengabing0207-oss/thesisboard`
- Branch: `main`
- Main file path: `app.py`

## Methodology Focus

The current methodological focus is validation before automation:

- Raw return is not enough because market and sector movement can contaminate apparent thesis confirmation.
- Abnormal return helps compare ticker movement against market and theme proxies.
- Forward-return tracking records a signal at creation time and evaluates it after the intended horizon.
- Hit rate must be compared with base rate for the same universe and horizon.
- Adjusted-price history is accessed through a provider boundary with reproducible local caching.
- Daily data cannot prove precise intraday causality.

## Limitations

- Validation Lab currently uses synthetic demo signals to demonstrate the workflow; it does not establish predictive power.
- Market News depends on best-effort yfinance data and is not a complete point-in-time historical news archive.
- News Signal Lab starts collecting availability history only when the user captures headlines; vendor timestamps are never treated as proof that an article was historically available to ThesisBoard.
- News Signal Lab selects a model/threshold only on a purged validation segment, then reports a later held-out event audit. Re-running after viewing the test makes it exploratory.
- Cost-adjusted event returns are not portfolio P&L or turnover; the app has no stateful position-book backtest and makes no alpha claim.
- The first News Signal Lab UI uses a SPY beta-adjusted target; ticker-specific sector-proxy mappings remain an engine-level option.
- The optional Anthropic output summarizes only the displayed headlines and can be unavailable when no local API key is configured.
- The app does not deploy a forecasting model or generate live trade instructions.
- No brokerage integration or trade execution.
- No claims of predictive power.

## Disclaimer

ThesisBoard is not financial advice. It is a research and validation workflow prototype. It does not recommend buying or selling securities and does not execute trades.
