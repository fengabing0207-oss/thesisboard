# ThesisBoard

ThesisBoard is a user-controlled thematic trading research dashboard for testing whether the market validates a thematic relationship.

## Current Status

This repository is in public-review MVP mode. The current `main` branch demonstrates a validation-first research workflow: record a thesis before its outcome is known, apply reproducible risk checks, and compare forward outcomes with market- and sector-adjusted benchmarks.

The application has six pages:

- **Home** — product scope and research constraints.
- **Pre-Trade Check** — structured thesis capture, heuristic risk checks, and an append-only local journal.
- **Market News** — a yfinance market snapshot, raw ticker headlines, and an optional Anthropic topic summary.
- **Validation Lab** — horizon-specific signal outcomes, abnormal returns, hit rates, and cohort base-rate comparisons.
- **Methodology** — the reasoning behind the validation design.
- **Roadmap** — planned evidence and automation layers.

Market News shows the raw source headlines before any optional AI summary. The summary is limited to headline topics and does not provide an overall bullish/bearish verdict, predict price direction, or recommend a trade.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The optional headline summary requires an `ANTHROPIC_API_KEY` in the local environment. The rest of the application runs without it.

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
- The optional Anthropic output summarizes only the displayed headlines and can be unavailable when no local API key is configured.
- The app does not yet train or deploy a news-return forecasting model.
- No brokerage integration or trade execution.
- No claims of predictive power.

## Disclaimer

ThesisBoard is not financial advice. It is a research and validation workflow prototype. It does not recommend buying or selling securities and does not execute trades.
