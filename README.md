# ThesisBoard

ThesisBoard is a user-controlled thematic trading research dashboard for testing whether the market validates a thematic relationship.

## Current Status

This repository is in public-review MVP mode. The current branch focuses on making ThesisBoard deployable and understandable while demonstrating the validation spine: horizon-specific signal tracking, abnormal returns, forward-return outcomes, hit rate, and base-rate comparison.

The app does not use OpenAI APIs yet, does not connect to brokerages, and does not provide trading recommendations.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

- Repository: `fengabing0207-oss/thesisboard`
- Branch: `feature/deployable-review-version`
- Main file path: `app.py`

## Methodology Focus

The current methodological focus is validation before automation:

- Raw return is not enough because market and sector movement can contaminate apparent thesis confirmation.
- Abnormal return helps compare ticker movement against market and theme proxies.
- Forward-return tracking records a signal at creation time and evaluates it after the intended horizon.
- Hit rate must be compared with base rate for the same universe and horizon.
- Daily data cannot prove precise intraday causality.

## Limitations

- Demo data only in the deployable review app.
- No real-time market data in this branch.
- No news ingestion or AI summarization in this branch.
- No brokerage integration or trade execution.
- No claims of predictive power.

## Disclaimer

ThesisBoard is not financial advice. It is a research and validation workflow prototype. It does not recommend buying or selling securities and does not execute trades.
