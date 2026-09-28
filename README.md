# ThesisBoard

ThesisBoard is a user-controlled thematic trading research dashboard for testing whether the market validates a thematic relationship.

## Current Status

This repository is in public-review MVP mode. The current `main` branch demonstrates a validation-first research workflow: record a thesis before its outcome is known, apply reproducible risk checks, and compare forward outcomes with market- and sector-adjusted benchmarks.

The application has seven pages:

- **Home** — product scope and research constraints.
- **Pre-Trade Check** — structured thesis capture, heuristic risk checks, and an append-only local journal.
- **Market News** — a yfinance market snapshot, raw ticker headlines, and an optional Anthropic topic summary.
- **News Signal Lab** — immutable first-seen headline capture, operational and model-window readiness checks, hashed point-in-time feature vintages, leakage-audited walk-forward evaluation, a locked rule-based `StrategySpec`, a turnover-aware held-out position book, and an append-only scheduled-research audit.
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

Build a real point-in-time headline history by running, for example, `python scripts/collect_news.py NVDA AVGO MRVL MU VRT ORCL INTC`. Each batch writes an immutable collection-run audit record.

For a guarded collection-and-calibration cycle, run `python scripts/run_research_cycle.py`. The cycle re-evaluates matured history and may append a challenger `StrategySpec`, but it never promotes that candidate. Local SQLite is suitable for development only. Scheduled runners require `THESISBOARD_DATABASE_URL` to point to PostgreSQL and refuse to run otherwise.

Each completed priced cycle records a `dataset_readiness` snapshot: signal, matured, and usable row/session counts; label balance; data-quality counts; and progress through the default 10-session training, 10-session validation, and 5-session test floor. This is a progress guardrail, not a countdown promise. Class diversity, minimum rows/signals, leakage audits, or policy constraints can require more history.

The repository includes a two-hourly GitHub Actions workflow, disabled by default. After the workflow reaches the default branch, configure the `THESISBOARD_DATABASE_URL` repository secret, set the `THESISBOARD_AUTOMATION_ENABLED` repository variable to `true`, and optionally set `THESISBOARD_WATCHLIST`. A reviewed candidate is approved separately, for example:

```bash
python scripts/promote_strategy.py 42 approved \
  --decided-by "research-owner" \
  --note "Reviewed validation and held-out diagnostics"
```

See [Automation Operations](docs/automation_operations.md) for enablement, failure behavior, and audit semantics.

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
- Once explicitly enabled with durable PostgreSQL storage, a two-hour headline-capture workflow accumulates new `first_seen_at` observations and a separate weekday-evening research cycle evaluates only settled labels. It cannot recreate observations from before enablement and it stops candidate generation whenever collection or dataset-integrity checks fail.
- Scheduled calibration selects only on the validation window. Held-out metrics are logged as diagnostics, candidates are never auto-promoted, and every approval or rejection is a separate append-only event with a database-level one-decision constraint.
- News Signal Lab selects a model/threshold only on a purged validation segment, maximizing net return relative to exposure-matched SPY subject to signal-frequency and average-daily-turnover limits, then reports a later held-out audit. Re-running after viewing the test makes it exploratory.
- Every modeled feature row records an `as_of_timestamp` and deterministic `data_vintage`; policy selection fails closed if feature times, training-label cutoffs, target-label times, or common OOS keys violate the point-in-time contract.
- The selected model, threshold, universe, holding horizon, benchmark, entry/exit rules, weighting, overlap handling, rebalancing schedule, exposure cap, cash assumption, and cost are hashed into one immutable `StrategySpec`. Validation and held-out books must report the same spec and calculation-engine versions.
- The position-book audit starts flat on the held-out segment, equal-weights active tickers, collapses overlapping same-ticker signals, and charges one-way cost on realized weight turnover.
- Calculation parity means the two research windows use identical code and rules. It does not establish parity with broker fills or live production. Adjusted-close execution, zero-return cash, small held-out samples, and best-effort data remain research assumptions—not executable performance or an alpha claim.
- The first News Signal Lab UI uses a SPY beta-adjusted target; ticker-specific sector-proxy mappings remain an engine-level option.
- The optional Anthropic output summarizes only the displayed headlines and can be unavailable when no local API key is configured.
- The app does not deploy a forecasting model or generate live trade instructions.
- No brokerage integration or trade execution.
- No claims of predictive power.

## Disclaimer

ThesisBoard is not financial advice. It is a research and validation workflow prototype. It does not recommend buying or selling securities and does not execute trades.
