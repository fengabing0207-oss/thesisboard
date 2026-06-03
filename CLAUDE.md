# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `python -m pip install -r requirements.txt`
- Run the app: `streamlit run app.py`
- Run all tests: `python -m pytest`
- Run a single test: `python -m pytest tests/test_forward_tracker.py::test_forward_tracker_records_matured_outcome`
- Run the end-to-end demo (also run in CI): `python scripts/run_validation_demo.py`

CI (`.github/workflows/tests.yml`) runs `pytest` then the validation demo on every push/PR. Both must pass.

## What this project is

ThesisBoard tests whether the market *validates* a thematic trading thesis — it does not predict or recommend trades. The repo is in public-review MVP mode focused on the "validation spine," not AI/agent automation. Hard product boundaries (see `AGENTS.md`): no brokerage integration, no execution, no buy/sell advice, no required paid API keys in V1, and no predictive-power claims. `app.py` runs on **demo data only**; the validation engine in `src/` is the real, tested logic.

## Architecture: the validation pipeline

The core flow lives in `src/` and chains in this order. Each stage feeds the next:

1. **`event_study.classify_event_reaction`** — Maps news sentiment + an *abnormal* return (never raw return) into a labeled reaction (`Positive Abnormal Reaction`, `Sell the News`, `Bad News Absorbed`, `Inconclusive`, etc.). Attaches an explicit `causal_claim` because daily data cannot prove intraday causality.
2. **`rule_based_validator.classify_signal`** — Combines catalyst sentiment, the event-reaction label, setup score, and overextension into one of: `Tradeable`, `Watch`, `Wait for Confirmation`, `Avoid Chase`, `Avoid`. Stamps `rule_version` (`RULE_VERSION = "rules.v1"`).
3. **`forward_tracker.record_signal`** — Persists the classified signal as a snapshot at creation time (price, horizon, benchmark, sector proxy, rule version) via `signal_store`.
4. **`forward_tracker.evaluate_forward_returns`** — After the horizon matures, computes forward return + abnormal return and writes the outcome. Maturity uses **benchmark trading sessions**, not calendar days (`trading_session_horizon_end`): a horizon means N market sessions after creation, measured on the benchmark calendar so sparse ticker data does not shift horizons.
5. **`signal_evaluator.evaluate_signal_records`** — Aggregates matured outcomes into hit rates grouped by `(horizon_days, classification)`, and crucially computes **excess hit rate vs. a cohort base rate** (`compute_cohort_base_rate`) for the same universe/horizon. A hit rate is meaningless without its base rate.

`scripts/run_validation_demo.py` wires all five stages together with synthetic price series and is the best executable reference for how they compose.

### Abnormal return is the central concept

`abnormal_returns.py` exists because raw return contaminates thesis validation (market drift, sector moves, beta amplification). Key points:

- `abnormal_return_summary` is the main entry. It estimates a **rolling beta** from returns *strictly before* `beta_estimation_end` (no look-ahead), falls back to `beta = 1.0` when history is insufficient (flagged via `beta_fallback_used`), and returns a `data_quality_flag` (`ok`, `beta_fallback_used`, `missing_sector_proxy`, `insufficient_price_history`, ...).
- `combined_abnormal_return` blends a beta-adjusted market component with a sector/theme proxy component (default `sector_weight=0.5`).
- Themes map to ETF proxies via `THEME_PROXIES` / `proxy_for_theme` (substring match, e.g. "AI Infrastructure" → `SMH`).

### Outcome semantics are classification-specific

There is **no single "hit" field** — the `hit` column is deprecated (`DEPRECATED_HIT_FIELD_NOTE`). `classify_outcome_semantics` produces four orthogonal booleans depending on the signal's classification:

- `Tradeable` → `trade_hit` (abnormal return > 0)
- `Watch` / `Wait for Confirmation` → `watch_followthrough`
- `Avoid` / `Avoid Chase` → `avoided_bad_trade`, and `false_negative` if abnormal return exceeds `FALSE_NEGATIVE_ABNORMAL_THRESHOLD` (0.02)

When changing scoring or outcome logic, update all four consistently and the tests that cover them.

### Persistence (`signal_store.py`)

SQLite at `data/thesisboard.db` (gitignored). Two tables: `signal_snapshots` (created at signal time) and `signal_outcomes` (1:1, written at evaluation, upserted on `signal_id`). `init_signal_store` is idempotent and self-migrating via `_ensure_column` — add new outcome columns there, not with a manual migration. Every public function takes a `db_path` param so tests pass `tmp_path`.

## Conventions and guardrails

- `src/` modules use relative imports (`from .abnormal_returns import ...`); scripts/tests insert the repo root on `sys.path` and import as `from src...`.
- Horizons are restricted to `STANDARD_HORIZONS` (1, 3, 5, 20, 60); `validate_horizon` raises on anything else.
- Never commit databases, secrets, `.env`, `.streamlit/secrets.toml`, caches, or account data (already in `.gitignore`).
- **Git workflow (from `AGENTS.md`)**: never push to `main`; every task uses a feature branch, gets committed/pushed, and opens a PR (draft for WIP). Do not auto-merge — user review required. Before pushing run `git status`, `python -m pytest`, and the demo script. Use `--force-with-lease`, never plain `--force`.
- Add/update tests when changing database, scoring, abnormal-return, or aggregation behavior.

## Docs

`docs/validation_methodology.md` (the "why"), `docs/roadmap.md` (V1 spine → V1.5 evidence → V2 agentic → V3 production), and `docs/decision_log.md` (why automation was paused for the validation spine).
