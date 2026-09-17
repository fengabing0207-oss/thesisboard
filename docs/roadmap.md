# Roadmap

## V1: Validation Spine

- Horizon definitions.
- Abnormal return logic.
- Forward-return tracking.
- Hit-rate and base-rate comparison.
- Public deployable review app.

## V1.5: Price/News Evidence Layer

- Prototype adjusted-price and raw-headline evidence.
- Immutable first-seen news capture and close-to-close session alignment.
- TF-IDF/VADER chronological baseline with label-availability guards.
- Common-window logistic/random-forest challenger comparison.
- Purged validation-only model/threshold selection with frequency and turnover constraints.
- Locked equal-weight position book with overlap-aware turnover and one-way costs.
- Repeatable watchlist collection with immutable collection-run audits and ticker-level coverage diagnostics.
- Hashed as-of feature vintages plus fail-closed feature/label leakage audits.
- Versioned rule-based `StrategySpec` shared by validation and held-out calculation paths.
- Explicit expanding-training, purged-validation, and later-test window metadata.
- Event reaction tables.
- Guardrails around daily-data causality.
- Future: verified historical-news adapter, delayed execution variants, calibration/stability/subgroup diagnostics, and preregistered evaluation on newly accumulated data.

## V2: Agentic Theme Classification And Semantic Expansion

- AI-assisted theme classification.
- Related-asset semantic expansion.
- User overrides and validation tracking.

## V3: Production-Grade Data Providers And Portfolio Risk

- Production market data providers.
- More robust macro/sector proxies.
- Portfolio overlap and concentration risk workflows.
- Historical validation cohorts.
