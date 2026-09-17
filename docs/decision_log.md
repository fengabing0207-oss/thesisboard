# Decision Log

## 2026-06-02

- Paused broader AI/agent expansion to build a validation spine first.
- Reason: without forward-return tracking, thesis validation risks becoming persuasive but untested narrative generation.
- Chose a minimal deployable Streamlit app for public review before adding real data providers or AI features.
- Kept demo data clearly labeled and avoided predictive-power claims.

## 2026-09-17

- Kept the existing expanding-window and purged holdout design instead of adding a second, conflicting chronological splitter.
- Added explicit as-of/vintage and leakage audits that fail closed when availability evidence is missing or inconsistent.
- Chose one narrow, immutable `StrategySpec` for the strategy rules the current engine actually supports; unsupported weighting, execution, or exposure modes are rejected.
- Defined research-production parity narrowly as shared rule and calculation versions across validation and held-out research runs. Broker execution parity remains out of scope.
- Deferred Scenario Lab, calibration/stability panels, and broader QIS machinery until the rule contract and chronological validation layer are reviewable.
- Added an opt-in scheduled research loop that accumulates real first-seen observations in durable PostgreSQL, but refuses ephemeral SQLite in scheduled mode.
- Kept self-calibration validation-only: the loop may propose a challenger but cannot modify the approved champion. Approval/rejection is a separate append-only event with one decision allowed per candidate.
- Chose fail-closed collection semantics. Any requested-ticker fetch failure records the attempt and stops candidate generation for that cycle.
