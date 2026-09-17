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
