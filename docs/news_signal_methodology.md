# News Signal Methodology

## Purpose

News Signal Lab tests whether headline text contains out-of-sample information about future abnormal returns. It is a research workflow, not a recommendation engine or brokerage system.

## The Four Clocks

Every experiment distinguishes four timestamps:

1. `published_at`: the timestamp reported by the news provider. This is metadata and may be incomplete, revised, or wrong.
2. `first_seen_at`: the UTC timestamp when ThesisBoard actually captured the headline. This is the availability clock used by the model.
3. `signal_session`: the first regular market close at which the captured headline can enter a daily close-to-close feature set. Headlines first observed at or after 4:00 p.m. America/New_York move to the next benchmark session.
4. `label_available_at`: the market close when the forward-return horizon finishes and the target becomes observable.

Vendor publication time is never substituted for `first_seen_at`. This prevents a recently installed system from pretending that it possessed old headlines historically.

## Immutable Capture

Normalized headline versions are stored in an append-only SQLite table. Identical content is idempotent. If a provider changes the title, timestamp, publisher, or canonical URL, the changed content receives a new content hash and is stored as another immutable version. When multiple versions of the same article arrive before one signal close, feature construction uses only the latest version known by that close; the older version remains available for audit.

The repository ignores the local database. Captured headlines are user-local research data and must not be committed.

## Daily Features

Headlines are grouped by ticker and eligible signal session. The baseline uses:

- TF-IDF unigrams and bigrams fit inside each training window;
- VADER compound-score mean, minimum, and maximum;
- positive, negative, and neutral headline shares;
- headline count and mean title length.

VADER is a generic lexical baseline, not a finance-specialized language model. Its purpose is to establish an interpretable floor that a more complex model must beat.

## Targets

The initial supported horizons are one and three benchmark sessions. Labels use split/dividend-adjusted closes and ThesisBoard's existing abnormal-return engine. A ticker must have an exact adjusted close on both the signal session and horizon end; the lab does not silently shift a label across a missing session. Rolling beta is estimated only from returns before the signal session. The engine accepts ticker-specific sector proxies, while the first UI slice currently runs the SPY beta-adjusted target without a sector mapping. Rows with missing prices or beta fallback are retained for audit but excluded from the strict modeling sample.

## Chronological Model Evaluation

The lab runs an expanding-window logistic regression baseline and a constrained random-forest challenger. Both models receive the same TF-IDF/VADER feature set and are compared only on their common ticker/session rows. The tree is intentionally depth- and leaf-constrained; adding capacity is not evidence of an improvement.

For each test session:

- TF-IDF vocabulary, scaling, and model parameters are fit from scratch;
- a training row is eligible only if its `label_available_at` precedes the test-session close;
- at least the configured number of rows and sessions must be available;
- both outcome classes must exist in training.

The lab reports out-of-sample probability diagnostics such as directional accuracy, a training-history positive-rate baseline, Brier score, log loss, ROC AUC when defined, and Spearman information coefficient when defined. The baseline probability is recomputed from each expanding training window; it is not inferred from the future test distribution. Full-window diagnostics are descriptive and do not choose the event policy.

## Validation-Selected Strategy Policy

The policy audit makes a chronological split inside the out-of-sample predictions. It considers a fixed threshold grid and both model families on the earlier validation segment. A candidate must satisfy the configured maximum selection rate, minimum signal count, and maximum average daily turnover. Each candidate is passed through the stateful position book described below. Selection maximizes validation-period net compounded return minus the exposure-matched SPY return after realized weight-change costs, rather than optimizing an event average or a test-period statistic.

Before selection, validation rows whose `label_available_at` is not earlier than the first test close are purged. The chosen model and probability threshold are then locked and evaluated on the later test segment. Test outcomes never enter the programmatic selection rule.

This discipline does not survive repeated human tuning. If a user changes settings after seeing the test result, that period has become exploratory and a future untouched period is required for confirmation.

## What The Event Policy Is Not

Selected-event averages are retained as a diagnostic, not a portfolio. They do not define capital allocation, concurrent-position handling, cash drag, or a daily position ledger. The separate position-book audit below supplies a deliberately narrow portfolio interpretation; the event table itself must not be called portfolio performance.

## Locked Holdout Position Book

The same position-book engine first evaluates candidates on the validation segment and then starts a fresh, flat book at the first test session for the locked model and threshold. A selected event creates a long position from its eligible signal close until its recorded horizon-end close. Concurrent active tickers are equal weighted, unallocated capital is zero-return cash, and overlapping selected events for one ticker collapse to one position instead of creating accidental leverage.

At every close, turnover is the sum of absolute changes in ticker weights. The audit charges a configurable one-way cost per unit of turnover, including initial entry, rebalancing, and final liquidation. It reports gross and net compounded return, total and annualized turnover, average exposure, maximum drawdown, and an exposure-matched SPY comparison. Missing adjusted closes on any active interval fail the audit instead of silently shifting the trade.

This is a reproducible accounting layer, not proof of executable performance. It assumes a signal using pre-close observations can transact at the eligible adjusted close, ignores bid/ask dynamics and market impact, assumes zero cash return, and does not model order fills, liquidity, borrow, taxes, or capacity. A real deployment claim would need a delayed execution rule, higher-quality point-in-time data, and an untouched future evaluation window.

## Claims The Lab Does Not Make

The lab does not claim deployable alpha from model diagnostics, the synthetic demo, or the position-book audit. It does not recommend a security or size a live position. A held-out research result is still only one estimate and needs confirmation on a newly accumulated, untouched period.

## Known Calendar Limitation

The current availability rule assumes the regular 4:00 p.m. America/New_York close. It does not yet model exchange-specific early closes. The rule version is recorded on every dataset row so a future calendar implementation can be compared explicitly.
