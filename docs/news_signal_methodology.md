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

## Chronological Evaluation

The baseline is an expanding-window logistic regression. For each test session:

- TF-IDF vocabulary, scaling, and model parameters are fit from scratch;
- a training row is eligible only if its `label_available_at` precedes the test-session close;
- at least the configured number of rows and sessions must be available;
- both outcome classes must exist in training.

The lab reports out-of-sample probability diagnostics such as directional accuracy, a training-history positive-rate baseline, Brier score, log loss, ROC AUC when defined, and Spearman information coefficient when defined. The baseline probability is recomputed from each expanding training window; it is not inferred from the future test distribution.

## Claims The Lab Does Not Make

The first version does not optimize a trading threshold, position size, signal frequency, turnover, or transaction-cost-adjusted return. It does not claim alpha from an in-sample fit. Strategy evaluation belongs in a later layer after enough point-in-time observations and matured labels exist.

## Known Calendar Limitation

The current availability rule assumes the regular 4:00 p.m. America/New_York close. It does not yet model exchange-specific early closes. The rule version is recorded on every dataset row so a future calendar implementation can be compared explicitly.
