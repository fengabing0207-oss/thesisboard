# Automated Research Loop Operations

## Safety Contract

The automation collects first-seen headlines, evaluates matured point-in-time history, and may propose a rule-based challenger. It does not execute trades and cannot automatically promote a strategy. The held-out window is logged for review but never used by the programmatic selection rule.

Scheduled mode fails closed when durable storage is unavailable or any requested ticker fails collection. Local SQLite remains available for development and manual capture, but `--require-durable` refuses it because a hosted runner's filesystem is ephemeral.

## Enable The Schedulers

The workflow `.github/workflows/news-headline-collection.yml` captures headlines every two hours at minute 17. The existing `.github/workflows/news-research-cycle.yml` performs settled-label calibration at 23:30 UTC on weekdays, safely after 6:30 p.m. New York in both daylight and standard time. GitHub schedules execute only after the workflows are present on the default branch, and GitHub may delay cron jobs.

Configure these repository settings:

| Type | Name | Required | Purpose |
| --- | --- | --- | --- |
| Secret | `THESISBOARD_DATABASE_URL` | Yes | PostgreSQL connection URL for durable headlines, collection runs, and research events. |
| Variable | `THESISBOARD_AUTOMATION_ENABLED` | Yes for cron | Set exactly to `true` to enable scheduled cycles. |
| Variable | `THESISBOARD_WATCHLIST` | No | Comma- or space-separated ticker list; otherwise the code uses its documented default watchlist. |

The database role needs permission to create tables and indexes on first run and to select and insert afterward. Do not place the database URL in a repository variable, command line, committed file, or issue. The application reports only the backend type and never returns the configured URL.

Before enabling cron, run both workflows manually once. Confirm that News Signal Lab shows PostgreSQL, a successful capture, a completed calibration, and expected ticker coverage. A manual dispatch still requires the database secret even when the cron enable variable is false.

## Local Commands

Run a cycle against local SQLite for development:

```bash
python scripts/run_research_cycle.py AAA BBB
```

Capture headlines without running prices or models:

```bash
python scripts/collect_news.py AAA BBB
```

Require durable PostgreSQL, matching the scheduled workflow:

```bash
THESISBOARD_DATABASE_URL='postgresql://…' \
  python scripts/run_research_cycle.py --require-durable AAA BBB
```

Approve or reject a proposed candidate only after reviewing its recorded validation window, held-out diagnostics, costs, turnover, and calculation parity:

```bash
THESISBOARD_DATABASE_URL='postgresql://…' \
  python scripts/promote_strategy.py 42 approved \
  --decided-by 'research-owner' \
  --note 'Reviewed the candidate audit'
```

The reviewer string is an audit assertion, not identity verification. Restrict workflow and database access through the hosting platform.

## Event Ledger

The `research_events` table records these event types:

| Event | Meaning |
| --- | --- |
| `automation_run_started` | Immutable run ID, configuration, watchlist, and storage mode. |
| `collection_completed` | Counts, provider errors, and the linked collection-run ID. |
| `dataset_integrity_checkpoint` | Last accepted settled-label manifest and readiness state. |
| `dataset_integrity_violation` | A decrease, removal, or direction flip that stopped calibration. |
| `strategy_candidate_proposed` | New spec or new dataset vintage pending review. |
| `strategy_candidate_unchanged` | Same spec and same dataset vintage as the last proposal. |
| `automation_run_completed` | Terminal non-error status, including insufficient-data states. |
| `automation_run_failed` | Unexpected exception type and message; the exception is re-raised so the workflow fails visibly. |
| `strategy_promotion_approved` / `strategy_promotion_rejected` | Explicit review decision. Each candidate can receive only one. |

Expected early statuses such as `no_captured_headlines`, `insufficient_matured_data`, insufficient chronological rows, or insufficient validation/test sessions are successful no-candidate outcomes. `insufficient_matured_data` is the normal cold-start state before captured headlines have forward-return labels; a non-empty dataset that fails the feature-availability audit remains a distinct fail-closed result. These states are evidence that the guardrails are working, not reasons to loosen the split after seeing results.

`dataset_integrity_regressed` is different: the command exits non-zero and the
workflow is red. Inspect its recorded violations before retrying. Do not delete
the previous checkpoint or loosen the invariant to make the run green.

## Readiness During Cold Start

News Signal Lab reports two separate kinds of progress:

- **Operational readiness** checks durable storage, the latest collection result, scheduler freshness, expected-universe capture, and timezone-aware `observed_at` provenance.
- **Model-window readiness** is written into each completed priced-cycle event as `dataset_readiness`. It counts exact matured and usable signal sessions against the configured training, validation, and test floors.

Scheduled labels are not usable until 90 minutes after the recorded horizon
close. This prevents a provisional yfinance daily row from being treated as a
final close during market hours.

Complete collection weekdays are only a scheduler-continuity proxy. They are not interchangeable with usable signal sessions: a day without an eligible headline or a mature, clean return label does not create a model row. Reaching the default 25-session floor permits the full evaluation gate to run but does not guarantee a candidate or establish predictive power.

## Disable And Recover

Set `THESISBOARD_AUTOMATION_ENABLED` to anything other than `true` to stop future cron jobs. Existing headlines and events remain untouched. Because records are append-only, an incorrect approval is not edited or deleted; record a superseding reviewed candidate in a future product change. Investigate repeated provider failures before re-enabling rather than allowing partial-universe calibration.
