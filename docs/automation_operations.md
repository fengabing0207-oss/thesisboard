# Automated Research Loop Operations

## Safety Contract

The automation collects first-seen headlines, evaluates matured point-in-time history, and may propose a rule-based challenger. It does not execute trades and cannot automatically promote a strategy. The held-out window is logged for review but never used by the programmatic selection rule.

Scheduled mode fails closed when durable storage is unavailable or any requested ticker fails collection. Local SQLite remains available for development and manual capture, but `--require-durable` refuses it because a hosted runner's filesystem is ephemeral.

## Enable The Schedulers

The workflow `.github/workflows/news-headline-collection.yml` wakes hourly at minute 37, with a 90-minute freshness gate to target an approximately two-hour capture cadence. Only a complete run with usable headlines (new or already stored) can satisfy that gate. The existing `.github/workflows/news-research-cycle.yml` performs settled-label calibration at 23:30 UTC on weekdays, safely after 6:30 p.m. New York in both daylight and standard time. GitHub schedules execute only after the workflows are present on the default branch, and GitHub may delay or drop cron jobs.

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
| `collection_completed` | Counts, provider errors, per-ticker fetch methods/fallback reasons, and the linked collection-run ID. |
| `dataset_integrity_checkpoint` | Last accepted settled-label manifest and readiness state. |
| `dataset_integrity_violation` | A decrease, removal, or direction flip that stopped calibration. |
| `strategy_candidate_proposed` | New spec or new dataset vintage pending review. |
| `strategy_candidate_unchanged` | Same spec and same dataset vintage as the last proposal. |
| `automation_run_completed` | Terminal outcome, including data-health stops and insufficient-data states; inspect the payload status. |
| `automation_run_failed` | Unexpected exception type and message; the exception is re-raised so the workflow fails visibly. |
| `strategy_promotion_approved` / `strategy_promotion_rejected` | Explicit review decision. Each candidate can receive only one. |

Expected early statuses such as `insufficient_matured_data`, insufficient chronological rows, or insufficient validation/test sessions are successful no-candidate outcomes. `insufficient_matured_data` is the normal cold-start state before captured headlines have forward-return labels; a non-empty dataset that fails the feature-availability audit remains a distinct fail-closed result. These states are evidence that the guardrails are working, not reasons to loosen the split after seeing results.

`collection_incomplete`, `no_captured_headlines`, and `missing_benchmark_prices`
are data-health failures: the research command exits non-zero, so execution-failure
monitors see a failed job. A successful exit is not evidence of predictive value.

## Empty News Responses And Recovery

The fetch adapter tries `Ticker.news`, then `Search(...).news` if the primary
response is empty, malformed, title-less, or raises. Search results must have
a nonblank text title and an exact `relatedTickers` match. No paid key is needed.
These tags establish provider association, not headline relevance or complete
coverage; search can return a different selection of articles from ticker news.
Both routes keep the `yfinance` provider identity and existing URL/content
deduplication rules. The fetch route and fallback reason appear in per-ticker
CLI diagnostics and in the research cycle's durable `collection_completed` event.

If neither route yields usable headlines for a requested ticker, that ticker
is a failed collection, even if the HTTP client raised no exception. Good
tickers are still stored, but partial batches stop calibration and make both
commands exit non-zero. An entirely duplicate, non-empty feed remains healthy:
`inserted_versions=0` alone is not a failure. Legacy all-empty "success" rows
no longer satisfy the freshness gate or count as complete collection weekdays
in the readiness report. Historical rows are never rewritten.

On 2026-10-27, a live check reproduced a JSON 404 from Yahoo's `/xhr/ncp`
endpoint while yfinance 1.7.0 returned `[]`. The search route returned tagged
headlines for all seven default symbols. The last productive run before the
observed gap was [2026-10-01 16:42 UTC](https://github.com/fengabing0207-oss/thesisboard/actions/runs/36893786673)
(70 fetched, 46 inserted); the next capture at
[22:00 UTC](https://github.com/fengabing0207-oss/thesisboard/actions/runs/36932219486)
returned zero. Inspection through the 2026-10-27 06:48 UTC run found 138 actual
zero-yield collections, six freshness skips, and one failed run whose logs were
unavailable. This establishes the observed outage window, not Yahoo's exact
endpoint-change time. Upstream also reports the silent-404 failure in
[yfinance issue #3004](https://github.com/ranaroussi/yfinance/issues/3004).

After merging a provider repair, manually dispatch headline collection and
confirm each ticker's usable counts and durable inserts; then dispatch the
research cycle and inspect its status/readiness. Newly recovered articles must
use the time actually observed, even when their publication date is old.
Do not backdate `first_seen_at` or manufacture missing sessions to reach 25 days.

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
