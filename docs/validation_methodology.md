# Validation Methodology

ThesisBoard validation asks whether a recorded signal later outperformed a relevant benchmark or theme proxy. The system should evaluate recorded classifications against forward abnormal returns, not persuasive narrative quality.

## Why Raw Return Is Not Enough

Raw return can be misleading. A stock can move because the whole market moved, because its sector moved, or because high beta amplified a broad benchmark move.

## Abnormal Return

Raw return is not enough because a ticker can rise with the whole market or sector. Validation therefore tracks beta-adjusted, sector-adjusted, and combined abnormal return. Beta is estimated only from observations strictly before the signal start date. If there is not enough history to estimate beta, the fallback beta is 1.0 and the result is flagged with `beta_fallback_used`.

The MVP concept:

```text
abnormal_return = ticker_return - beta * market_return
```

When a theme proxy exists:

```text
abnormal_return = ticker_return - 0.5 * beta * market_return - 0.5 * sector_proxy_return
```

## Forward-Return Tracking

A signal should be recorded at creation time, then evaluated after its intended horizon. Example horizons:

- 1D and 3D: event reaction
- 5D: swing setup validation
- 20D: theme thesis validation
- 60D: broader context

## Trading-Session Horizons

Forward horizons use benchmark trading sessions as the canonical calendar. For example, a 5D horizon means the fifth available benchmark session after the signal timestamp, not five calendar days and not five ticker-specific rows.

The benchmark calendar is used because it gives all signals a common market-session clock. Ticker histories may have missing sessions because of data gaps, halts, or sparse prototype data, but the validation horizon should still mean the same market interval across the cohort.

## Base Rate

Base rate must come from a same-date, same-horizon universe cohort, not only from the model's selected matured signals. Model metrics should be grouped by horizon and classification so the tradeable hit rate can be compared with an appropriate cohort base rate.

Hit rate is only meaningful relative to a base rate. If a Tradeable signal has a 56% hit rate but the same universe has a 54% base rate, the excess hit rate is only 2 percentage points.

## Outcome Semantics

The generic `hit` field is deprecated. Validation should use explicit outcome fields:

- `trade_hit` for Tradeable signals with positive forward abnormal return.
- `watch_followthrough` for Watch or Wait for Confirmation signals with positive forward abnormal return.
- `avoided_bad_trade` for Avoid or Avoid Chase signals with non-positive forward abnormal return.
- `false_negative` mainly for Avoid or Avoid Chase signals that later strongly outperform.

Watch and Wait for Confirmation are not Avoid classifications. A positive forward abnormal return after Watch is follow-through, not a false negative.

## Causality Limits

Daily data cannot establish precise intraday causality. It can support a qualitative event-study workflow, but it cannot prove that a headline caused a move.

## Current App Scope

The deployable app uses demo data only. It exists to make the validation workflow reviewable before adding real signal history, price data, or news evidence.
