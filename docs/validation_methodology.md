# Validation Methodology

## Why Raw Return Is Not Enough

Raw return can be misleading. A stock can move because the whole market moved, because its sector moved, or because high beta amplified a broad benchmark move.

## Abnormal Return

ThesisBoard's validation spine is built around abnormal return: ticker return after adjusting for market and, when available, sector or theme proxy movement.

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

## Hit Rate And Base Rate

Hit rate is only meaningful relative to a base rate. If a Tradeable signal has a 56% hit rate but the same universe has a 54% base rate, the excess hit rate is only 2 percentage points.

## Causality Limits

Daily data cannot establish precise intraday causality. It can support a qualitative event-study workflow, but it cannot prove that a headline caused a move.

## Current App Scope

The deployable app uses demo data only. It exists to make the validation workflow reviewable before adding real signal history, price data, or news evidence.
