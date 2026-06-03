from __future__ import annotations

HORIZON_1D = 1
HORIZON_3D = 3
HORIZON_5D = 5
HORIZON_20D = 20
HORIZON_60D = 60

STANDARD_HORIZONS = [HORIZON_1D, HORIZON_3D, HORIZON_5D, HORIZON_20D, HORIZON_60D]

HORIZON_LABELS = {
    HORIZON_1D: "1D event reaction",
    HORIZON_3D: "3D event digestion",
    HORIZON_5D: "5D swing validation",
    HORIZON_20D: "20D theme validation",
    HORIZON_60D: "60D broader context",
}


def validate_horizon(horizon_days: int) -> int:
    if horizon_days not in STANDARD_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_days}")
    return horizon_days


def horizon_label(horizon_days: int) -> str:
    return HORIZON_LABELS[validate_horizon(horizon_days)]


def default_horizons(signal_type: str) -> list[int]:
    defaults = {
        "event_reaction": [HORIZON_1D, HORIZON_3D],
        "trade_setup": [HORIZON_5D],
        "theme_thesis": [HORIZON_20D],
        "context": [HORIZON_60D],
    }
    if signal_type not in defaults:
        raise ValueError(f"Unsupported signal type: {signal_type}")
    return defaults[signal_type]
