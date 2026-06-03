import pandas as pd
import pytest

from app import (
    build_demo_validation_table,
    calculate_excess_hit_rate,
    positive_abnormal_return_hit,
    split_tradeable_and_avoid_chase,
)


def test_excess_hit_rate_is_hit_rate_minus_base_rate():
    assert calculate_excess_hit_rate(0.56, 0.51) == pytest.approx(0.05)


def test_positive_abnormal_return_produces_bullish_hit():
    assert positive_abnormal_return_hit(0.02, bullish_signal=True) is True
    assert positive_abnormal_return_hit(-0.01, bullish_signal=True) is False


def test_avoid_chase_tracked_separately_from_tradeable_hit_rate():
    df = build_demo_validation_table()
    tradeable, avoid_chase = split_tradeable_and_avoid_chase(df)
    assert "Avoid Chase" not in set(tradeable["classification"])
    assert set(avoid_chase["classification"]) == {"Avoid Chase"}
    assert pd.isna(avoid_chase.iloc[0]["hit"])
