"""Tests for documented corrections (midnight label fix)."""

from __future__ import annotations

import pandas as pd
import pytest

from steelops.data.clean import CorrectionError, add_interval_start, fix_midnight_labels
from steelops.data.load import parse_frame


def _raw_like_official(synthetic_raw_factory, schema) -> pd.DataFrame:
    """Reproduce the official quirk: 'D+1 00:00' is written as 'D 00:00' (file order kept)."""
    df = parse_frame(synthetic_raw_factory(n=3 * 96), schema)
    midnight = (df["timestamp"].dt.hour == 0) & (df["timestamp"].dt.minute == 0)
    df.loc[midnight, "timestamp"] -= pd.Timedelta("1D")
    return df


def test_fix_restores_regular_increasing_grid(synthetic_raw_factory, schema):
    buggy = _raw_like_official(synthetic_raw_factory, schema)
    assert not buggy["timestamp"].is_monotonic_increasing
    fixed = fix_midnight_labels(buggy)
    assert fixed["timestamp"].is_monotonic_increasing
    assert (fixed["timestamp"].diff().dropna() == pd.Timedelta("15min")).all()
    assert fixed["timestamp"].iloc[-1] == pd.Timestamp("2018-01-04 00:00")


def test_fix_keeps_values_attached_to_their_rows(synthetic_raw_factory, schema):
    buggy = _raw_like_official(synthetic_raw_factory, schema)
    fixed = fix_midnight_labels(buggy)
    pd.testing.assert_series_equal(fixed["usage_kwh"], buggy["usage_kwh"].reset_index(drop=True))


def test_fix_refuses_unexpected_pattern(synthetic_raw_factory, schema):
    clean = parse_frame(synthetic_raw_factory(n=3 * 96), schema)  # already correct labels
    with pytest.raises(CorrectionError):
        fix_midnight_labels(clean)


def test_fix_refuses_sorted_input(synthetic_raw_factory, schema):
    """Sorting the raw labels first destroys the evidence; the fix must refuse it."""
    buggy = _raw_like_official(synthetic_raw_factory, schema)
    with pytest.raises(CorrectionError):
        fix_midnight_labels(buggy.sort_values("timestamp"))


def test_interval_start_is_fifteen_minutes_before(synthetic_raw_factory, schema):
    df = add_interval_start(parse_frame(synthetic_raw_factory(n=4), schema))
    assert ((df["timestamp"] - df["interval_start"]) == pd.Timedelta("15min")).all()
