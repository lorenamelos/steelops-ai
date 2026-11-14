"""Tests for parsing and audit checks, using synthetic data with planted defects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steelops.data.audit import (
    check_interval_labelling,
    check_load_type_determinism,
    check_power_factor_consistency,
    check_timeline,
    implied_power_factor,
    run_audit,
)
from steelops.data.load import SchemaError, parse_frame


# ---------------------------------------------------------------- parsing
def test_parse_renames_and_types(synthetic_raw, schema):
    df = parse_frame(synthetic_raw, schema)
    assert list(df.columns) == list(schema["columns"].values())
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["usage_kwh"])


def test_explicit_format_parses_day_first(synthetic_raw_factory, schema):
    """01/02/2018 must be 1 February, never 2 January."""
    raw = synthetic_raw_factory(n=1)
    raw.loc[0, "date"] = "01/02/2018 10:00"
    df = parse_frame(raw, schema)
    assert df.loc[0, "timestamp"] == pd.Timestamp("2018-02-01 10:00")


def test_missing_column_raises(synthetic_raw, schema):
    with pytest.raises(SchemaError, match="Missing"):
        parse_frame(synthetic_raw.drop(columns=["Load_Type"]), schema)


def test_unexpected_column_raises(synthetic_raw, schema):
    with pytest.raises(SchemaError, match="Unexpected"):
        parse_frame(synthetic_raw.assign(extra="x"), schema)


def test_strict_parse_rejects_bad_timestamp(synthetic_raw, schema):
    synthetic_raw.loc[3, "date"] = "2018-01-01T00:45"
    with pytest.raises(ValueError):
        parse_frame(synthetic_raw, schema, strict=True)
    lenient = parse_frame(synthetic_raw, schema, strict=False)
    assert lenient["timestamp"].isna().sum() == 1


# ---------------------------------------------------------------- timeline
def _ts(n: int = 20) -> pd.Series:
    return pd.Series(pd.date_range("2018-01-01 00:15", periods=n, freq="15min"))


def test_clean_timeline():
    out = check_timeline(_ts(), "15min")
    assert out["is_monotonic_increasing"]
    assert out["n_duplicate_timestamps"] == 0
    assert out["n_missing_grid_timestamps"] == 0
    assert out["share_regular_intervals"] == 1.0


def test_timeline_detects_gap_duplicate_and_disorder():
    ts = _ts()
    ts = ts.drop(index=5)  # gap
    ts = pd.concat([ts, ts.iloc[[2]]])  # duplicate appended at the end -> backward step
    out = check_timeline(ts.reset_index(drop=True), "15min")
    assert out["n_missing_grid_timestamps"] == 1
    assert out["n_duplicate_timestamps"] == 1
    assert not out["is_monotonic_increasing"]
    assert out["n_backward_steps"] == 1


# ---------------------------------------------------------------- semantics
def test_interval_labelling_on_end_labelled_synthetic(synthetic_raw, schema):
    df = parse_frame(synthetic_raw, schema)
    out = check_interval_labelling(df)
    assert out["first_record_time_of_day"] == "00:15:00"
    assert out["nsm_equals_clock_seconds_share"] == 1.0
    assert out["day_of_week_matches_interval_start_day_share"] == 1.0
    assert out["day_of_week_matches_timestamp_day_share"] < 1.0  # midnight rows differ


def test_implied_power_factor_formula():
    # 3-4-5 triangle: P=3, Q=4 -> S=5 -> PF=60 %
    pf = implied_power_factor(pd.Series([3.0, 0.0, 5.0]), pd.Series([4.0, 0.0, 0.0]))
    assert pf.iloc[0] == pytest.approx(60.0)
    assert np.isnan(pf.iloc[1])  # undefined when P = Q = 0
    assert pf.iloc[2] == pytest.approx(100.0)


def test_power_factor_consistency_detects_corruption(synthetic_raw, schema):
    df = parse_frame(synthetic_raw, schema)
    clean = check_power_factor_consistency(df)["lagging"]
    assert clean["share_within_0_5_pp"] == 1.0
    df.loc[:9, "lagging_power_factor"] = 10.0
    corrupted = check_power_factor_consistency(df)["lagging"]
    assert corrupted["share_within_0_5_pp"] < 1.0


def test_load_type_determinism_purity(synthetic_raw, schema):
    df = parse_frame(synthetic_raw, schema)
    assert check_load_type_determinism(df)["time_of_day"]["purity"] == 1.0
    rng = np.random.default_rng(0)
    df["load_type"] = pd.array(rng.choice(["A", "B", "C"], len(df)), dtype="string")
    assert check_load_type_determinism(df)["time_of_day"]["purity"] < 0.9


def test_run_audit_is_json_serialisable(synthetic_raw, schema):
    import json

    report = run_audit(synthetic_raw, schema)
    json.dumps(report, default=str)
    assert report["structure"]["n_rows"] == len(synthetic_raw)
    assert sum(report["missing_tokens"].values()) == 0
