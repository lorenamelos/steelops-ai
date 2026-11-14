"""Reproducible data-quality audit of the raw dataset.

Every check is a small pure function that returns JSON-serialisable facts. The audit
reports what the data *is*; it never modifies it. Interpretation lives in the docs.

Run with: ``uv run steelops-audit``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steelops.config import DEFAULT_DATA_CONFIG, load_config, resolve_path
from steelops.data.load import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    check_columns,
    parse_frame,
    raw_csv_path,
    read_raw_frame,
)

NA_TOKENS = {"", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "-"}
SECONDS_PER_DAY = 86_400


# --------------------------------------------------------------------------- helpers
def _num(x: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON output."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, pd.Timedelta):
        return str(x)
    return x


# --------------------------------------------------------------------------- checks
def check_structure(raw: pd.DataFrame) -> dict[str, Any]:
    return {"n_rows": len(raw), "n_columns": raw.shape[1], "columns": list(raw.columns)}


def check_missing_tokens(raw: pd.DataFrame) -> dict[str, int]:
    """Count empty / NA-like tokens per raw column (read as strings, no NA inference)."""
    return {col: int(raw[col].str.strip().isin(NA_TOKENS).sum()) for col in raw.columns}


def check_parse_failures(raw: pd.DataFrame, parsed: pd.DataFrame, schema: dict) -> dict[str, int]:
    """Values that were present in the raw file but could not be parsed."""
    inv = {v: k for k, v in schema["columns"].items()}
    out = {}
    for col in ("timestamp", *NUMERIC_COLUMNS):
        present = ~raw[inv[col]].str.strip().isin(NA_TOKENS)
        out[col] = int((parsed[col].isna() & present).sum())
    return out


def check_timeline(ts: pd.Series, freq: str) -> dict[str, Any]:
    """Ordering, duplicates, spacing, and completeness against a regular grid."""
    ts = ts.dropna()
    step = pd.Timedelta(freq)
    diffs = ts.diff().dropna()
    diff_counts = diffs.value_counts().sort_index()
    sorted_unique = pd.Series(ts.sort_values().unique())
    grid = pd.date_range(sorted_unique.iloc[0], sorted_unique.iloc[-1], freq=freq)
    missing = grid.difference(pd.DatetimeIndex(sorted_unique))
    return {
        "start": _num(ts.min()),
        "end": _num(ts.max()),
        "span": str(ts.max() - ts.min()),
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
        "n_backward_steps": int((diffs < pd.Timedelta(0)).sum()),
        "n_duplicate_timestamps": int(ts.duplicated().sum()),
        "n_unique_timestamps": int(ts.nunique()),
        "expected_grid_size": len(grid),
        "n_missing_grid_timestamps": len(missing),
        "missing_grid_timestamps_sample": [t.isoformat() for t in missing[:20]],
        "interval_counts": {str(k): int(v) for k, v in diff_counts.items()},
        "share_regular_intervals": _num((diffs == step).mean()) if len(diffs) else None,
    }


def check_interval_labelling(df: pd.DataFrame) -> dict[str, Any]:
    """Infer whether timestamps mark the START or the END of each 15-minute interval.

    Evidence used:
    * time-of-day of the first record (00:15 suggests interval-end labels);
    * agreement of NSM (seconds since midnight) with the timestamp clock time;
    * agreement of Day_of_week / WeekStatus with the calendar day of the timestamp
      versus the calendar day of the interval start (timestamp - 15 min).
    """
    ts = df["timestamp"]
    clock_seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    interval_start = ts - pd.Timedelta("15min")
    midnight = ts.dt.time == pd.Timestamp("00:00").time()

    def weekend_label(day: pd.Series) -> pd.Series:
        return day.isin(["Saturday", "Sunday"]).map({True: "Weekend", False: "Weekday"})

    dow = df["day_of_week"]
    ws = df["week_status"]
    return {
        "first_record_time_of_day": str(ts.iloc[0].time()),
        "last_record_time_of_day": str(ts.iloc[-1].time()),
        "nsm_equals_clock_seconds_share": _num((df["nsm"] == clock_seconds).mean()),
        "nsm_at_midnight_values": sorted({int(v) for v in df.loc[midnight, "nsm"].unique()}),
        "nsm_min": _num(df["nsm"].min()),
        "nsm_max": _num(df["nsm"].max()),
        "day_of_week_matches_timestamp_day_share": _num((dow == ts.dt.day_name()).mean()),
        "day_of_week_matches_interval_start_day_share": _num(
            (dow == interval_start.dt.day_name()).mean()
        ),
        "day_of_week_mismatch_rows_timestamp_day": int((dow != ts.dt.day_name()).sum()),
        "week_status_values": sorted(ws.dropna().unique().tolist()),
        "week_status_matches_timestamp_day_share": _num(
            (ws == weekend_label(ts.dt.day_name())).mean()
        ),
        "week_status_matches_interval_start_day_share": _num(
            (ws == weekend_label(interval_start.dt.day_name())).mean()
        ),
    }


def check_numeric_ranges(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for col in NUMERIC_COLUMNS:
        s = df[col]
        q = s.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        out[col] = {
            "mean": _num(s.mean()),
            "std": _num(s.std()),
            "min": _num(q.loc[0.0]),
            "p01": _num(q.loc[0.01]),
            "p25": _num(q.loc[0.25]),
            "median": _num(q.loc[0.5]),
            "p75": _num(q.loc[0.75]),
            "p99": _num(q.loc[0.99]),
            "max": _num(q.loc[1.0]),
            "n_zero": int((s == 0).sum()),
            "n_negative": int((s < 0).sum()),
            "n_unique": int(s.nunique()),
            "max_decimals": _max_decimals(s),
        }
    return out


def _max_decimals(s: pd.Series) -> int:
    """Largest number of decimals needed to represent the values (measurement resolution)."""
    vals = s.dropna().to_numpy(dtype=float)
    for d in range(0, 7):
        if np.allclose(vals, np.round(vals, d), atol=1e-9, rtol=0):
            return d
    return 7


def check_categoricals(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
        for col in CATEGORICAL_COLUMNS
    }


def implied_power_factor(active_kwh: pd.Series, reactive_kvarh: pd.Series) -> pd.Series:
    """Power factor (%) implied by active and reactive energy over the same interval.

    PF = P / S = kWh / sqrt(kWh^2 + kVArh^2). Undefined (NaN) when both are zero.
    """
    apparent = np.sqrt(active_kwh**2 + reactive_kvarh**2)
    return (100 * active_kwh / apparent).where(apparent > 0)


def check_power_factor_consistency(df: pd.DataFrame) -> dict[str, Any]:
    """Compare reported power factors with the PF implied by energy columns."""
    out: dict[str, Any] = {}
    pairs = {
        "lagging": ("lagging_reactive_kvarh", "lagging_power_factor"),
        "leading": ("leading_reactive_kvarh", "leading_power_factor"),
    }
    for name, (kvarh, pf) in pairs.items():
        implied = implied_power_factor(df["usage_kwh"], df[kvarh])
        defined = implied.notna()
        abs_err = (implied - df[pf]).abs()[defined]
        both_zero = (df["usage_kwh"] == 0) & (df[kvarh] == 0)
        out[name] = {
            "n_defined": int(defined.sum()),
            "median_abs_diff_pp": _num(abs_err.median()),
            "p99_abs_diff_pp": _num(abs_err.quantile(0.99)),
            "max_abs_diff_pp": _num(abs_err.max()),
            "share_within_0_5_pp": _num((abs_err <= 0.5).mean()),
            "n_both_zero": int(both_zero.sum()),
            "reported_pf_when_both_zero": {
                str(k): int(v) for k, v in df.loc[both_zero, pf].value_counts().head(5).items()
            },
            "n_out_of_0_100": int(((df[pf] < 0) | (df[pf] > 100)).sum()),
        }
    return out


def check_co2(df: pd.DataFrame) -> dict[str, Any]:
    """Characterise CO2 relative to active energy (ratio = implied emission factor)."""
    co2, kwh = df["co2_tco2"], df["usage_kwh"]
    positive = kwh > 0
    ratio = (co2 / kwh)[positive]
    nonzero_co2 = co2[co2 > 0]
    diffs = np.diff(np.sort(co2.unique()))
    return {
        "n_unique": int(co2.nunique()),
        "smallest_positive_step": _num(diffs[diffs > 0].min()) if (diffs > 0).any() else None,
        "share_zero_co2": _num((co2 == 0).mean()),
        "share_zero_co2_when_usage_positive": _num((co2[positive] == 0).mean()),
        "share_positive_co2_when_usage_zero": _num((co2[~positive] > 0).mean())
        if (~positive).any()
        else None,
        "ratio_co2_per_kwh_quantiles": {
            k: _num(v)
            for k, v in ratio[co2[positive] > 0]
            .quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            .rename(lambda q: f"p{int(q * 100):02d}")
            .items()
        },
        "pearson_corr_with_usage": _num(co2.corr(kwh)),
        "min_positive_value": _num(nonzero_co2.min()) if len(nonzero_co2) else None,
    }


def check_load_type_determinism(df: pd.DataFrame) -> dict[str, Any]:
    """Is Load_Type a deterministic function of the calendar?

    For each candidate grouping, purity = share of rows whose Load_Type equals the
    modal Load_Type of its group. Purity of 1.0 means Load_Type is fully determined
    by that calendar key (and would be known in advance, like a tariff schedule).
    """
    ts = df["timestamp"]
    keys = {
        "time_of_day": ts.dt.strftime("%H:%M"),
        "month": ts.dt.month,
        "is_weekend": df["week_status"].eq("Weekend"),
        "day_of_week": df["day_of_week"],
    }
    groupings = {
        "time_of_day": ["time_of_day"],
        "time_of_day+is_weekend": ["time_of_day", "is_weekend"],
        "time_of_day+month": ["time_of_day", "month"],
        "time_of_day+month+is_weekend": ["time_of_day", "month", "is_weekend"],
        "time_of_day+month+day_of_week": ["time_of_day", "month", "day_of_week"],
    }
    frame = pd.DataFrame(keys).assign(load_type=df["load_type"])
    out = {}
    for name, cols in groupings.items():
        modal = frame.groupby(cols)["load_type"].transform(lambda s: s.mode().iloc[0])
        out[name] = {
            "purity": _num((frame["load_type"] == modal).mean()),
            "n_groups": int(frame.groupby(cols).ngroups),
        }
    return out


# --------------------------------------------------------------------------- runner
def run_audit(raw: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    """Run every check on a raw string frame and return a JSON-serialisable report."""
    check_columns(raw, schema["columns"])
    parsed = parse_frame(raw, schema, strict=False)
    return {
        "structure": check_structure(raw),
        "missing_tokens": check_missing_tokens(raw),
        "parse_failures": check_parse_failures(raw, parsed, schema),
        "timeline": check_timeline(parsed["timestamp"], schema["expected_frequency"]),
        "interval_labelling": check_interval_labelling(parsed),
        "numeric_ranges": check_numeric_ranges(parsed),
        "categoricals": check_categoricals(parsed),
        "power_factor_consistency": check_power_factor_consistency(parsed),
        "co2": check_co2(parsed),
        "load_type_determinism": check_load_type_determinism(parsed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the raw SteelOps dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run_audit(read_raw_frame(raw_csv_path(config)), config["schema"])
    out_path = resolve_path(config["paths"]["audit_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nAudit report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
