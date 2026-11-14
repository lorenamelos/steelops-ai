"""Explicit, documented corrections applied on top of the immutable raw data.

Correction 1 — midnight timestamp labels
----------------------------------------
Timestamps mark the END of each 15-minute interval (the first record is 00:15).
The record that closes each day is stored as ``D 00:00`` but sits in the file right
after ``D 23:45``; its value is continuous with ``D 23:45`` and ``D+1 00:15``. It is the
interval 23:45-24:00 of day D, i.e. timestamp ``D+1 00:00``. Sorting the raw labels
would move each day's last measurement to the start of the same day (a 1-day leak).

See docs/data_dictionary.md and the EDA notebook for the evidence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from steelops.data.load import load_raw

INTERVAL = pd.Timedelta("15min")


class CorrectionError(ValueError):
    """Raised when the data does not match the pattern a correction assumes."""


def fix_midnight_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Shift each ``D 00:00`` record to ``D+1 00:00``, after validating the pattern.

    Expects the frame in original file order. Returns a corrected copy whose
    timestamps are strictly increasing on a regular 15-minute grid.
    """
    out = df.reset_index(drop=True).copy()
    ts = out["timestamp"]
    is_midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0)
    positions = is_midnight[is_midnight].index

    # Validate the assumption: every midnight row directly follows 23:45 of the same date.
    if (positions == 0).any():
        raise CorrectionError("A midnight record is the first row; pattern not as expected.")
    previous = ts.loc[positions - 1].to_numpy()
    expected_previous = (ts.loc[positions].dt.normalize() + pd.Timedelta("23:45:00")).to_numpy()
    if not (previous == expected_previous).all():
        raise CorrectionError("Midnight records do not all follow 23:45 of the same date.")

    out.loc[is_midnight, "timestamp"] = ts[is_midnight] + pd.Timedelta("1D")

    diffs = out["timestamp"].diff().dropna()
    if not (diffs == INTERVAL).all():
        raise CorrectionError("Corrected timestamps are not a regular 15-minute sequence.")
    return out


def add_interval_start(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``interval_start`` (= timestamp - 15 min). Calendar labels follow this time."""
    return df.assign(interval_start=df["timestamp"] - INTERVAL)


def load_clean(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Raw data with documented corrections applied, in chronological order."""
    return add_interval_start(fix_midnight_labels(load_raw(config)))
