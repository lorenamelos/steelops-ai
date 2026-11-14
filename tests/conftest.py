"""Shared fixtures: a small synthetic raw frame that mimics the official CSV layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steelops.config import load_config

RAW_COLUMNS = [
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
]


@pytest.fixture(scope="session")
def schema() -> dict:
    return load_config()["schema"]


def make_synthetic_raw(n: int = 8 * 96, start: str = "2018-01-01 00:15") -> pd.DataFrame:
    """Build a clean synthetic raw frame (all values as strings, like read_raw_frame).

    Timestamps label the END of each 15-minute interval; calendar labels follow the
    interval start. Load_Type is a deterministic function of time of day.
    """
    rng = np.random.default_rng(42)
    ts = pd.date_range(start, periods=n, freq="15min")
    usage = rng.uniform(5, 100, n).round(2)
    lag_kvarh = rng.uniform(0, 40, n).round(2)
    lead_kvarh = np.zeros(n)
    lag_pf = (100 * usage / np.sqrt(usage**2 + lag_kvarh**2)).round(2)
    lead_pf = np.full(n, 100.0)
    start_of_interval = ts - pd.Timedelta("15min")
    day = start_of_interval.day_name()
    hour = ts.hour
    load = np.where(hour < 8, "Light_Load", np.where(hour < 18, "Maximum_Load", "Medium_Load"))
    return pd.DataFrame(
        {
            "date": ts.strftime("%d/%m/%Y %H:%M"),
            "Usage_kWh": usage.astype(str),
            "Lagging_Current_Reactive.Power_kVarh": lag_kvarh.astype(str),
            "Leading_Current_Reactive_Power_kVarh": lead_kvarh.astype(str),
            "CO2(tCO2)": (usage * 0.0005).round(2).astype(str),
            "Lagging_Current_Power_Factor": lag_pf.astype(str),
            "Leading_Current_Power_Factor": lead_pf.astype(str),
            "NSM": (ts.hour * 3600 + ts.minute * 60).astype(str),
            "WeekStatus": np.where(day.isin(["Saturday", "Sunday"]), "Weekend", "Weekday"),
            "Day_of_week": day,
            "Load_Type": load,
        },
        columns=RAW_COLUMNS,
    )


@pytest.fixture()
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_raw()


@pytest.fixture()
def synthetic_raw_factory():
    return make_synthetic_raw
