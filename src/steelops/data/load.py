"""Load the raw CSV into a typed DataFrame with canonical column names.

This module performs *no cleaning*. It parses, renames, and type-casts only, so the
returned frame is a faithful, inspectable representation of the immutable raw file.
Any correction (e.g. gap handling) belongs in a later, explicit, documented step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from steelops.config import load_config, resolve_path

NUMERIC_COLUMNS = (
    "usage_kwh",
    "lagging_reactive_kvarh",
    "leading_reactive_kvarh",
    "co2_tco2",
    "lagging_power_factor",
    "leading_power_factor",
    "nsm",
)
CATEGORICAL_COLUMNS = ("week_status", "day_of_week", "load_type")


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected schema."""


def raw_csv_path(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    return resolve_path(config["paths"]["raw_dir"]) / config["source"]["csv_member"]


def read_raw_frame(path: Path) -> pd.DataFrame:
    """Read the CSV exactly as stored, every column as string. Used by the audit."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_columns(raw: pd.DataFrame, column_map: dict[str, str]) -> None:
    expected = list(column_map)
    missing = [c for c in expected if c not in raw.columns]
    unexpected = [c for c in raw.columns if c not in column_map]
    if missing or unexpected:
        raise SchemaError(f"Column mismatch. Missing: {missing}. Unexpected: {unexpected}.")


def parse_frame(raw: pd.DataFrame, schema: dict[str, Any], strict: bool = True) -> pd.DataFrame:
    """Rename columns, parse timestamps with an explicit format, and cast types.

    With ``strict=True`` any unparseable timestamp or numeric value raises. With
    ``strict=False`` such values become NaT/NaN so the audit can count them.
    """
    column_map: dict[str, str] = schema["columns"]
    check_columns(raw, column_map)
    df = raw.rename(columns=column_map)

    errors = "raise" if strict else "coerce"
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=schema["timestamp_format"], errors=errors
    )
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors=errors)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")
    return df


def load_raw(config: dict[str, Any] | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset with canonical names and parsed types (strict)."""
    config = config or load_config()
    path = path or raw_csv_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `uv run steelops-download` first."
        )
    return parse_frame(read_raw_frame(path), config["schema"], strict=True)
