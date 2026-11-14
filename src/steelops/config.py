"""Configuration loading and repository path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: Path | str = DEFAULT_DATA_CONFIG) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path, root: Path = REPO_ROOT) -> Path:
    """Resolve a config path relative to the repository root."""
    relative = Path(relative)
    return relative if relative.is_absolute() else root / relative
