"""Reproducible acquisition of the official UCI dataset.

Design principles
-----------------
* The raw layer is immutable: files are written once, checksummed, and made read-only.
  Re-running the pipeline never overwrites an existing raw file; it only verifies it.
* Provenance is recorded in ``data/raw/MANIFEST.json`` (source URL, retrieval time,
  checksums, sizes), so any downstream result can be traced to an exact input file.
* Two acquisition modes share the same verification path:
  1. download from the official UCI URL (default);
  2. ingest a locally supplied copy of the official zip (``--from-zip``), for
     environments where the UCI host is not reachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from steelops.config import DEFAULT_DATA_CONFIG, load_config, resolve_path

ZIP_FILENAME = "steel_industry_energy_consumption.zip"
MANIFEST_FILENAME = "MANIFEST.json"
_CHUNK = 1 << 16


class ChecksumMismatchError(RuntimeError):
    """Raised when a file does not match its pinned or recorded checksum."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _verify(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_of(path)
    if expected and actual != expected:
        raise ChecksumMismatchError(
            f"{label}: SHA-256 mismatch for {path.name}\n  expected {expected}\n  actual   {actual}"
        )
    return actual


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "steelops-ai-data-acquisition"})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        destination.open("wb") as fh,
    ):
        shutil.copyfileobj(response, fh, _CHUNK)


def _extract_member(zip_path: Path, member: str, destination: Path) -> None:
    """Extract a single named member to a fixed destination (no path traversal possible)."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if member not in names:
            raise FileNotFoundError(f"{member!r} not found in archive. Members: {names}")
        with archive.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, _CHUNK)


def acquire(
    config: dict[str, Any],
    source_zip: Path | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Any]:
    """Acquire, verify, and register the raw dataset. Returns the manifest.

    Parameters
    ----------
    config:
        Parsed ``configs/data.yaml``.
    source_zip:
        Optional path to a locally supplied copy of the official zip. When omitted,
        the zip is downloaded from ``config["source"]["download_url"]``.
    raw_dir:
        Override for the raw data directory (used in tests).
    """
    source = config["source"]
    expected = source.get("expected_sha256") or {}
    raw_dir = raw_dir or resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / ZIP_FILENAME
    csv_path = raw_dir / source["csv_member"]
    manifest_path = raw_dir / MANIFEST_FILENAME

    # Idempotent path: raw files already exist -> verify only, never overwrite.
    if zip_path.exists() and csv_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify(zip_path, expected.get("zip") or manifest["files"]["zip"]["sha256"], "zip")
        _verify(csv_path, expected.get("csv") or manifest["files"]["csv"]["sha256"], "csv")
        return manifest
    if manifest_path.exists() and not (zip_path.exists() or csv_path.exists()):
        # Fresh clone: the manifest is versioned in Git, the data files are not.
        # Re-acquire and verify against the recorded checksums.
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        expected = {
            "zip": expected.get("zip") or recorded["zip"]["sha256"],
            "csv": expected.get("csv") or recorded["csv"]["sha256"],
        }
        manifest_path.chmod(0o644)
        manifest_path.unlink()
    elif zip_path.exists() or csv_path.exists():
        raise RuntimeError(
            f"Partial raw state in {raw_dir} (files present without a manifest). "
            "Raw data is immutable: inspect and remove these files manually before re-acquiring."
        )

    with tempfile.TemporaryDirectory() as tmp:
        staged_zip = Path(tmp) / ZIP_FILENAME
        if source_zip is None:
            _download(source["download_url"], staged_zip)
            acquisition_method = "download"
        else:
            shutil.copyfile(source_zip, staged_zip)
            acquisition_method = "local_copy_of_official_zip"

        zip_sha = _verify(staged_zip, expected.get("zip"), "zip")
        staged_csv = Path(tmp) / source["csv_member"]
        _extract_member(staged_zip, source["csv_member"], staged_csv)
        csv_sha = _verify(staged_csv, expected.get("csv"), "csv")

        shutil.move(staged_zip, zip_path)
        shutil.move(staged_csv, csv_path)

    for path in (zip_path, csv_path):
        _make_read_only(path)

    manifest = {
        "dataset": source["name"],
        "provider": source["provider"],
        "dataset_id": source["dataset_id"],
        "landing_page": source["landing_page"],
        "download_url": source["download_url"],
        "doi": source["doi"],
        "license": source["license"],
        "acquisition_method": acquisition_method,
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {
            "zip": {"name": zip_path.name, "sha256": zip_sha, "bytes": zip_path.stat().st_size},
            "csv": {"name": csv_path.name, "sha256": csv_sha, "bytes": csv_path.stat().st_size},
        },
        "checksums_pinned_in_config": bool(expected.get("zip") and expected.get("csv")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _make_read_only(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire the official UCI dataset #851.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--from-zip",
        type=Path,
        default=None,
        help="Ingest a locally downloaded copy of the official zip instead of downloading.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    manifest = acquire(config, source_zip=args.from_zip)
    print(json.dumps(manifest, indent=2))
    pinned = config["source"].get("expected_sha256") or {}
    if not (pinned.get("zip") and pinned.get("csv")):
        print(
            "\nNOTE: checksums are not pinned in configs/data.yaml. After reviewing this "
            "first acquisition, pin the values above under source.expected_sha256.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
