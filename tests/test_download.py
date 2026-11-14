"""Tests for raw-data acquisition: checksums, immutability, idempotency."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from steelops.data.download import (
    MANIFEST_FILENAME,
    ZIP_FILENAME,
    ChecksumMismatchError,
    acquire,
    sha256_of,
)

CSV_MEMBER = "Steel_industry_data.csv"


def _config(expected_zip: str | None = None, expected_csv: str | None = None) -> dict:
    return {
        "source": {
            "name": "test",
            "provider": "test",
            "dataset_id": 851,
            "landing_page": "https://example.org",
            "download_url": "https://example.org/never-called.zip",
            "doi": "x",
            "license": "CC BY 4.0",
            "csv_member": CSV_MEMBER,
            "expected_sha256": {"zip": expected_zip, "csv": expected_csv},
        },
        "paths": {"raw_dir": "unused"},
    }


@pytest.fixture()
def official_zip(tmp_path: Path) -> Path:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(CSV_MEMBER, "date,Usage_kWh\n01/01/2018 00:15,3.17\n")
    return path


def test_acquire_from_local_zip_writes_manifest_and_read_only_files(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    manifest = acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)

    assert manifest["acquisition_method"] == "local_copy_of_official_zip"
    assert manifest["files"]["zip"]["sha256"] == sha256_of(official_zip)
    for name in (ZIP_FILENAME, CSV_MEMBER, MANIFEST_FILENAME):
        path = raw_dir / name
        assert path.exists()
        assert not os.access(path, os.W_OK) or os.geteuid() == 0  # root ignores mode bits
        assert not (path.stat().st_mode & 0o222), f"{name} should be read-only"
    on_disk = json.loads((raw_dir / MANIFEST_FILENAME).read_text())
    assert on_disk["files"]["csv"]["sha256"] == sha256_of(raw_dir / CSV_MEMBER)


def test_acquire_is_idempotent_and_never_overwrites(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    first = acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    mtime = (raw_dir / CSV_MEMBER).stat().st_mtime_ns
    second = acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    assert first == second
    assert (raw_dir / CSV_MEMBER).stat().st_mtime_ns == mtime


def test_pinned_checksum_mismatch_aborts_before_writing(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    with pytest.raises(ChecksumMismatchError):
        acquire(_config(expected_zip="0" * 64), source_zip=official_zip, raw_dir=raw_dir)
    assert not any(raw_dir.iterdir())


def test_tampered_raw_csv_is_detected(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    csv = raw_dir / CSV_MEMBER
    csv.chmod(0o644)
    csv.write_text("tampered\n")
    with pytest.raises(ChecksumMismatchError):
        acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)


def test_partial_raw_state_is_refused(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / CSV_MEMBER).write_text("orphan\n")
    with pytest.raises(RuntimeError, match="Partial raw state"):
        acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)


def test_fresh_clone_reacquires_and_verifies_against_manifest(tmp_path, official_zip):
    """Manifest is in Git, data files are not: re-acquire and check recorded checksums."""
    raw_dir = tmp_path / "raw"
    first = acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    for name in (ZIP_FILENAME, CSV_MEMBER):
        (raw_dir / name).chmod(0o644)
        (raw_dir / name).unlink()
    second = acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    assert second["files"] == first["files"]


def test_fresh_clone_rejects_different_file(tmp_path, official_zip):
    raw_dir = tmp_path / "raw"
    acquire(_config(), source_zip=official_zip, raw_dir=raw_dir)
    for name in (ZIP_FILENAME, CSV_MEMBER):
        (raw_dir / name).chmod(0o644)
        (raw_dir / name).unlink()
    other_zip = tmp_path / "other.zip"
    with zipfile.ZipFile(other_zip, "w") as zf:
        zf.writestr(CSV_MEMBER, "different content\n")
    with pytest.raises(ChecksumMismatchError):
        acquire(_config(), source_zip=other_zip, raw_dir=raw_dir)


def test_missing_member_in_archive_raises(tmp_path):
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("other.csv", "x\n")
    with pytest.raises(FileNotFoundError, match=CSV_MEMBER):
        acquire(_config(), source_zip=bad_zip, raw_dir=tmp_path / "raw")
