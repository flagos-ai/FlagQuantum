"""Wheel response reuse must not freeze index or version-discovery metadata."""

import zipfile
from pathlib import Path

import pytest

from tools.ci_pip_wheel_cache import seed_wheel_responses

pytestmark = pytest.mark.unit


def _wheel(source: Path, name: str, *, metadata: bool = True) -> Path:
    path = source / "http-v2/a/b" / f"{name}.body"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("example-1.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    if metadata:
        path.with_suffix("").write_bytes(b"opaque pip HTTP response metadata")
    return path


def test_reuses_complete_wheels_without_copying_index_responses(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "private cache"
    wheel = _wheel(source, "wheel")
    index = source / "http-v2/a/b/index.body"
    index.write_text('{"versions": ["1.0"]}')
    index.with_suffix("").write_bytes(b"index HTTP metadata")
    assert seed_wheel_responses(source, destination) == 1
    target = destination / wheel.relative_to(source)
    assert target.read_bytes() == wheel.read_bytes()
    assert target.with_suffix("").read_bytes() == wheel.with_suffix("").read_bytes()
    assert not (destination / index.relative_to(source)).exists()
    target.write_bytes(b"incomplete previous download")
    assert seed_wheel_responses(source, destination) == 1
    assert target.read_bytes() == wheel.read_bytes()
    assert not list(destination.rglob(".fq-wheel-*"))


def test_missing_response_metadata_and_non_wheel_archives_are_ignored(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _wheel(source, "incomplete", metadata=False)
    archive = source / "http-v2/a/b/source.body"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("README", "source archive")
    archive.with_suffix("").write_bytes(b"HTTP metadata")
    assert seed_wheel_responses(source, tmp_path / "destination") == 0


def test_a_corrupt_cached_wheel_is_not_published(tmp_path: Path) -> None:
    source = tmp_path / "source"
    body = _wheel(source, "broken")
    body.write_bytes(b"PK\x03\x04truncated archive")
    destination = tmp_path / "destination"
    with pytest.raises(zipfile.BadZipFile):
        seed_wheel_responses(source, destination)
    assert not destination.exists()
