"""Frozen test inputs have one inventory and cannot publish changed snapshots."""

from __future__ import annotations

import csv
import io
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from tools import ftqc_source_inputs as inputs

pytestmark = pytest.mark.unit
REVISION = "a" * 40


@pytest.fixture(autouse=True)
def controlled_disk_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = shutil.disk_usage(tmp_path)._replace(
        total=8 * 1024**3, used=4 * 1024**3, free=4 * 1024**3
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage)


def archive(path: Path, rows: list[tuple[str, bytes, int]]) -> Path:
    with tarfile.open(path, "w:gz") as stream:
        for name, data, mode in rows:
            member = tarfile.TarInfo("repository/" + name)
            member.size = len(data)
            member.mode = mode
            stream.addfile(member, io.BytesIO(data))
    return path


def record(name: str = "check/run", data: bytes = b"original\n") -> inputs.SourceFile:
    return inputs.SourceFile("Q-0001", REVISION, name, inputs.git_blob(data))


def test_canonical_inventory_includes_all_unavailable_labels(tmp_path: Path) -> None:
    records = inputs.read_inventory()
    assert len(records) == 1280
    assert len({r.identifier for r in records}) == 1280
    assert {r.revision for r in records} == {
        "bcb17bb25360ed68ce3bc084645db3e929799389",
        "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453",
    }
    assert records[0].identifier == "Q-0001"
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "revision", "path", "git_blob", "source_audit"))
        writer.writerow(
            ("Q-0001", REVISION, "check/run", record().git_blob, "unavailable")
        )
    assert inputs.read_inventory(inventory) == (record(),)


@pytest.mark.parametrize(
    "path", ("../escape", "/escape", ".", "dir//file", "C:escape", r"dir\escape")
)
def test_inventory_rejects_nonportable_or_escaping_paths(path: str) -> None:
    with pytest.raises(ValueError, match="source path"):
        record(path)


def test_prepare_two_revisions_and_preserve_executable_mode(tmp_path: Path) -> None:
    second = inputs.SourceFile(
        "K-0001", "b" * 40, "kickmix/data", inputs.git_blob(b"second")
    )
    archives = {
        REVISION: archive(tmp_path / "a.tar.gz", [("check/run", b"original\n", 0o755)]),
        second.revision: archive(
            tmp_path / "b.tar.gz", [(second.path, b"second", 0o644)]
        ),
    }
    destination = tmp_path / "snapshot"
    inputs.prepare_sources(archives, destination, (record(), second))
    inputs.verify_sources(destination, (record(), second))
    if os.name != "nt":
        assert (destination / "check/run").stat().st_mode & 0o777 == 0o755
    sentinel = destination / "user-note"
    sentinel.write_text("keep")
    inputs.prepare_sources({}, destination, (record(), second))
    assert sentinel.read_text() == "keep"


def test_changed_existing_source_is_not_repaired(tmp_path: Path) -> None:
    target = tmp_path / "snapshot/check/run"
    target.parent.mkdir(parents=True)
    target.write_text("user change")
    with pytest.raises(ValueError, match="Missing or changed"):
        inputs.prepare_sources({}, tmp_path / "snapshot", (record(),))
    assert target.read_text() == "user change"


@pytest.mark.parametrize("contents", (b"changed", b""))
def test_bad_archive_never_publishes_a_partial_snapshot(
    tmp_path: Path, contents: bytes
) -> None:
    source = archive(tmp_path / "bad.tar.gz", [("check/run", contents, 0o755)])
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Changed frozen"):
        inputs.prepare_sources({REVISION: source}, destination, (record(),))
    assert not destination.exists()
    assert not list(tmp_path.glob(".fq-sources-*"))


def test_missing_and_duplicate_archive_members_refuse(tmp_path: Path) -> None:
    source = archive(tmp_path / "missing.tar.gz", [("other", b"other", 0o644)])
    with pytest.raises(ValueError, match="Missing frozen"):
        inputs.prepare_sources({REVISION: source}, tmp_path / "missing", (record(),))
    source = archive(
        tmp_path / "duplicate.tar.gz", [("check/run", b"original\n", 0o644)] * 2
    )
    with pytest.raises(ValueError, match="duplicate members"):
        inputs.prepare_sources({REVISION: source}, tmp_path / "duplicate", (record(),))


def test_archive_cannot_replace_a_source_with_a_symlink(tmp_path: Path) -> None:
    source = tmp_path / "link.tar.gz"
    with tarfile.open(source, "w:gz") as stream:
        member = tarfile.TarInfo("repository/check/run")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        stream.addfile(member)
    with pytest.raises(ValueError, match="regular archive files"):
        inputs.prepare_sources({REVISION: source}, tmp_path / "snapshot", (record(),))


def test_new_snapshot_is_rolled_back_on_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = archive(tmp_path / "source.tar.gz", [("check/run", b"original\n", 0o644)])

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(shutil, "copytree", fail)
    with pytest.raises(OSError, match="copy failed"):
        inputs.prepare_sources({REVISION: source}, tmp_path / "snapshot", (record(),))
    assert not (tmp_path / "snapshot").exists()


def test_download_is_bounded_cached_and_has_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []

    def response(url: str, *, timeout: int) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(b"archive bytes")

    monkeypatch.setattr(inputs, "urlopen", response)
    path = inputs.download_archive(REVISION, tmp_path / "cache")
    assert path.read_bytes() == b"archive bytes"
    assert calls == [
        (f"https://codeload.github.com/quantumlib/Qualtran/tar.gz/{REVISION}", 60)
    ]
    assert inputs.download_archive(REVISION, path.parent) == path
    assert len(calls) == 1
    monkeypatch.setattr(inputs, "_MAX_ARCHIVE_BYTES", 2)
    with pytest.raises(ValueError, match="download budget"):
        inputs.download_archive("c" * 40, path.parent)
    assert list(path.parent.iterdir()) == [path]


def test_existing_tree_created_during_preparation_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = archive(tmp_path / "source.tar.gz", [("check/run", b"original\n", 0o644)])
    destination = tmp_path / "snapshot"
    original_verify = inputs.verify_sources

    def concurrent(
        root: Path, records: tuple[inputs.SourceFile, ...] | list[inputs.SourceFile]
    ) -> None:
        original_verify(root, records)
        destination.mkdir()
        (destination / "user-note").write_text("keep")

    monkeypatch.setattr(inputs, "verify_sources", concurrent)
    with pytest.raises(FileExistsError):
        inputs.prepare_sources({REVISION: source}, destination, (record(),))
    assert (destination / "user-note").read_text() == "keep"


def test_low_disk_budget_refuses_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = shutil.disk_usage(tmp_path)._replace(free=0)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage)
    source = archive(tmp_path / "source.tar.gz", [("check/run", b"original\n", 0o644)])
    with pytest.raises(OSError, match="at least 1 GiB"):
        inputs.prepare_sources({REVISION: source}, tmp_path / "snapshot", (record(),))
    assert not (tmp_path / "snapshot").exists()
