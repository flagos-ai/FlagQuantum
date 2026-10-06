from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qdiffusion_kaiwu import private_io as private_io_module
from examples.qdiffusion_kaiwu.private_io import (
    open_private_binary,
    validate_private_directory,
    validate_private_json_output_path,
    write_private_json_exclusive,
)

pytestmark = pytest.mark.unit


def test_validate_private_directory_accepts_absolute_private_directory(
    tmp_path: Path,
) -> None:
    validate_private_directory(tmp_path, label="checkpoint directory")


def test_validate_private_directory_rejects_foreign_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(private_io_module.os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(ValueError, match="checkpoint directory must"):
        validate_private_directory(tmp_path, label="checkpoint directory")


def _with_uid(metadata: os.stat_result, uid: int) -> SimpleNamespace:
    fields = {
        name: getattr(metadata, name)
        for name in dir(metadata)
        if name.startswith("st_")
    }
    fields["st_uid"] = uid
    return SimpleNamespace(**fields)


def test_open_private_binary_rejects_foreign_owned_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"reviewed")
    path.chmod(0o600)
    real_fstat = private_io_module.os.fstat

    def foreign_leaf(descriptor: int):
        metadata = real_fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            return _with_uid(metadata, os.geteuid() + 1)
        return metadata

    monkeypatch.setattr(private_io_module.os, "fstat", foreign_leaf)

    with pytest.raises(ValueError, match="owned by the current effective user"):
        with open_private_binary(path, label="artifact", max_bytes=64):
            pytest.fail("foreign-owned evidence must fail before consumption")


def test_open_private_binary_rechecks_leaf_owner_after_consumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"reviewed")
    path.chmod(0o600)
    real_fstat = private_io_module.os.fstat
    regular_checks = 0

    def owner_changes(descriptor: int):
        nonlocal regular_checks
        metadata = real_fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            regular_checks += 1
            if regular_checks == 2:
                return _with_uid(metadata, os.geteuid() + 1)
        return metadata

    monkeypatch.setattr(private_io_module.os, "fstat", owner_changes)

    with (
        pytest.raises(ValueError, match="changed during validation"),
        open_private_binary(path, label="artifact", max_bytes=64) as stream,
    ):
        assert stream.read() == b"reviewed"


def test_open_private_binary_rechecks_leaf_binding_after_consumption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"reviewed")
    path.chmod(0o600)
    moved = tmp_path / "opened-artifact.bin"

    with (
        pytest.raises(ValueError, match="changed during validation|binding changed"),
        open_private_binary(path, label="artifact", max_bytes=64) as stream,
    ):
        assert stream.read() == b"reviewed"
        path.rename(moved)
        path.write_bytes(b"replacement")
        path.chmod(0o600)


def test_validate_private_directory_rejects_unsafe_directory(
    tmp_path: Path,
) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(private, target_is_directory=True)

    unsafe = (
        Path("relative"),
        tmp_path / "missing",
        public,
        linked,
        tmp_path / "file",
    )
    (tmp_path / "file").write_text("not a directory", encoding="utf-8")
    for path in unsafe:
        with pytest.raises(ValueError, match="checkpoint directory must"):
            validate_private_directory(path, label="checkpoint directory")


def test_validate_private_json_output_path_accepts_absent_private_target(
    tmp_path: Path,
) -> None:
    output = tmp_path / "record.json"

    validate_private_json_output_path(output)

    assert not output.exists()


@pytest.mark.parametrize("kind", ("file", "directory", "dangling_symlink"))
def test_validate_private_json_output_path_rejects_existing_target(
    tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "record.json"
    if kind == "file":
        output.write_text("existing", encoding="utf-8")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError, match="already exists"):
        validate_private_json_output_path(output)


def test_validate_private_json_output_path_rejects_unsafe_parent(
    tmp_path: Path,
) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o755)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(private_parent, target_is_directory=True)

    unsafe_paths = (
        Path("relative.json"),
        tmp_path / "missing" / "record.json",
        public_parent / "record.json",
        linked_parent / "record.json",
    )
    for output in unsafe_paths:
        with pytest.raises(ValueError):
            validate_private_json_output_path(output)


def test_writer_rechecks_destination_and_never_overwrites_raced_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "record.json"
    validate_private_json_output_path(output)
    output.write_text("raced", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_private_json_exclusive(output, {"accepted": False})

    assert output.read_text(encoding="utf-8") == "raced"


def test_writer_publishes_valid_json_exclusively(tmp_path: Path) -> None:
    output = tmp_path / "record.json"

    write_private_json_exclusive(output, {"accepted": False})

    assert json.loads(output.read_text(encoding="utf-8")) == {"accepted": False}
    with pytest.raises(FileExistsError):
        write_private_json_exclusive(output, {"accepted": True})


@pytest.mark.parametrize("replacement_check", (2, 3))
def test_writer_rejects_parent_replacement_without_leaving_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement_check: int,
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    moved = tmp_path / "moved-private"
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    output = parent / "record.json"
    real_verify = private_io_module._verify_open_directory_binding
    checks = 0

    def replace_then_verify(path: Path, metadata: os.stat_result) -> None:
        nonlocal checks
        checks += 1
        if checks == replacement_check:
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
        real_verify(path, metadata)

    monkeypatch.setattr(
        private_io_module, "_verify_open_directory_binding", replace_then_verify
    )

    with pytest.raises(ValueError, match="parent changed"):
        write_private_json_exclusive(output, {"accepted": False})

    assert not any(moved.iterdir())
    assert not any(replacement.iterdir())
