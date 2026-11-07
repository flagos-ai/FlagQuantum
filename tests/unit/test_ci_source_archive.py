"""API source restoration preserves Git identity and refuses altered archives."""

from __future__ import annotations

import io
import subprocess
import tarfile
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from urllib.request import Request

import pytest

from tools.ci_source_archive import _Commit, _Redirect, _restore, _verified_commit

pytestmark = pytest.mark.unit


def _git(directory: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=directory,
        input=data,
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout


@dataclass(frozen=True)
class _Source:
    archive: Path
    commit: _Commit
    metadata: dict[str, object]


@pytest.fixture(params=[True, False], ids=["github-signature", "git-signature"])
def source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> _Source:
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    directory = tmp_path / "source"
    directory.mkdir()
    _git(directory, "init", "--quiet")
    _git(directory, "config", "user.name", "FlagQuantum")
    _git(directory, "config", "user.email", "liuwei.chem.phys@gmail.com")
    (directory / "tracked.txt").write_text("source\n")
    (directory / "nested").mkdir()
    (directory / "nested/source.txt").write_text("nested source\n")
    (directory / ".gitignore").write_text("ignored.data\n")
    (directory / "ignored.data").write_bytes(bytes(range(256)))
    (directory / "link").symlink_to("tracked.txt")
    executable = directory / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    _git(directory, "add", "--force", "--all", ".")
    _git(directory, "commit", "--quiet", "-m", "source")
    payload = _git(directory, "cat-file", "commit", "HEAD").decode()
    tree = _git(directory, "rev-parse", "HEAD^{tree}").decode().strip()
    signature = (
        "-----BEGIN PGP SIGNATURE-----\n\nfixture\n-----END PGP SIGNATURE-----\n"
    )
    serialized = signature if request.param else signature.rstrip("\n")
    headers, message = payload.split("\n\n", 1)
    signature_header = serialized.replace("\n", "\n ")
    raw = f"{headers}\ngpgsig {signature_header}\n\n{message}".encode()
    sha = (
        _git(directory, "hash-object", "-t", "commit", "-w", "--stdin", data=raw)
        .decode()
        .strip()
    )
    archive = tmp_path / "source.tar"
    _git(
        directory,
        "archive",
        "--format=tar",
        "--prefix=snapshot/",
        "-o",
        str(archive),
        sha,
    )
    metadata = {
        "sha": sha,
        "tree": {"sha": tree},
        "verification": {"payload": payload, "signature": signature},
    }
    return _Source(archive, _Commit(sha, tree, raw), metadata)


def test_signed_metadata_reproduces_the_original_commit(source: _Source) -> None:
    assert _verified_commit(source.metadata, source.commit.sha) == source.commit


def test_changed_signed_payload_is_refused(source: _Source) -> None:
    verification = source.metadata["verification"]
    assert isinstance(verification, dict)
    metadata = {
        **source.metadata,
        "verification": {
            **verification,
            "payload": str(verification["payload"]) + "tampered",
        },
    }
    with pytest.raises(ValueError, match="cannot reproduce"):
        _verified_commit(metadata, source.commit.sha)


def test_archive_restoration_handles_cold_and_dirty_cached_trees(
    tmp_path: Path, source: _Source
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert _restore(workspace, source.archive, source.commit) is None
    (workspace / "tracked.txt").write_text("dirty\n")
    (workspace / "stale.txt").write_text("stale\n")
    assert _restore(workspace, source.archive, source.commit) == source.commit.sha
    assert _git(workspace, "rev-parse", "HEAD").decode().strip() == source.commit.sha
    assert _git(workspace, "status", "--porcelain") == b""
    assert (workspace / "tracked.txt").read_text() == "source\n"
    assert (workspace / "ignored.data").read_bytes() == bytes(range(256))
    assert (workspace / "link").is_symlink()
    assert (workspace / "run.sh").stat().st_mode & 0o111 == 0o111
    assert not (workspace / "stale.txt").exists()


def test_an_archive_with_a_different_tree_is_refused(
    tmp_path: Path, source: _Source
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invalid = _Commit(source.commit.sha, "0" * 40, source.commit.raw)
    with pytest.raises(ValueError, match="does not match.*Git tree"):
        _restore(workspace, source.archive, invalid)
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=workspace,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("name", ["snapshot/.git/config", "snapshot/../escape"])
def test_archive_paths_cannot_overwrite_git_metadata_or_escape(
    tmp_path: Path, source: _Source, name: str
) -> None:
    archive = tmp_path / "invalid.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo(name)
        member.size = 4
        output.addfile(member, io.BytesIO(b"test"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "retained.txt"
    marker.write_text("retained")
    with pytest.raises(ValueError):
        _restore(workspace, archive, source.commit)
    assert marker.read_text() == "retained"
    assert not (tmp_path / "escape").exists()


def test_redirects_do_not_forward_the_api_token_to_another_host() -> None:
    request = Request(
        "https://api.github.com/archive", headers={"Authorization": "Bearer fixture"}
    )
    redirected = _Redirect().redirect_request(
        request,
        io.BytesIO(),
        302,
        "Found",
        Message(),
        "https://codeload.github.com/archive",
    )
    assert redirected is not None
    assert not redirected.has_header("Authorization")


def test_api_objects_restore_exact_source_and_reuse_cache(
    tmp_path: Path, source: _Source
) -> None:
    import base64

    from tools.ci_source_archive import _checkout, _materialize_api

    directory = source.archive.parent / "source"
    calls = []

    def get_json(path: str) -> object:
        calls.append(path)
        sha = path.rsplit("/", 1)[1]
        if path.startswith("/git/trees/"):
            entries = []
            for record in _git(directory, "ls-tree", "-z", sha).split(b"\0"):
                if not record:
                    continue
                header, filename = record.decode().split("\t", 1)
                mode, kind, child_sha = header.split()
                entries.append(
                    {"mode": mode, "type": kind, "sha": child_sha, "path": filename}
                )
            return {"sha": sha, "truncated": False, "tree": entries}
        raw = _git(directory, "cat-file", "blob", sha)
        return {
            "sha": sha,
            "encoding": "base64",
            "content": base64.b64encode(raw).decode(),
        }

    workspace = tmp_path / "api-workspace"
    workspace.mkdir()
    assert _materialize_api(workspace, source.commit, get_json)
    _checkout(workspace, source.commit, None)
    assert _git(workspace, "status", "--porcelain") == b""
    assert (workspace / "ignored.data").read_bytes() == bytes(range(256))
    assert (workspace / "link").is_symlink()
    calls.clear()
    assert _materialize_api(workspace, source.commit, get_json)
    assert calls == []
    blob = _git(directory, "rev-parse", "HEAD:tracked.txt").decode().strip()
    (workspace / ".git/objects" / blob[:2] / blob[2:]).unlink()
    assert _materialize_api(workspace, source.commit, get_json)
    assert calls == [f"/git/blobs/{blob}"]


def test_api_truncated_tree_requires_archive(tmp_path: Path, source: _Source) -> None:
    from tools.ci_source_archive import _materialize_api

    assert not _materialize_api(
        tmp_path,
        source.commit,
        lambda _: {"sha": source.commit.tree, "truncated": True},
    )


@pytest.mark.parametrize("removal", ["success", "auto-removal", "persistent"])
@pytest.mark.parametrize("imported_image", [False, True])
def test_cleanup_only_removes_previous_container_for_this_runner_and_workspace(
    tmp_path, monkeypatch, removal, imported_image
):
    import json

    from tools import ci_source_archive as archive

    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    monkeypatch.setenv("RUNNER_NAME", "cpu01")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(archive.shutil, "which", lambda _: "/usr/bin/docker")
    image = "sha256:" + "f" * 64
    if not imported_image:
        image = "ghcr.io/flagos-ai/flagquantum-ci@" + image
    ids = [letter * 64 for letter in "abcd"]

    def container(identifier, runner, workspace):
        return {
            "Id": identifier,
            "Config": {
                "Env": [
                    f"RUNNER_NAME={runner}",
                    "FQ_CI_PRIVATE=/tmp/fq-offline.fixture",
                ]
            },
            "Mounts": [{"Type": "bind", "Source": str(workspace)}],
        }

    metadata = [
        container(ids[0], "cpu01", tmp_path),
        container(ids[1], "cpu02", tmp_path),
        container(ids[2], "cpu01", tmp_path / "other"),
    ]
    calls = []

    def command(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["timeout"] == 20
        if arguments[1] == "inspect":
            return subprocess.CompletedProcess(arguments, 0, json.dumps(metadata), "")
        if arguments[1] == "rm":
            assert arguments[2:] == ["--force", ids[0]]
            return subprocess.CompletedProcess(
                arguments, int(removal != "success"), "", "removal already in progress"
            )
        if "{{.ID}} {{.Image}}" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                "\n".join(f"{i} {image}" for i in ids[:3]) + f"\n{ids[3]} redis:latest",
                "",
            )
        return subprocess.CompletedProcess(
            arguments, 0, ids[0] if removal == "persistent" else "", ""
        )

    monkeypatch.setattr(archive.subprocess, "run", command)
    monkeypatch.setattr(archive.time, "sleep", lambda _: None)
    if removal == "persistent":
        with pytest.raises(
            RuntimeError, match="Could not remove previous CI containers"
        ):
            archive.cleanup_runner_containers()
    else:
        archive.cleanup_runner_containers()
    assert not any(ids[3] in command for command in calls)
    assert sum(command[1] == "rm" for command in calls) == (
        2 if removal == "persistent" else 1
    )


def test_cleanup_does_not_access_docker_on_hosted_or_non_docker_runners(monkeypatch):
    from tools import ci_source_archive as archive

    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setattr(
        archive.subprocess, "run", lambda *a, **k: pytest.fail("Docker accessed")
    )
    archive.cleanup_runner_containers()
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    monkeypatch.setattr(archive.shutil, "which", lambda _: None)
    archive.cleanup_runner_containers()


@pytest.mark.parametrize("tamper", [False, True])
def test_large_source_addition_uses_bounded_parallel_verified_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: bool
) -> None:
    import base64
    from threading import Barrier, Lock

    from tools.ci_source_archive import _checkout, _materialize_api

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    directory = tmp_path / "source"
    directory.mkdir()
    _git(directory, "init", "--quiet")
    for index in range(96):
        (directory / f"example-{index:03d}.txt").write_text(
            f"original source {index}\n"
        )
    _git(directory, "add", ".")
    tree = _git(directory, "write-tree").decode().strip()
    raw = f"tree {tree}\nauthor Test <test@example.org> 1 +0000\ncommitter Test <test@example.org> 1 +0000\n\nsource\n".encode()
    sha = (
        _git(directory, "hash-object", "-t", "commit", "-w", "--stdin", data=raw)
        .decode()
        .strip()
    )
    objects: dict[str, object] = {}
    entries = []
    for record in _git(directory, "ls-tree", "-z", tree).split(b"\0"):
        if not record:
            continue
        header, filename = record.decode().split("\t", 1)
        mode, kind, child_sha = header.split()
        entries.append({"mode": mode, "type": kind, "sha": child_sha, "path": filename})
        content = (
            b"altered" if tamper else _git(directory, "cat-file", "blob", child_sha)
        )
        objects[f"/git/blobs/{child_sha}"] = {
            "sha": child_sha,
            "encoding": "base64",
            "content": base64.b64encode(content).decode(),
        }
    objects[f"/git/trees/{tree}"] = {"sha": tree, "tree": entries}
    barrier = Barrier(8, timeout=10)
    lock = Lock()
    calls: list[str] = []

    def get_json(path: str) -> object:
        with lock:
            calls.append(path)
        if path.startswith("/git/blobs/"):
            barrier.wait()
        return objects[path]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commit = _Commit(sha, tree, raw)
    if tamper:
        with pytest.raises(ValueError, match="does not match its SHA"):
            _materialize_api(workspace, commit, get_json)
        with pytest.raises(subprocess.CalledProcessError):
            _git(workspace, "rev-parse", "--verify", "HEAD")
        return
    assert _materialize_api(workspace, commit, get_json)
    assert len(calls) == 97
    _checkout(workspace, commit, None)
    assert _git(workspace, "write-tree").decode().strip() == tree
    assert _git(workspace, "status", "--porcelain") == b""
    calls.clear()
    assert _materialize_api(workspace, commit, get_json)
    assert calls == []
