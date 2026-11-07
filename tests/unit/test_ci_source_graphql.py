"""Batched GitHub source objects keep exact Git identity and bounded API use."""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import pytest

from tools import ci_source_archive as source

pytestmark = pytest.mark.unit


def _git(directory: Path, *args: str, data: bytes | None = None) -> str:
    return (
        subprocess.run(
            ["git", *args],
            cwd=directory,
            input=data,
            capture_output=True,
            check=True,
            timeout=20,
        )
        .stdout.decode()
        .strip()
    )


@pytest.mark.parametrize("tamper", [False, True])
def test_graphql_restores_large_source_with_few_requests_and_refuses_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: bool
) -> None:
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    directory = tmp_path / "source"
    directory.mkdir()
    _git(directory, "init", "--quiet")
    contents: dict[str, bytes] = {}
    for index in range(96):
        raw = f"source {index}\r\n".encode()
        (directory / f"example-{index:03d}.txt").write_bytes(raw)
        contents[_git(directory, "hash-object", "--stdin", data=raw)] = raw
    binary = b"\x00\xff\x01"
    (directory / "binary.dat").write_bytes(binary)
    contents[_git(directory, "hash-object", "--stdin", data=binary)] = binary
    executable = b"#!/bin/sh\nexit 0\n"
    (directory / "run.sh").write_bytes(executable)
    (directory / "run.sh").chmod(0o755)
    contents[_git(directory, "hash-object", "--stdin", data=executable)] = executable
    (directory / "link").symlink_to("example-000.txt")
    contents[_git(directory, "hash-object", "--stdin", data=b"example-000.txt")] = (
        b"example-000.txt"
    )
    _git(directory, "add", ".")
    tree = _git(directory, "write-tree")
    raw_commit = f"tree {tree}\nauthor Test <test@example.org> 1 +0000\ncommitter Test <test@example.org> 1 +0000\n\nsource\n".encode()
    sha = _git(
        directory, "hash-object", "-t", "commit", "-w", "--stdin", data=raw_commit
    )
    entries = []
    for record in subprocess.run(
        ["git", "ls-tree", "-z", tree],
        cwd=directory,
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout.split(b"\0"):
        if not record:
            continue
        header, name = record.decode().split("\t", 1)
        mode, kind, oid = header.split()
        entries.append({"mode": int(mode, 8), "type": kind, "oid": oid, "name": name})
    queries: list[str] = []
    rest_calls: list[str] = []

    def request(
        url: str, headers: dict[str, str], payload: dict[str, object] | None = None
    ) -> object:
        assert url == "https://api.github.com/graphql"
        assert payload is not None
        query = str(payload["query"])
        queries.append(query)
        result: dict[str, object] = {}
        for alias, oid in re.findall(
            r'(commit|o\d+):object\(oid:"([a-f0-9]{40})"\)', query
        ):
            if alias == "commit":
                result[alias] = {
                    "oid": sha,
                    "message": "source\n",
                    "tree": {"oid": tree},
                    "author": {
                        "name": "Test",
                        "email": "test@example.org",
                        "date": "1970-01-01T00:00:01Z",
                    },
                    "committer": {
                        "name": "Test",
                        "email": "test@example.org",
                        "date": "1970-01-01T00:00:01Z",
                    },
                    "parents": {"nodes": [], "pageInfo": {"hasNextPage": False}},
                    "signature": None,
                }
            elif oid == tree:
                result[alias] = {"oid": tree, "entries": entries}
            else:
                content = contents[oid]
                result[alias] = {
                    "oid": oid,
                    "isBinary": content == binary,
                    "isTruncated": False,
                    "text": (
                        None
                        if content == binary
                        else ("altered" if tamper else content.decode())
                    ),
                    "byteSize": (
                        len(b"altered")
                        if tamper and content != binary
                        else len(content)
                    ),
                }
        return {"data": {"repository": result}}

    def rest(path: str) -> object:
        rest_calls.append(path)
        oid = path.rsplit("/", 1)[1]
        assert contents[oid] == binary
        return {
            "sha": oid,
            "encoding": "base64",
            "content": base64.b64encode(binary).decode(),
        }

    monkeypatch.setattr(source, "_get_json", request)
    client = source._GraphQLSource("https://api.github.com/graphql", "owner/repo", {})
    commit = source._verified_commit(client.commit(sha), sha)
    assert commit.raw == raw_commit
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if tamper:
        with pytest.raises(ValueError, match="does not match its SHA"):
            source._materialize_api(
                workspace,
                commit,
                rest,
                get_objects=lambda missing: client.objects(missing, rest),
            )
        return
    assert source._materialize_api(
        workspace,
        commit,
        rest,
        get_objects=lambda missing: client.objects(missing, rest),
    )
    assert len(queries) <= 15
    assert len(rest_calls) == 1
    source._checkout(workspace, commit, None)
    assert _git(workspace, "status", "--porcelain") == ""
    assert (workspace / "binary.dat").read_bytes() == binary
    assert (workspace / "run.sh").stat().st_mode & 0o111
    assert (workspace / "link").is_symlink()
    assert (workspace / "example-095.txt").read_bytes() == b"source 95\r\n"
    queries.clear()
    assert source._materialize_api(
        workspace,
        commit,
        rest,
        get_objects=lambda missing: client.objects(missing, rest),
    )
    assert queries == []


@pytest.mark.parametrize(
    "response",
    [
        {"errors": [{"type": "FORBIDDEN"}]},
        {"data": {"repository": None}},
        {"data": {"repository": {}}},
    ],
)
def test_graphql_errors_and_missing_objects_are_visible(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, object]
) -> None:
    monkeypatch.setattr(source, "_get_json", lambda *args: response)
    client = source._GraphQLSource("https://api.github.com/graphql", "owner/repo", {})
    with pytest.raises((ValueError, RuntimeError)):
        client.objects([("a" * 40, "blob")], lambda _: pytest.fail("REST fallback"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("isBinary", None),
        ("isTruncated", None),
        ("byteSize", True),
        ("byteSize", 4),
        ("text", None),
        ("oid", "b" * 40),
    ],
)
def test_graphql_incomplete_blob_metadata_is_refused(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    row = {
        "oid": "a" * 40,
        "isBinary": False,
        "isTruncated": False,
        "text": "abc",
        "byteSize": 3,
        field: value,
    }
    monkeypatch.setattr(
        source, "_get_json", lambda *args: {"data": {"repository": {"o0": row}}}
    )
    client = source._GraphQLSource("https://api.github.com/graphql", "owner/repo", {})
    with pytest.raises(ValueError):
        client.objects([("a" * 40, "blob")], lambda _: pytest.fail("REST fallback"))


@pytest.mark.parametrize(
    "missing",
    [[], [("a" * 40, "blob")] * 9, [("not-a-sha", "blob")], [("a" * 40, "commit")]],
)
def test_graphql_object_batches_are_bounded_before_network(
    monkeypatch: pytest.MonkeyPatch, missing: list[tuple[str, str]]
) -> None:
    monkeypatch.setattr(source, "_get_json", lambda *args: pytest.fail("network"))
    client = source._GraphQLSource("https://api.github.com/graphql", "owner/repo", {})
    with pytest.raises(ValueError):
        client.objects(missing, lambda _: pytest.fail("REST fallback"))


@pytest.mark.parametrize("workflow", ["ci.yml", "local-gpu.yml"])
@pytest.mark.parametrize(
    "alteration", ["none", "content", "length", "binary", "truncated", "denied"]
)
def test_workflow_bootstrap_checks_graphql_bytes_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, alteration: str
) -> None:
    import hashlib
    import io
    import json
    import runpy
    import urllib.request

    import yaml

    document = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows" / workflow
        ).read_text()
    )
    step = next(
        s
        for s in document["jobs"]["prepare-source"]["steps"]
        if s.get("id") == "archive-source"
    )
    script = step["run"].split("python3 - <<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    code = b"print('checked source')\n"
    digest = hashlib.sha1(b"blob " + str(len(code)).encode() + b"\0" + code).hexdigest()
    row = {
        "oid": digest,
        "byteSize": len(code),
        "isBinary": False,
        "isTruncated": False,
        "text": code.decode(),
    }
    if alteration == "content":
        row.update(text="print('changed source')\n")
        row["byteSize"] = len(str(row["text"]).encode())
    elif alteration == "length":
        row["byteSize"] = len(code) + 1
    elif alteration == "binary":
        row["isBinary"] = True
    elif alteration == "truncated":
        row["isTruncated"] = True

    def open_request(request: urllib.request.Request, timeout: int) -> io.BytesIO:
        assert timeout == 30
        assert request.full_url == "https://api.github.com/graphql"
        assert request.get_header("Authorization") == "Bearer fixture"
        assert request.get_method() == "POST"
        assert request.data is not None
        payload = json.loads(request.data)
        assert payload["variables"] == {
            "owner": "owner",
            "name": "repo",
            "expression": "a" * 40 + ":tools/ci_source_archive.py",
        }
        data = (
            {"errors": [{"type": "FORBIDDEN"}]}
            if alteration == "denied"
            else {"data": {"repository": {"object": row}}}
        )
        return io.BytesIO(json.dumps(data).encode())

    executed: list[bytes] = []
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_GRAPHQL_URL", "https://api.github.com/graphql")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "fixture")
    monkeypatch.setattr(urllib.request, "urlopen", open_request)

    def run_path(path: str, run_name: str) -> None:
        assert run_name == "__main__"
        executed.append(Path(path).read_bytes())

    monkeypatch.setattr(runpy, "run_path", run_path)
    if alteration == "none":
        exec(compile(script, workflow, "exec"), {})
        assert executed == [code]
    else:
        with pytest.raises(ValueError):
            exec(compile(script, workflow, "exec"), {})
        assert executed == []
        assert not (tmp_path / "ci_source_archive.py").exists()
