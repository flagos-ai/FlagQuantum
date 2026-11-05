"""Restore exact CI source using GitHub's API and source archive endpoints."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class _Commit:
    sha: str
    tree: str
    raw: bytes


def _verified_commit(metadata: object, expected_sha: str) -> _Commit:
    if not isinstance(metadata, dict) or metadata.get("sha") != expected_sha:
        raise ValueError("GitHub returned a different source commit")
    tree = metadata.get("tree")
    verification = metadata.get("verification")
    if not isinstance(tree, dict) or not isinstance(verification, dict):
        raise ValueError("GitHub commit metadata is incomplete")
    tree_sha = tree.get("sha")
    payload = verification.get("payload")
    signature = verification.get("signature")
    if not isinstance(tree_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise ValueError("GitHub returned an invalid source tree")
    if not isinstance(payload, str) or not isinstance(signature, str):
        raise ValueError("Git transport is required for this unsigned commit")
    if not payload.startswith(f"tree {tree_sha}\n"):
        raise ValueError("The signed commit describes a different source tree")
    headers, separator, message = payload.partition("\n\n")
    if not separator:
        raise ValueError("GitHub returned an invalid signed commit payload")
    # GitHub and Git differ in whether the signature retains its final newline.
    for signing_text in (signature, signature.rstrip("\n")):
        signature_header = signing_text.replace("\n", "\n ")
        raw = f"{headers}\ngpgsig {signature_header}\n\n{message}".encode()
        digest = hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw)
        if digest.hexdigest() == expected_sha:
            return _Commit(expected_sha, tree_sha, raw)
    raise ValueError("GitHub metadata cannot reproduce the requested commit SHA")


def _git(workspace: Path, *arguments: str, data: bytes | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        input=data,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return result.stdout.decode().strip()


def _archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ValueError("GitHub returned an empty source archive")
    root = PurePosixPath(members[0].name).parts[0]
    result = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or path.parts[0] != root or ".." in path.parts:
            raise ValueError("Source archive contains an invalid path")
        if len(path.parts) == 1:
            continue
        relative = PurePosixPath(*path.parts[1:])
        if any(part.lower() == ".git" for part in relative.parts) or member.islnk():
            raise ValueError("Source archive contains Git metadata or hard links")
        extracted = copy.copy(member)
        extracted.name = str(relative)
        result.append(extracted)
    return result


def _restore(workspace: Path, archive_path: Path, commit: _Commit) -> str | None:
    with tarfile.open(archive_path) as archive:
        members = _archive_members(archive)
        _git(workspace, "init", "--quiet")
        try:
            baseline = _git(workspace, "rev-parse", "--verify", "HEAD")
            _git(
                workspace,
                "rev-list",
                "--objects",
                "--missing=error",
                "--no-walk",
                "HEAD",
            )
        except subprocess.CalledProcessError:
            baseline = None
        for key, value in (
            ("core.autocrlf", "false"),
            ("core.filemode", "true"),
            ("core.sparseCheckout", "false"),
            ("core.sparseCheckoutCone", "false"),
            ("gc.auto", "0"),
        ):
            _git(workspace, "config", "--local", key, value)
        _git(workspace, "read-tree", "--empty")
        _git(workspace, "clean", "-ffdx")
        archive.extractall(workspace, members=members, filter="data")
    _git(workspace, "add", "--force", "--all", ".")
    if _git(workspace, "write-tree") != commit.tree:
        raise ValueError("The source archive does not match the commit's Git tree")
    return _checkout(workspace, commit, baseline)


def _checkout(workspace: Path, commit: _Commit, baseline: str | None) -> str | None:
    if (
        _git(workspace, "hash-object", "-t", "commit", "-w", "--stdin", data=commit.raw)
        != commit.sha
    ):
        raise ValueError("The restored commit does not match the requested SHA")
    shallow = Path(_git(workspace, "rev-parse", "--git-path", "shallow"))
    if not shallow.is_absolute():
        shallow = workspace / shallow
    previous = shallow.read_text().splitlines() if shallow.exists() else []
    staged = shallow.with_name("shallow.ci-source-tmp")
    staged.write_text("\n".join(sorted(set(previous) | {commit.sha})) + "\n")
    staged.replace(shallow)
    _git(workspace, "checkout", "--force", "--detach", commit.sha)
    _git(workspace, "clean", "-ffdx")
    if _git(workspace, "rev-parse", "HEAD") != commit.sha:
        raise ValueError("The restored working tree has a different source commit")
    return baseline


def _materialize_api(
    workspace: Path, commit: _Commit, get_json: Callable[[str], object]
) -> bool:
    _git(workspace, "init", "--quiet")
    pending = [(commit.tree, "tree")]
    visited: set[str] = set()
    requests = 0
    while pending:
        sha, kind = pending.pop()
        if sha in visited:
            continue
        visited.add(sha)
        status = _git(workspace, "cat-file", "--batch-check", data=f"{sha}\n".encode())
        if not status.endswith(" missing"):
            if kind == "tree":
                try:
                    _git(workspace, "rev-list", "--objects", "--missing=error", sha)
                except subprocess.CalledProcessError:
                    records = _git(workspace, "ls-tree", "-z", sha).split("\0")
                    for record in records:
                        if not record:
                            continue
                        header = record.split("\t", 1)[0]
                        _, child_kind, child_sha = header.split()
                        if child_kind in ("tree", "blob"):
                            pending.append((child_sha, child_kind))
            continue
        requests += 1
        if requests > 64:
            print("Cache needs more than 64 requests; using the archive", flush=True)
            return False
        print(f"Fetching missing {kind} {sha} through the API", flush=True)
        data = get_json(f"/git/{kind}s/{sha}")
        if not isinstance(data, dict) or data.get("sha") != sha:
            raise ValueError("GitHub returned a different Git object")
        if kind == "blob":
            content = data.get("content")
            if data.get("encoding") != "base64" or not isinstance(content, str):
                raise ValueError("GitHub returned an invalid blob encoding")
            raw = base64.b64decode("".join(content.split()), validate=True)
            actual = _git(workspace, "hash-object", "-w", "--stdin", data=raw)
        else:
            if data.get("truncated") is True:
                return False
            children = data.get("tree")
            if not isinstance(children, list):
                raise ValueError("GitHub returned an invalid tree")
            records = []
            for child in children:
                if not isinstance(child, dict) or not all(
                    isinstance(child.get(key), str)
                    for key in ("mode", "type", "sha", "path")
                ):
                    raise ValueError("GitHub returned an invalid tree member")
                if child["type"] in ("tree", "blob"):
                    pending.append((child["sha"], child["type"]))
                records.append(
                    f"{child['mode']} {child['type']} {child['sha']}\t{child['path']}\0"
                )
            actual = _git(
                workspace, "mktree", "--missing", "-z", data="".join(records).encode()
            )
        if actual != sha:
            raise ValueError("GitHub object data does not match its SHA")
    return True


class _Redirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            result is not None
            and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc
        ):
            result.remove_header("Authorization")
        return result


def _main() -> None:
    sha = os.environ["GITHUB_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GITHUB_SHA must identify a Git commit")
    repository = os.environ["GITHUB_REPOSITORY"]
    api = os.environ["GITHUB_API_URL"].rstrip("/")
    headers = {"Accept": "application/vnd.github+json", "Accept-Encoding": "gzip"}
    token = os.environ.get("GH_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = build_opener(_Redirect())

    def get_json(path: str) -> object:
        request = Request(f"{api}/repos/{repository}{path}", headers=headers)
        with opener.open(request, timeout=30) as response:
            data = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            return json.loads(data)

    print("Fetching GitHub commit metadata", flush=True)
    commit = _verified_commit(get_json(f"/git/commits/{sha}"), sha)
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    if _materialize_api(workspace, commit, get_json):
        try:
            baseline = _git(workspace, "rev-parse", "--verify", "HEAD")
        except subprocess.CalledProcessError:
            baseline = None
        _git(workspace, "config", "--local", "core.sparseCheckout", "false")
        _git(workspace, "config", "--local", "core.sparseCheckoutCone", "false")
        _git(workspace, "read-tree", "--empty")
        _git(workspace, "clean", "-ffdx")
        _checkout(workspace, commit, baseline)
        if baseline:
            with Path(os.environ["GITHUB_ENV"]).open("a") as environment:
                environment.write(f"FQ_CI_SOURCE_BASE={baseline}\n")
        print(f"Restored and verified source commit {sha}")
        return
    with tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"]) as directory:
        archive_path = Path(directory) / "source.tar.gz"
        request = Request(f"{api}/repos/{repository}/tarball/{sha}", headers=headers)
        print("Downloading the exact source archive", flush=True)
        with (
            opener.open(request, timeout=60) as response,
            archive_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        print("Verifying the archived Git tree and commit", flush=True)
        baseline = _restore(Path(os.environ["GITHUB_WORKSPACE"]), archive_path, commit)
    if baseline:
        with Path(os.environ["GITHUB_ENV"]).open("a") as environment:
            environment.write(f"FQ_CI_SOURCE_BASE={baseline}\n")
    print(f"Restored and verified source commit {sha}")


if __name__ == "__main__":
    _main()
