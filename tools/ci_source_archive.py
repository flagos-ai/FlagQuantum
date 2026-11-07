"""Restore exact CI source using GitHub's API and source archive endpoints."""

from __future__ import annotations

import argparse
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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPMessage, RemoteDisconnected
from pathlib import Path, PurePosixPath
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class _Commit:
    sha: str
    tree: str
    raw: bytes


def _unsigned_commit(metadata: dict[str, object], sha: str, tree: str) -> _Commit:
    """Recover omitted timezone spellings only when the exact Git SHA matches.

    GitHub dates preserve Unix seconds but normalize away the original offsets;
    messages may omit the final newline. Bound the supported encodings and keep
    Git transport as the fallback for uncommon headers or unrecoverable bytes.
    """
    parents = metadata.get("parents")
    message = metadata.get("message")
    if (
        not isinstance(parents, list)
        or len(parents) > 16
        or not isinstance(message, str)
        or len(message) > 16384
    ):
        raise ValueError(
            "Git transport is required for unsupported unsigned commit metadata"
        )
    headers = [f"tree {tree}"]
    for parent in parents:
        identifier = parent.get("sha") if isinstance(parent, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[0-9a-f]{40}", identifier
        ):
            raise ValueError("Unsigned source commit has an invalid parent")
        headers.append(f"parent {identifier}")
    actors: list[str] = []
    for role in ("author", "committer"):
        actor = metadata.get(role)
        if not isinstance(actor, dict) or any(
            not isinstance(actor.get(key), str) for key in ("name", "email", "date")
        ):
            raise ValueError("Unsigned source commit has incomplete actor metadata")
        name, email, date = actor["name"], actor["email"], actor["date"]
        if (
            any(character in name + email for character in "\r\n\0")
            or len(name) + len(email) > 2048
        ):
            raise ValueError("Unsigned source commit has invalid actor metadata")
        instant = datetime.fromisoformat(date.replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.microsecond:
            raise ValueError(
                "Unsigned source commit needs exact second-resolution dates"
            )
        seconds = int(instant.astimezone(timezone.utc).timestamp())
        actors.append(f"{role} {name} <{email}> {seconds}")
    # Modern civil offsets use quarter hours. Unsupported historical spellings
    # fail closed rather than silently substituting a different source commit.
    offsets = ["+0000", "-0000"] + [
        f"{'+' if minutes >= 0 else '-'}{abs(minutes)//60:02d}{abs(minutes)%60:02d}"
        for minutes in range(-12 * 60, 14 * 60 + 1, 15)
        if minutes
    ]
    prefix = "\n".join(headers)
    for author_offset in offsets:
        for committer_offset in offsets:
            header = f"{prefix}\n{actors[0]} {author_offset}\n{actors[1]} {committer_offset}\n\n"
            for ending in ("", "\n"):
                raw = (header + message + ending).encode()
                digest = hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw)
                if digest.hexdigest() == sha:
                    return _Commit(sha, tree, raw)
    raise ValueError(
        "Git transport is required: unsigned metadata cannot reproduce the requested commit SHA"
    )


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
        if (
            payload is None
            and signature is None
            and verification.get("reason") == "unsigned"
        ):
            return _unsigned_commit(metadata, expected_sha, tree_sha)
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
    workspace: Path,
    commit: _Commit,
    get_json: Callable[[str], object],
    *,
    get_objects: Callable[[list[tuple[str, str]]], list[object]] | None = None,
) -> bool:
    _git(workspace, "init", "--quiet")
    pending = [(commit.tree, "tree")]
    visited: set[str] = set()
    requests = 0
    # Fetch independent missing objects concurrently, but write the object store
    # serially. Large source additions must not force a slow full archive fetch.
    with ThreadPoolExecutor(max_workers=8) as executor:
        while pending:
            missing: list[tuple[str, str]] = []
            while pending and len(missing) < 8:
                sha, kind = pending.pop()
                if sha in visited:
                    continue
                visited.add(sha)
                status = _git(
                    workspace, "cat-file", "--batch-check", data=f"{sha}\n".encode()
                )
                if not status.endswith(" missing"):
                    if kind == "tree":
                        try:
                            _git(
                                workspace,
                                "rev-list",
                                "--objects",
                                "--missing=error",
                                sha,
                            )
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
                missing.append((sha, kind))
            if not missing:
                continue
            requests += len(missing)
            if requests > 2048:
                print(
                    "Cache needs more than 2048 objects; using the archive", flush=True
                )
                return False
            paths = [f"/git/{kind}s/{sha}" for sha, kind in missing]
            if missing:
                print(
                    f"Fetching {len(missing)} missing objects through the API ({requests} total)",
                    flush=True,
                )
            payloads = (
                get_objects(missing)
                if get_objects is not None
                else executor.map(get_json, paths)
            )
            for (sha, kind), data in zip(missing, payloads, strict=True):
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
                        workspace,
                        "mktree",
                        "--missing",
                        "-z",
                        data="".join(records).encode(),
                    )
                if actual != sha:
                    raise ValueError("GitHub object data does not match its SHA")
    return True


class _Redirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            result is not None
            and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc
        ):
            result.remove_header("Authorization")
        return result


def _get_json(
    url: str, headers: dict[str, str], payload: dict[str, object] | None = None
) -> object:
    """Retry transient API transport failures at most twice, preserving refusals."""
    for attempt in range(3):
        try:
            # Each parallel request owns its redirect handler and connection.
            opener = build_opener(_Redirect())
            data = None if payload is None else json.dumps(payload).encode()
            request_headers = dict(headers)
            if data is not None:
                request_headers["Content-Type"] = "application/json"
            request = Request(url, data=data, headers=request_headers)
            with opener.open(request, timeout=30) as response:
                data = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return json.loads(data)
        except HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, RemoteDisconnected, TimeoutError):
            if attempt == 2:
                raise
        print(f"Retrying transient GitHub API request ({attempt + 1}/2)", flush=True)
        time.sleep(attempt + 1)
    raise RuntimeError("GitHub API retry loop ended unexpectedly")


@dataclass(frozen=True)
class _GraphQLSource:
    """Batch Git objects using the same repository contents-read permission."""

    url: str
    repository: str
    headers: dict[str, str]

    def _query(self, fields: str) -> dict[str, object]:
        owner, name = self.repository.split("/", 1)
        query = (
            "query($owner:String!,$name:String!){"
            "repository(owner:$owner,name:$name){" + fields + "}}"
        )
        response = _get_json(
            self.url,
            self.headers,
            {"query": query, "variables": {"owner": owner, "name": name}},
        )
        if not isinstance(response, dict):
            raise ValueError("GitHub returned an invalid GraphQL response")
        if response.get("errors"):
            raise RuntimeError(
                "GitHub GraphQL source request failed: "
                + json.dumps(response["errors"])[:1000]
            )
        data = response.get("data")
        repository = data.get("repository") if isinstance(data, dict) else None
        if not isinstance(repository, dict):
            raise ValueError("GitHub returned no source repository")
        return repository

    def commit(self, sha: str) -> object:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Invalid source commit SHA")
        row = self._query(
            f'commit:object(oid:"{sha}"){{... on Commit{{'
            "oid message tree{oid} author{name email date} committer{name email date} "
            "parents(first:16){nodes{oid} pageInfo{hasNextPage}} "
            "signature{payload signature state}}}"
        ).get("commit")
        if not isinstance(row, dict) or row.get("oid") != sha:
            raise ValueError("GitHub returned a different source commit")
        parents, tree, signature = (
            row.get("parents"),
            row.get("tree"),
            row.get("signature"),
        )
        if not isinstance(parents, dict) or not isinstance(tree, dict):
            raise ValueError("GitHub returned incomplete source commit metadata")
        page = parents.get("pageInfo")
        nodes = parents.get("nodes")
        if (
            not isinstance(page, dict)
            or page.get("hasNextPage") is not False
            or not isinstance(nodes, list)
            or any(not isinstance(node, dict) for node in nodes)
        ):
            raise ValueError("GitHub source commit parents are incomplete")
        verification = {"payload": None, "signature": None, "reason": "unsigned"}
        if signature is not None:
            if not isinstance(signature, dict):
                raise ValueError("GitHub returned an invalid source signature")
            verification = signature
        return {
            "sha": sha,
            "tree": {"sha": tree.get("oid")},
            "author": row.get("author"),
            "committer": row.get("committer"),
            "message": row.get("message"),
            "parents": [{"sha": node.get("oid")} for node in nodes],
            "verification": verification,
        }

    def objects(
        self,
        missing: list[tuple[str, str]],
        get_json: Callable[[str], object],
    ) -> list[object]:
        if not 1 <= len(missing) <= 8 or any(
            not re.fullmatch(r"[0-9a-f]{40}", sha) or kind not in ("tree", "blob")
            for sha, kind in missing
        ):
            raise ValueError("GraphQL source batches require at most eight Git objects")
        fields = " ".join(
            f'o{index}:object(oid:"{sha}"){{oid '
            "... on Blob{byteSize isBinary isTruncated text} "
            "... on Tree{entries{mode name type oid}}}"
            for index, (sha, _) in enumerate(missing)
        )
        response = self._query(fields)
        result: list[object] = []
        for index, (sha, kind) in enumerate(missing):
            row = response.get(f"o{index}")
            if not isinstance(row, dict) or row.get("oid") != sha:
                raise ValueError("GitHub returned a different Git object")
            if kind == "blob":
                if row.get("isBinary") is True or row.get("isTruncated") is True:
                    result.append(get_json(f"/git/blobs/{sha}"))
                    continue
                text = row.get("text")
                if (
                    row.get("isBinary") is not False
                    or row.get("isTruncated") is not False
                    or not isinstance(text, str)
                    or type(row.get("byteSize")) is not int
                    or len(text.encode()) != row["byteSize"]
                ):
                    raise ValueError("GitHub returned incomplete source blob text")
                result.append(
                    {
                        "sha": sha,
                        "encoding": "base64",
                        "content": base64.b64encode(text.encode()).decode(),
                    }
                )
                continue
            entries = row.get("entries")
            if not isinstance(entries, list):
                raise ValueError("GitHub returned an incomplete source tree")
            children = []
            for entry in entries:
                if not isinstance(entry, dict) or type(entry.get("mode")) is not int:
                    raise ValueError("GitHub returned an invalid tree mode")
                children.append(
                    {
                        "mode": format(entry["mode"], "o"),
                        "type": entry.get("type"),
                        "sha": entry.get("oid"),
                        "path": entry.get("name"),
                    }
                )
            result.append({"sha": sha, "tree": children})
        return result


def cleanup_runner_containers() -> None:
    """Stop leftover CI writers before this exclusive runner starts a new checkout."""
    if os.environ.get("RUNNER_ENVIRONMENT") != "self-hosted" or not shutil.which(
        "docker"
    ):
        return
    runner = os.environ.get("RUNNER_NAME")
    workspace_text = os.environ.get("GITHUB_WORKSPACE")
    if not runner or not workspace_text:
        raise RuntimeError(
            "Runner identity and workspace are required for container cleanup"
        )
    workspace = Path(workspace_text).resolve()

    def docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            check=check,
            timeout=20,
        )

    listing = docker("ps", "--all", "--no-trunc", "--format", "{{.ID}} {{.Image}}")
    candidates = []
    for line in listing.stdout.splitlines():
        identifier, _, image = line.partition(" ")
        if re.fullmatch(r"[a-f0-9]{64}", identifier) and re.fullmatch(
            r"(?:ghcr\.io/flagos-ai/flagquantum-ci@)?sha256:[a-f0-9]{64}", image
        ):
            candidates.append(identifier)
    if not candidates:
        return
    inspection = docker("inspect", *candidates, check=False)
    # A --rm container may disappear between listing and inspection.
    metadata = json.loads(inspection.stdout or "[]")
    if inspection.returncode:
        inspected = {container["Id"] for container in metadata}
        remaining = docker("ps", "--all", "--no-trunc", "--format", "{{.ID}}")
        if (set(candidates) - inspected).intersection(remaining.stdout.splitlines()):
            raise RuntimeError("Cannot inspect previous CI containers")
    owned = []
    for container in metadata:
        environment = dict(
            value.split("=", 1)
            for value in container["Config"].get("Env", [])
            if "=" in value
        )
        mounts = container.get("Mounts", [])
        if (
            environment.get("RUNNER_NAME") == runner
            and environment.get("FQ_CI_PRIVATE")
            and any(
                mount.get("Type") == "bind"
                and Path(mount["Source"]).resolve() == workspace
                for mount in mounts
            )
        ):
            owned.append(container["Id"])
    if not owned:
        return
    print(
        f"Removing {len(owned)} previous CI containers for exclusive runner {runner}",
        flush=True,
    )
    for delay in (0.5, 2.0):
        result = docker("rm", "--force", *owned, check=False)
        if not result.returncode:
            return
        # Docker's --rm cleanup can already be removing the same container.
        # Give that removal time to finish before retrying or refusing checkout.
        time.sleep(delay)
        remaining = docker("ps", "--all", "--no-trunc", "--format", "{{.ID}}")
        present = set(remaining.stdout.splitlines())
        owned = [identifier for identifier in owned if identifier in present]
        if not owned:
            print("Previous CI containers were removed by Docker", flush=True)
            return
    raise RuntimeError(
        f"Could not remove previous CI containers: {result.stderr.strip()[:500]}"
    )


def _main() -> None:
    cleanup_runner_containers()
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
        return _get_json(f"{api}/repos/{repository}{path}", headers)

    print("Fetching GitHub commit metadata", flush=True)
    graphql_url = os.environ.get("GITHUB_GRAPHQL_URL")
    graphql = _GraphQLSource(graphql_url, repository, headers) if graphql_url else None
    commit = _verified_commit(
        graphql.commit(sha) if graphql else get_json(f"/git/commits/{sha}"), sha
    )

    def get_objects(missing: list[tuple[str, str]]) -> list[object]:
        assert graphql is not None
        return graphql.objects(missing, get_json)

    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    if _materialize_api(
        workspace, commit, get_json, get_objects=get_objects if graphql else None
    ):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-containers", action="store_true")
    if parser.parse_args().cleanup_containers:
        cleanup_runner_containers()
    else:
        _main()
