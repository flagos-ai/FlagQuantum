"""Restore this run's complete coverage inputs with eight active downloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import HTTPMessage, IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_DOWNLOADS = 8
_MAX_ZIP_BYTES = 8 * 1024 * 1024
_MAX_DATABASE_BYTES = 16 * 1024 * 1024
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class _Artifact:
    identifier: int
    index: int
    digest: str


class _Redirect(HTTPRedirectHandler):
    """Require HTTPS and never forward credentials to another origin."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        target = urlsplit(newurl)
        if target.scheme != "https" or target.username or target.password:
            raise ValueError("Artifact redirect must use authenticated-free HTTPS")
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urlsplit(req.full_url).netloc != target.netloc:
            result.remove_header("Authorization")
        return result


def _fetch(url: str, token: str, deadline: float) -> bytes:
    """Bound responses and retry only transient transport errors, at most twice."""
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Coverage artifact transport deadline exceeded")
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            # Each worker owns its redirect handler and connection. Logs never
            # contain credentials or the signed object-storage redirect URL.
            with build_opener(_Redirect()).open(
                request, timeout=min(15.0, remaining)
            ) as response:
                chunks: list[bytes] = []
                size = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Coverage artifact transport deadline exceeded"
                        )
                    chunk = response.read(min(65536, _MAX_ZIP_BYTES + 1 - size))
                    if not chunk:
                        return b"".join(chunks)
                    size += len(chunk)
                    if size > _MAX_ZIP_BYTES:
                        raise ValueError(
                            "Coverage artifact response exceeds the byte limit"
                        )
                    chunks.append(chunk)
        except HTTPError as error:
            error.close()
            if error.code not in _TRANSIENT_STATUS or attempt == 2:
                raise
        except (URLError, RemoteDisconnected, IncompleteRead, TimeoutError):
            if attempt == 2:
                raise
        print(f"Retrying transient coverage transport ({attempt + 1}/2)", flush=True)
        time.sleep(min(attempt + 1, max(0.0, deadline - time.monotonic())))
    raise RuntimeError("Coverage transport retry loop ended unexpectedly")


def _select(
    records: list[object], count: int, run_id: int, head_sha: str, source_sha: str
) -> list[_Artifact]:
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("Invalid coverage shard count")
    expected = {f"coverage-shard-{i}-{source_sha}": i for i in range(count)}
    found: dict[str, _Artifact] = {}
    identifiers: set[int] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("Invalid artifact metadata")
        name = record["name"]
        if not name.startswith("coverage-shard-"):
            continue
        workflow = record.get("workflow_run")
        identifier, digest = record.get("id"), record.get("digest")
        if (
            name not in expected
            or name in found
            or type(identifier) is not int
            or identifier <= 0
            or identifier in identifiers
            or record.get("expired") is not False
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(workflow, dict)
            or workflow.get("id") != run_id
            or workflow.get("head_sha") != head_sha
            or type(record.get("size_in_bytes")) is not int
            or not 0 < record["size_in_bytes"] <= _MAX_ZIP_BYTES
        ):
            raise ValueError(
                "Coverage artifact identity, uniqueness or integrity metadata differs"
            )
        identifiers.add(identifier)
        found[name] = _Artifact(identifier, expected[name], digest)
    if set(found) != set(expected):
        raise ValueError("Coverage artifact set is incomplete")
    return sorted(found.values(), key=lambda artifact: artifact.index)


def _restore(raw: bytes, artifact: _Artifact, directory: Path) -> str:
    if "sha256:" + hashlib.sha256(raw).hexdigest() != artifact.digest:
        raise ValueError("Coverage artifact SHA-256 differs")
    name = f".coverage.{artifact.index}"
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 1:
            raise ValueError("Coverage ZIP must contain exactly one database")
        entry = entries[0]
        mode = entry.external_attr >> 16
        if (
            entry.filename != name
            or entry.is_dir()
            or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
            or not 0 < entry.file_size <= _MAX_DATABASE_BYTES
            or entry.flag_bits & 1
        ):
            raise ValueError("Coverage ZIP database identity or file type differs")
        # Reading the single size-bounded member checks its CRC too. Never use
        # extractall or accept an archive-controlled destination path.
        data = archive.read(entry)
    with (directory / name).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def download(
    api: str,
    repository: str,
    run_id: int,
    attempt: int,
    head_sha: str,
    source_sha: str,
    count: int,
    output: Path,
    token: str,
) -> dict[str, str]:
    """Publish the exact verified input set atomically, without partial success."""
    address = urlsplit(api)
    if (
        address.scheme != "https"
        or not address.netloc
        or address.username
        or address.password
        or address.query
        or address.fragment
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or type(run_id) is not int
        or run_id <= 0
        or type(attempt) is not int
        or attempt <= 0
        or not re.fullmatch(r"[0-9a-f]{40}", head_sha)
        or not re.fullmatch(r"[0-9a-f]{40}", source_sha)
        or type(count) is not int
        or not 1 <= count <= 256
        or not token
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("Invalid coverage transport inputs or existing output")
    deadline = time.monotonic() + 180
    base = f"{api.rstrip('/')}/repos/{repository}/actions"
    run = json.loads(_fetch(f"{base}/runs/{run_id}", token, deadline))
    if (
        not isinstance(run, dict)
        or run.get("id") != run_id
        or run.get("head_sha") != head_sha
        or run.get("run_attempt") != attempt
    ):
        raise ValueError("Coverage workflow identity differs")
    records: list[object] = []
    for page in range(1, 65):
        listing = json.loads(
            _fetch(
                f"{base}/runs/{run_id}/artifacts?per_page=100&page={page}",
                token,
                deadline,
            )
        )
        if (
            not isinstance(listing, dict)
            or type(listing.get("total_count")) is not int
            or not isinstance(listing.get("artifacts"), list)
        ):
            raise ValueError("Invalid workflow artifact listing")
        records.extend(listing["artifacts"])
        if len(records) >= listing["total_count"]:
            break
    else:
        raise ValueError("Workflow artifact listing exceeds the page limit")
    artifacts = _select(records, count, run_id, head_sha, source_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".coverage-download-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "verified"
        staging.mkdir()

        def restore(artifact: _Artifact) -> tuple[str, str]:
            raw = _fetch(f"{base}/artifacts/{artifact.identifier}/zip", token, deadline)
            return f".coverage.{artifact.index}", _restore(raw, artifact, staging)

        hashes: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=_DOWNLOADS) as pool:
            # Only workers initiate requests; queued futures do no network I/O.
            futures = [pool.submit(restore, artifact) for artifact in artifacts]
            try:
                for future in as_completed(futures):
                    name, digest = future.result()
                    hashes[name] = digest
                    if len(hashes) % 16 == 0 or len(hashes) == count:
                        print(
                            f"Verified coverage artifacts {len(hashes)}/{count}",
                            flush=True,
                        )
            except BaseException:
                # Cleanup boundary: cancel queued work, drain active requests,
                # and propagate the original failure without publishing output.
                for future in futures:
                    future.cancel()
                raise
        if time.monotonic() >= deadline:
            raise TimeoutError("Coverage artifact transport deadline exceeded")
        staging.rename(output)
    return dict(sorted(hashes.items()))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "attempt", "count"):
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ("head-sha", "source-sha"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        hashes = download(
            os.environ["GITHUB_API_URL"],
            os.environ["GITHUB_REPOSITORY"],
            args.run_id,
            args.attempt,
            args.head_sha,
            args.source_sha,
            args.count,
            args.output,
            os.environ["GH_TOKEN"],
        )
    except (
        HTTPError,
        URLError,
        RemoteDisconnected,
        IncompleteRead,
        TimeoutError,
    ) as error:
        # HTTPError includes its signed URL in __str__; never log that value.
        raise SystemExit(
            f"Coverage artifact transport failed: {type(error).__name__}"
        ) from None
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(
            f"Coverage artifact validation failed: {type(error).__name__}"
        ) from None
    print(f"Restored {len(hashes)} exact coverage databases", flush=True)


if __name__ == "__main__":
    _main()
