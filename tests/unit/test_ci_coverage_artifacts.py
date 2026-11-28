"""Coverage transport bounds active I/O and refuses incomplete or altered inputs."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import stat
import threading
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from tools.ci import ci_coverage_artifacts as transport

pytestmark = pytest.mark.unit
HEAD = "a" * 40
SOURCE = "b" * 40
RUN = 42


def _zip(name: str, data: bytes, mode: int = stat.S_IFREG | 0o600) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        info = ZipInfo(name)
        info.external_attr = mode << 16
        archive.writestr(info, data)
    return output.getvalue()


def _records(payloads: list[bytes]) -> list[dict[str, Any]]:
    return [
        {
            "name": f"coverage-shard-{i}-{SOURCE}",
            "id": 100 + i,
            "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "expired": False,
            "size_in_bytes": len(raw),
            "workflow_run": {"id": RUN, "head_sha": HEAD},
        }
        for i, raw in enumerate(payloads)
    ]


def _api(
    records: list[dict[str, Any]], payloads: list[bytes]
) -> Callable[[str, str, float], bytes]:
    def fetch(url: str, token: str, deadline: float) -> bytes:
        assert token == "fixture-secret"
        assert deadline > transport.time.monotonic()
        if "/artifacts?" in url:
            return json.dumps(
                {"artifacts": records, "total_count": len(records)}
            ).encode()
        if url.endswith(f"/runs/{RUN}"):
            return json.dumps({"id": RUN, "head_sha": HEAD, "run_attempt": 1}).encode()
        return payloads[int(url.split("/")[-2]) - 100]

    return fetch


def _download(output: Path, count: int) -> dict[str, str]:
    return transport.download(
        "https://api.github.com",
        "owner/repo",
        RUN,
        1,
        HEAD,
        SOURCE,
        count,
        output,
        "fixture-secret",
    )


def test_roundtrip_preserves_original_sqlite_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "original.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('original lines')")
    original = database.read_bytes()
    payloads = [_zip(f".coverage.{i}", original) for i in range(3)]
    monkeypatch.setattr(transport, "_fetch", _api(_records(payloads), payloads))
    output = tmp_path / "inputs"
    hashes = _download(output, 3)
    assert set(hashes) == {f".coverage.{i}" for i in range(3)}
    for name, digest in hashes.items():
        assert (output / name).read_bytes() == original
        assert digest == hashlib.sha256(original).hexdigest()
        with sqlite3.connect(
            (output / name).as_uri() + "?mode=ro", uri=True
        ) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_only_active_workers_initiate_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = [_zip(f".coverage.{i}", b"unchanged") for i in range(8)]
    fetch = _api(_records(payloads), payloads)
    release, saturated = threading.Event(), threading.Event()
    lock = threading.Lock()
    started: list[str] = []
    failures: list[BaseException] = []
    active = peak = 0

    def blocked(url: str, token: str, deadline: float) -> bytes:
        nonlocal active, peak
        if not url.endswith("/zip"):
            return fetch(url, token, deadline)
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(url)
            if active == 2:
                saturated.set()
        try:
            assert release.wait(10), "test did not release bounded transfers"
            return fetch(url, token, deadline)
        finally:
            with lock:
                active -= 1

    def run() -> None:
        try:
            _download(tmp_path / "inputs", 8)
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(transport, "_fetch", blocked)
    monkeypatch.setattr(transport, "_DOWNLOADS", 2)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert saturated.wait(5)
        with lock:
            assert len(started) == active == peak == 2
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not failures
    assert len(started) == 8 and peak == 2


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "extra",
        "wrong-source",
        "wrong-run",
        "wrong-head",
        "expired",
        "duplicate-id",
        "boolean-id",
        "digest",
        "size",
    ],
)
def test_refuses_wrong_or_incomplete_inventory_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    payloads = [_zip(f".coverage.{i}", b"original") for i in range(2)]
    records = _records(payloads)
    if case == "missing":
        records.pop()
    elif case == "duplicate":
        records.append(records[0])
    elif case == "extra":
        records.append({**records[0], "name": f"coverage-shard-2-{SOURCE}"})
    elif case == "wrong-source":
        records[0]["name"] = f"coverage-shard-0-{HEAD}"
    elif case == "wrong-run":
        records[0]["workflow_run"]["id"] = RUN + 1
    elif case == "wrong-head":
        records[0]["workflow_run"]["head_sha"] = SOURCE
    elif case == "expired":
        records[0]["expired"] = True
    elif case == "duplicate-id":
        records[0]["id"] = records[1]["id"]
    elif case == "boolean-id":
        records[0]["id"] = True
    elif case == "digest":
        records[0]["digest"] = "missing"
    else:
        records[0]["size_in_bytes"] = transport._MAX_ZIP_BYTES + 1
    fetch = _api(records, payloads)

    def no_download(url: str, token: str, deadline: float) -> bytes:
        assert not url.endswith("/zip")
        return fetch(url, token, deadline)

    monkeypatch.setattr(transport, "_fetch", no_download)
    with pytest.raises(ValueError):
        _download(tmp_path / "inputs", 2)
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize(
    "case", ["path", "symlink", "directory", "duplicate", "digest", "size", "corrupt"]
)
def test_refuses_unsafe_or_altered_archive_without_publishing_partial_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    raw = _zip(".coverage.0", b"original")
    if case == "path":
        raw = _zip("../.coverage.0", b"original")
    elif case == "symlink":
        raw = _zip(".coverage.0", b"elsewhere", stat.S_IFLNK | 0o777)
    elif case == "directory":
        raw = _zip(".coverage.0", b"original", stat.S_IFDIR | 0o700)
    elif case == "duplicate":
        buffer = io.BytesIO(raw)
        with (
            pytest.warns(UserWarning, match="Duplicate name"),
            ZipFile(buffer, "a") as archive,
        ):
            archive.writestr(".coverage.0", b"duplicate")
        raw = buffer.getvalue()
    elif case == "size":
        monkeypatch.setattr(transport, "_MAX_DATABASE_BYTES", 2)
    elif case == "corrupt":
        raw = raw.replace(b"original", b"modified")
    payloads = [raw, _zip(".coverage.1", b"other shard")]
    records = _records(payloads)
    if case == "digest":
        records[0]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(transport, "_fetch", _api(records, payloads))
    with pytest.raises((ValueError, transport.zipfile.BadZipFile)):
        _download(tmp_path / "inputs", 2)
    assert not (tmp_path / "inputs").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target",
    ["https://blob.example/file?sig=private", "https://api.github.com:443/file"],
)
def test_redirect_strips_credentials_across_origins(target: str) -> None:
    request = Request(
        "https://api.github.com/archive", headers={"Authorization": "Bearer secret"}
    )
    result = transport._Redirect().redirect_request(
        request, io.BytesIO(), 302, "Found", Message(), target
    )
    assert result is not None and result.get_header("Authorization") is None


def test_same_origin_retains_credentials_and_https_downgrade_is_refused() -> None:
    request = Request(
        "https://api.github.com/archive", headers={"Authorization": "Bearer secret"}
    )
    redirect = transport._Redirect()
    result = redirect.redirect_request(
        request, io.BytesIO(), 302, "Found", Message(), "https://api.github.com/another"
    )
    assert result is not None and result.get_header("Authorization") == "Bearer secret"
    for target in (
        "http://api.github.com/another",
        "https://user:password@blob.example/file",
    ):
        with pytest.raises(ValueError):
            redirect.redirect_request(
                request, io.BytesIO(), 302, "Found", Message(), target
            )


@pytest.mark.parametrize("status,expected", [(503, 3), (429, 3), (403, 1)])
def test_retry_is_bounded_and_preserves_http_refusals(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: int
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)

    class Opener:
        def open(self, request: Request, timeout: float) -> io.BytesIO:
            calls.append(status)
            assert 0 < timeout <= 15
            raise HTTPError(
                request.full_url, status, "fixture", Message(), io.BytesIO()
            )

    monkeypatch.setattr(transport, "build_opener", lambda handler: Opener())
    with pytest.raises(HTTPError):
        transport._fetch(
            "https://api.github.com/archive", "fixture", transport.time.monotonic() + 60
        )
    assert len(calls) == expected


def test_transient_request_retries_fresh_response_and_enforces_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)

    class Opener:
        def open(self, request: Request, timeout: float) -> io.BytesIO:
            calls.append(1)
            if len(calls) == 1:
                raise URLError("fixture disconnect")
            return io.BytesIO(b"verified")

    monkeypatch.setattr(transport, "build_opener", lambda handler: Opener())
    deadline = transport.time.monotonic() + 60
    assert (
        transport._fetch("https://api.github.com/archive", "fixture", deadline)
        == b"verified"
    )
    assert len(calls) == 2
    monkeypatch.setattr(transport, "_MAX_ZIP_BYTES", 2)
    with pytest.raises(ValueError, match="byte limit"):
        transport._fetch("https://api.github.com/archive", "fixture", deadline)
    assert len(calls) == 3
    with pytest.raises(TimeoutError):
        transport._fetch("https://api.github.com/archive", "fixture", 0)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "field,value", [("id", RUN + 1), ("head_sha", SOURCE), ("run_attempt", 2)]
)
def test_refuses_different_workflow_before_listing_or_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: int | str
) -> None:
    def fetch(url: str, token: str, deadline: float) -> bytes:
        assert url.endswith(f"/runs/{RUN}")
        return json.dumps(
            {"id": RUN, "head_sha": HEAD, "run_attempt": 1, field: value}
        ).encode()

    monkeypatch.setattr(transport, "_fetch", fetch)
    with pytest.raises(ValueError, match="workflow identity"):
        _download(tmp_path / "inputs", 1)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_preserved_without_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "retained"
    marker.write_bytes(b"previous evidence")
    monkeypatch.setattr(
        transport, "_fetch", lambda *args: pytest.fail("unexpected network")
    )
    with pytest.raises(ValueError, match="existing output"):
        _download(tmp_path, 1)
    assert marker.read_bytes() == b"previous evidence"


def test_partial_response_is_discarded_before_transient_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)

    class Partial(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise transport.IncompleteRead(b"abandoned", 100)
            return super().read(size)

    class Opener:
        def open(self, request: Request, timeout: float) -> io.BytesIO:
            calls.append(1)
            return Partial(b"partial") if len(calls) == 1 else io.BytesIO(b"complete")

    monkeypatch.setattr(transport, "build_opener", lambda handler: Opener())
    assert (
        transport._fetch(
            "https://api.github.com/archive", "fixture", transport.time.monotonic() + 60
        )
        == b"complete"
    )
    assert len(calls) == 2


def test_cli_does_not_print_signed_urls_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "transport",
            "--run-id",
            str(RUN),
            "--attempt",
            "1",
            "--count",
            "1",
            "--head-sha",
            HEAD,
            "--source-sha",
            SOURCE,
            "--output",
            str(tmp_path / "inputs"),
        ],
    )
    for key, value in {
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_REPOSITORY": "owner/repo",
        "GH_TOKEN": "private-token",
    }.items():
        monkeypatch.setenv(key, value)

    def fail(*args: object) -> dict[str, str]:
        raise HTTPError(
            "https://blob.example/archive?sig=private-signature",
            503,
            "private-token",
            Message(),
            io.BytesIO(),
        )

    monkeypatch.setattr(transport, "download", fail)
    with pytest.raises(SystemExit) as error:
        transport._main()
    assert str(error.value) == "Coverage artifact transport failed: HTTPError"
