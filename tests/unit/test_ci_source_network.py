"""Transient source API failures retry within bounds without relaxing verification."""

from __future__ import annotations

import gzip
import io
from email.message import Message
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError

import pytest

from tools import ci_source_archive as source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("failure", ["disconnect", "timeout", "url", "503", "429"])
def test_transient_request_recovers_with_an_independent_opener(
    monkeypatch: pytest.MonkeyPatch, compressed: bool, failure: str
) -> None:
    calls: list[int] = []
    delays: list[float] = []
    content = b'{"sha": "verified-by-caller"}'

    class Response(io.BytesIO):
        headers = Message()

    class Opener:
        def open(self, request: source.Request, timeout: int) -> Response:
            assert timeout == 30
            assert request.get_header("Authorization") == "Bearer fixture"
            if len(calls) == 1:
                if failure == "disconnect":
                    raise RemoteDisconnected("connection closed")
                if failure == "timeout":
                    raise TimeoutError("deadline")
                if failure == "url":
                    raise URLError("connection reset")
                raise HTTPError(
                    request.full_url, int(failure), "transient", Message(), None
                )
            result = Response(gzip.compress(content) if compressed else content)
            if compressed:
                result.headers["Content-Encoding"] = "gzip"
            return result

    def opener(*handlers: object) -> Opener:
        calls.append(1)
        return Opener()

    monkeypatch.setattr(source, "build_opener", opener)
    monkeypatch.setattr(source.time, "sleep", delays.append)
    assert source._get_json(
        "https://api.github.com/repos/owner/repo/git/trees/tree",
        {"Authorization": "Bearer fixture"},
    ) == {"sha": "verified-by-caller"}
    assert len(calls) == 2
    assert delays == [1]


@pytest.mark.parametrize("status,attempts", [(401, 1), (403, 1), (404, 1), (503, 3)])
def test_request_failures_remain_visible_after_the_retry_bound(
    monkeypatch: pytest.MonkeyPatch, status: int, attempts: int
) -> None:
    calls: list[int] = []
    delays: list[float] = []

    class Opener:
        def open(self, request: source.Request, timeout: int) -> None:
            calls.append(timeout)
            raise HTTPError(request.full_url, status, "refused", Message(), None)

    monkeypatch.setattr(source, "build_opener", lambda *handlers: Opener())
    monkeypatch.setattr(source.time, "sleep", delays.append)
    with pytest.raises(HTTPError) as error:
        source._get_json("https://api.github.com/source", {})
    assert error.value.code == status
    assert len(calls) == attempts
    assert delays == ([1, 2] if attempts == 3 else [])


def test_graphql_post_serializes_payload_without_mutating_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    class Response(io.BytesIO):
        headers = Message()

    payload: dict[str, object] = {
        "query": "query{viewer{login}}",
        "variables": {"value": 'quoted " text'},
    }
    headers = {"Authorization": "Bearer fixture"}

    class Opener:
        def open(self, request: source.Request, timeout: int) -> Response:
            assert timeout == 30
            assert request.get_method() == "POST"
            assert request.get_header("Content-type") == "application/json"
            assert request.data is not None
            assert json.loads(request.data) == payload
            return Response(b'{"data": {}}')

    monkeypatch.setattr(source, "build_opener", lambda *handlers: Opener())
    assert source._get_json("https://api.github.com/graphql", headers, payload) == {
        "data": {}
    }
    assert headers == {"Authorization": "Bearer fixture"}


@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("compressed", [False, True])
def test_incomplete_response_reads_retry_without_accepting_partial_bytes(
    monkeypatch: pytest.MonkeyPatch, failures: int, compressed: bool
) -> None:
    responses: list[io.BytesIO] = []
    delays: list[float] = []
    content = b'{"sha": "verified-by-caller"}'

    class Response(io.BytesIO):
        def __init__(self, fail: bool) -> None:
            super().__init__(gzip.compress(content) if compressed else content)
            self.fail = fail
            self.headers = Message()
            if compressed:
                self.headers["Content-Encoding"] = "gzip"

        def read(self, size: int = -1) -> bytes:
            if self.fail:
                raise IncompleteRead(b'{"sha": "untrusted', 12)
            return super().read(size)

    class Opener:
        def open(self, request: source.Request, timeout: int) -> Response:
            assert timeout == 30
            response = Response(len(responses) < failures)
            responses.append(response)
            return response

    monkeypatch.setattr(source, "build_opener", lambda *handlers: Opener())
    monkeypatch.setattr(source.time, "sleep", delays.append)
    if failures == 3:
        with pytest.raises(IncompleteRead):
            source._get_json("https://api.github.com/source", {})
    else:
        assert source._get_json("https://api.github.com/source", {}) == {
            "sha": "verified-by-caller"
        }
    assert len(responses) == (2 if failures == 1 else 3)
    assert all(response.closed for response in responses)
    assert delays == ([1] if failures == 1 else [1, 2])
