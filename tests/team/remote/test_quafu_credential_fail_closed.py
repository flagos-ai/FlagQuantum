"""Credential and retired-transport boundaries for Quafu execution."""

from __future__ import annotations

from typing import Any

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.deployment import CloudBackendProfile
from flagquantum.errors import ValidationError
from flagquantum.remote.qpu import _quafu_sdk
from flagquantum.remote.qpu.contracts import ProviderTaskHandle
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class SpyTransport:
    """Record legacy HTTP requests; static execution must never use it."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, str]]] = []
        self.gets: list[str] = []

    def post_json(
        self,
        url: str,
        payload: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        self.posts.append((url, dict(headers)))
        raise AssertionError("retired execution transport was used")

    def get_json(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        self.gets.append(url)
        raise AssertionError("retired execution transport was used")


def _package() -> Any:
    backend = CloudBackendProfile(provider="quafu", name="Baihua", n_qubits=156)
    return fqd.create_deployment_package(
        fq.Circuit(2).h(0).cx(0, 1),
        backend=backend,
        shots=1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )


def test_no_sdk_key_refuses_before_loading_sdk_or_using_legacy_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.setenv("QUAFU_API_TOKEN", "legacy-token-must-not-authorize-execution")
    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: pytest.fail("credential validation must precede SDK loading"),
    )
    transport = SpyTransport()
    provider = QuafuProvider(transport=transport)

    with pytest.raises(ValidationError, match="QUAFU_API_KEY"):
        provider.submit(_package())

    assert transport.posts == [] and transport.gets == []


def test_qasm_entry_point_has_the_same_sdk_only_credential_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.setenv("QUAFU_API_TOKEN", "legacy-token-must-not-authorize-execution")
    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: pytest.fail("credential validation must precede SDK loading"),
    )
    transport = SpyTransport()
    provider = QuafuProvider(transport=transport)

    with pytest.raises(ValidationError, match="QUAFU_API_KEY"):
        provider.submit_qasm(
            "OPENQASM 2.0;\nqreg q[2];\n",
            chip="Baihua",
            name="job",
            shots=1024,
            target_qubits=(0, 1),
        )

    assert transport.posts == [] and transport.gets == []


@pytest.mark.parametrize("protocol", ["sqc_legacy", "task_api_v1", None])
def test_retired_receipts_are_rejected_without_sdk_or_http(
    monkeypatch: pytest.MonkeyPatch, protocol: str | None
) -> None:
    monkeypatch.setattr(
        _quafu_sdk,
        "_load_quafu_sdk",
        lambda: pytest.fail("retired receipts must fail before SDK loading"),
    )
    transport = SpyTransport()
    provider = QuafuProvider(transport=transport)
    payload = {} if protocol is None else {"quafu_protocol": protocol}
    handle = ProviderTaskHandle("quafu", "old-job", "Baihua", payload)

    with pytest.raises(ValidationError, match="retired transport protocol"):
        provider.query_status(handle)
    with pytest.raises(ValidationError, match="retired transport protocol"):
        provider.cancel(handle)
    with pytest.raises(ValidationError, match="retired transport protocol"):
        provider.fetch_result(handle)

    assert transport.posts == [] and transport.gets == []
