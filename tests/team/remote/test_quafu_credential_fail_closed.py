"""A submission with no credential fails before a request is assembled.

The defect this pins: `submit()` chose the legacy SQC route whenever the task
API probe answered `False`, and that probe answers `False` when `api_key` is
unset. So "no credential configured" and "the service is unhealthy" produced the
same branch, and the branch posted to the legacy endpoint with no authentication
header. The platform's rejection was what reached the caller, which describes the
platform's verdict rather than the missing prerequisite, and the request had
already left the machine.

Two entry points reach that endpoint -- `submit()` through `_submit_legacy()` and
`submit_qasm()` directly -- so both are pinned here. The tests assert on the
outbound call list, not only on the exception: a guard that raises *after*
posting would satisfy an exception-only assertion.
"""

from __future__ import annotations

from typing import Any

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.deployment import CloudBackendProfile
from flagquantum.errors import FlagQuantumError, ValidationError
from flagquantum.remote.qpu.quafu import QuafuProvider

pytestmark = pytest.mark.unit


class SpyTransport:
    """Records every outbound request and refuses to answer any of them."""

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
        raise RuntimeError("the spy reached a server; this test expects no request")

    def get_json(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        self.gets.append(url)
        raise RuntimeError("unreachable")


def _provider(transport: SpyTransport, **options: Any) -> QuafuProvider:
    return QuafuProvider(
        base_url="https://legacy.test",
        task_server_url="https://quafu.test/api/v1",
        transport=transport,
        **options,
    )


def _package(*, target: str = "ScQ-P18") -> Any:
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    backend = CloudBackendProfile(provider="quafu", name=target, n_qubits=18)
    metadata: dict[str, Any] = {
        "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
    }
    return fqd.create_deployment_package(
        circuit, backend=backend, shots=1024, metadata=metadata
    )


def test_no_credential_refuses_before_the_legacy_post(monkeypatch) -> None:
    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    transport = SpyTransport()
    provider = _provider(transport)

    with pytest.raises(ValidationError) as error:
        provider.submit(_package())

    assert transport.posts == []
    assert "QUAFU_API_TOKEN" in str(error.value)


def test_no_credential_refuses_the_qasm_entry_point(monkeypatch) -> None:
    """`submit_qasm` posts to the same endpoint; it needs the same guard.

    The proposal that records this defect names only `submit()`. The second entry
    point was reachable with the same empty credential and the same absent
    header, so pinning only the first would leave half the defect live.
    """

    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    transport = SpyTransport()
    provider = _provider(transport)

    with pytest.raises(ValidationError):
        provider.submit_qasm(
            "OPENQASM 2.0;\nqreg q[2];\n",
            chip="ScQ-P18",
            name="job",
            shots=1024,
            target_qubits=[0, 1],
        )

    assert transport.posts == []


def test_the_failure_names_both_credential_sources(monkeypatch) -> None:
    """The message must let a caller act without reading the source."""

    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    provider = _provider(SpyTransport())

    with pytest.raises(ValidationError) as error:
        provider.submit(_package())

    message = str(error.value)
    assert "QUAFU_API_KEY" in message
    assert "QUAFU_API_TOKEN" in message
    assert "No request was sent" in message
    assert error.value.category == "validation"
    assert isinstance(error.value, FlagQuantumError)


def test_a_configured_key_without_a_token_says_which_one_is_missing(
    monkeypatch,
) -> None:
    """`QUAFU_API_KEY` does not authenticate the legacy endpoint.

    A caller who has set one credential needs to be told that the *other* one is
    the one this route needs, not that both are absent.
    """

    monkeypatch.setenv("QUAFU_API_KEY", "qf_test")
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    transport = SpyTransport()
    provider = _provider(transport)

    with pytest.raises(ValidationError) as error:
        provider.submit(_package())

    message = str(error.value)
    assert transport.posts == []
    assert "QUAFU_API_TOKEN" in message
    assert "No request was sent" in message


def test_the_legitimate_fallback_still_posts_with_its_token(monkeypatch) -> None:
    """The guard must not close the route it exists to authenticate.

    A configured token with an unhealthy task API is a genuine fallback, so the
    request must still go out, and it must carry the token header. An
    implementation that simply refused every legacy submission would pass the
    tests above and fail this one.
    """

    monkeypatch.setenv("QUAFU_API_KEY", "qf_test")
    monkeypatch.setenv("QUAFU_API_TOKEN", "legacy-secret")
    transport = SpyTransport()
    provider = _provider(transport)

    with pytest.raises(RuntimeError, match="reached a server"):
        provider.submit(_package())

    assert len(transport.posts) == 1
    url, headers = transport.posts[0]
    assert url.startswith("https://legacy.test/task/run/")
    assert headers["token"] == "legacy-secret"


def test_a_task_api_only_target_blames_the_missing_credential(monkeypatch) -> None:
    """The routing refusal used to name an absent fallback, not the absent key.

    `sim` and the routing targets have no legacy route at all, so the old message
    ("no equivalent legacy SQC fallback is available") described a structural
    fact and hid the actionable one: the task API was probed without a key.
    """

    monkeypatch.delenv("QUAFU_API_KEY", raising=False)
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    transport = SpyTransport()
    provider = _provider(transport)

    with pytest.raises(ValidationError) as error:
        provider.submit(_package(target="sim"))

    message = str(error.value)
    assert transport.posts == []
    assert "QUAFU_API_KEY" in message
    assert "no equivalent legacy SQC fallback is available" not in message
