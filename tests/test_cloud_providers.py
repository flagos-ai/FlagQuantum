"""Tests for quantum cloud provider adapters."""

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.deployment import CloudBackendProfile
from flagquantum.remote import (
    HttpQuantumProvider,
    ProviderCredentials,
    QuafuProvider,
)


class FakeTransport:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.forms = []

    def post_json(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        return {"task_id": "job-1", "status": "submitted"}

    def get_json(self, url, headers, timeout):
        self.gets.append((url, headers, timeout))
        if url.endswith("/backends"):
            return {
                "backends": [
                    {
                        "name": "fake_chip",
                        "n_wires": 5,
                        "basis_gates": ["rx", "ry", "rz", "cx"],
                    }
                ]
            }
        if url.endswith("/status"):
            return {"status": "Finished"}
        if url.endswith("/result"):
            return {"counts": {"00": 3, "11": 5}}
        if "/task/status/" in url:
            return {"status": "finished"}
        if "/task/result/" in url:
            return {"status": "finished", "result": {"counts": {"00": 3, "11": 5}}}
        return {}

    def post_form(self, url, payload, headers, timeout):
        self.forms.append((url, payload, headers, timeout))
        return {"code": 0, "data": {"access_token": "access-1"}}


def _package(provider, n_wires=2, *, metadata=None):
    circuit = fq.Circuit(n_wires)
    circuit.h(0).cx(0, 1)
    backend = CloudBackendProfile(provider=provider, name="chip", n_wires=8)
    if provider == "quafu" and metadata is None:
        metadata = {"provider_options": {"compiler": None, "target_qubits": [3, 4]}}
    return fqd.create_deployment_package(
        circuit, backend=backend, shots=8, metadata=metadata
    )


def _assert_identity_chain(package, result):
    for key in (
        "deployment_receipt_schema",
        "deployment_package_schema",
        "deployment_program_format",
        "routing_evidence_sha256",
        "deployment_artifact_sha256",
    ):
        assert result.handle.payload[key] == result.metadata[key]
    assert (
        result.handle.payload["deployment_artifact_sha256"]
        == package.metadata["deployment_artifact_sha256"]
    )


def test_http_provider_submit_status_and_result():
    transport = FakeTransport()
    provider = HttpQuantumProvider(
        provider="fake",
        base_url="https://example.test",
        credentials=ProviderCredentials(token="secret"),
        transport=transport,
    )
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    backend = provider.discover_backends(2)[0]
    package = fqd.create_deployment_package(circuit, backend=backend, shots=8)

    result = provider.run(package)

    assert result.handle.provider == "fake"
    assert result.counts == {"00": 3, "11": 5}
    submit_payload = transport.posts[0][1]
    assert submit_payload["qasm"].startswith("OPENQASM")
    assert submit_payload["shots"] == 8
    assert submit_payload["schema"] == package.metadata["deployment_package_schema"]
    assert (
        submit_payload["routing_evidence_sha256"]
        == package.metadata["routing_evidence_sha256"]
        == result.handle.payload["routing_evidence_sha256"]
        == result.metadata["routing_evidence_sha256"]
    )
    assert (
        submit_payload["deployment_artifact_sha256"]
        == package.metadata["deployment_artifact_sha256"]
        == result.handle.payload["deployment_artifact_sha256"]
        == result.metadata["deployment_artifact_sha256"]
    )
    assert transport.posts[0][2]["Authorization"] == "Bearer secret"


def test_http_provider_discovery_fallback_without_network():
    class FailingTransport:
        def get_json(self, url, headers, timeout):
            raise RuntimeError("offline")

        def post_json(self, url, payload, headers, timeout):
            raise RuntimeError("offline")

        def post_form(self, url, payload, headers, timeout):
            raise RuntimeError("offline")

    provider = HttpQuantumProvider(
        provider="offline",
        base_url="https://offline.test",
        transport=FailingTransport(),
        default_backend="offline_sim",
    )

    backends = provider.discover_backends(3)

    assert backends[0].provider == "offline"
    assert backends[0].name == "offline_sim"
    assert backends[0].n_wires == 3


def test_quafu_provider_uses_platform_endpoint_and_token_header():
    transport = FakeTransport()
    provider = QuafuProvider(
        base_url="https://quafu.test", token="secret", transport=transport
    )
    package = _package("quafu")

    handle = provider.submit(package)

    assert handle.task_id == "job-1"
    url, payload, headers, _ = transport.posts[0]
    assert url.startswith("https://quafu.test/task/run/?")
    assert "chip=chip" in url
    assert payload["circuit"].startswith("OPENQASM")
    assert "compile" not in payload
    assert payload["options"] == {
        "compiler": None,
        "target_qubits": [3, 4],
    }
    assert headers == {"token": "secret"}


def test_quafu_provider_preserves_returned_bit_order_by_default():
    transport = FakeTransport()
    provider = QuafuProvider(base_url="https://quafu.test", transport=transport)
    package = _package("quafu")
    handle = provider.submit(package)

    result = provider.fetch_result(handle)

    assert result.counts == {"00": 3, "11": 5}
    _assert_identity_chain(package, result)


def test_quafu_provider_can_reverse_result_bits_for_legacy_consumers():
    class AsymmetricResultTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            if "/task/result/" in url:
                return {"status": "Finished", "count": {"01": 3, "10": 5}}
            return super().get_json(url, headers, timeout)

    provider = QuafuProvider(
        base_url="https://quafu.test",
        transport=AsymmetricResultTransport(),
        reverse_result_bits=True,
    )
    package = _package("quafu")

    result = provider.fetch_result(provider.submit(package))

    assert result.counts == {"10": 3, "01": 5}


def test_quafu_provider_uses_official_token_env_and_status_discovery(monkeypatch):
    class StatusTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/task/status/0"):
                return {"Dongling": 0, "Miaofeng": "Offline"}
            return super().get_json(url, headers, timeout)

    monkeypatch.setenv("QUAFU_API_TOKEN", "env-secret")
    transport = StatusTransport()
    provider = QuafuProvider(base_url="https://quafu.test", transport=transport)

    backends = provider.discover_backends(5)

    assert [backend.name for backend in backends] == ["Dongling", "Miaofeng"]
    assert all(backend.n_wires == 5 for backend in backends)
    assert backends[0].metadata["queue_status"] == 0
    assert transport.gets[0][1] == {"token": "env-secret"}


def test_quafu_provider_run_polls_until_finished():
    class PollingTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.statuses = iter(("Queued", "Running", "Finished"))

        def get_json(self, url, headers, timeout):
            if "/task/status/" in url:
                return {"status": next(self.statuses)}
            return super().get_json(url, headers, timeout)

    provider = QuafuProvider(
        base_url="https://quafu.test",
        transport=PollingTransport(),
        poll_interval=0,
        result_timeout=1,
    )
    package = _package("quafu")

    result = provider.run(package)

    assert result.counts == {"00": 3, "11": 5}
    _assert_identity_chain(package, result)


def test_quafu_provider_fetches_verbatim_chip_info():
    class ChipInfoTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/task/backendtest/Baihua1"):
                return {
                    "calibration_time": "2026-08-07 08:32:27",
                    "qubits_info": {"Q1": {}},
                    "couplers_info": {"C0": {}},
                }
            return super().get_json(url, headers, timeout)

    transport = ChipInfoTransport()
    provider = QuafuProvider(
        base_url="https://quafu.test", token="secret", transport=transport
    )

    info = provider.fetch_chip_info("Baihua")

    assert info["calibration_time"] == "2026-08-07 08:32:27"
    assert transport.gets[0][0] == "https://quafu.test/task/backendtest/Baihua1"
    assert transport.gets[0][1] == {"token": "secret"}


def test_quafu_provider_submits_precompiled_logical_qasm_with_mapping():
    transport = FakeTransport()
    provider = QuafuProvider(
        base_url="https://quafu.test", token="secret", transport=transport
    )
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'

    handle = provider.submit_qasm(
        qasm,
        chip="Baihua",
        name="direct",
        shots=1024,
        target_qubits=(3, 4),
    )

    _, payload, headers, _ = transport.posts[0]
    assert payload == {
        "circuit": qasm,
        "options": {"compiler": None, "target_qubits": [3, 4]},
    }
    assert headers == {"token": "secret"}
    assert handle.payload["compiler"] is None
    assert handle.payload["target_qubits"] == [3, 4]
    assert len(handle.payload["submitted_qasm_sha256"]) == 64


def test_quafu_provider_rejects_missing_or_invalid_physical_mapping():
    provider = QuafuProvider(base_url="https://quafu.test", transport=FakeTransport())
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'

    with pytest.raises(ValueError, match="target_qubits"):
        provider.submit(
            _package("quafu", metadata={"provider_options": {"compiler": None}})
        )
    with pytest.raises(ValueError, match="compiler=None"):
        provider.submit(
            _package("quafu", metadata={"provider_options": {"target_qubits": [3, 4]}})
        )
    with pytest.raises(ValueError, match="logical qreg"):
        provider.submit_qasm(
            qasm.replace("q[2]", "q[3]"),
            chip="Baihua",
            name="direct",
            shots=1024,
            target_qubits=(3, 4),
        )
    with pytest.raises(ValueError, match="logical q"):
        provider.submit_qasm(
            qasm + "x q[2];\n",
            chip="Baihua",
            name="direct",
            shots=1024,
            target_qubits=(3, 4),
        )
