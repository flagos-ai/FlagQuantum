"""Tests for generic cloud providers and Quafu's retained read-only bridge."""

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote import HttpQuantumProvider, ProviderCredentials, QuafuProvider
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.integration


class FakeTransport:
    def __init__(self) -> None:
        self.posts: list[tuple[object, ...]] = []
        self.gets: list[tuple[object, ...]] = []

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
        return {}


def test_http_provider_submit_status_and_result() -> None:
    transport = FakeTransport()
    provider = HttpQuantumProvider(
        provider="fake",
        base_url="https://example.test",
        credentials=ProviderCredentials(token="secret"),
        transport=transport,
    )
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    backend = provider.list_devices(n_qubits=2)[0]
    package = fqd.create_deployment_package(circuit, backend=backend, shots=8)

    result = provider.run(package)

    assert result.handle.provider == "fake"
    assert result.counts == {"00": 3, "11": 5}
    submit_payload = transport.posts[0][1]
    assert submit_payload["qasm"].startswith("OPENQASM")
    assert submit_payload["shots"] == 8
    assert transport.posts[0][2]["Authorization"] == "Bearer secret"


def test_http_provider_discovery_fallback_without_network() -> None:
    class FailingTransport:
        def get_json(self, url, headers, timeout):
            raise RuntimeError("offline")

        def post_json(self, url, payload, headers, timeout):
            raise RuntimeError("offline")

    provider = HttpQuantumProvider(
        provider="offline",
        base_url="https://offline.test",
        transport=FailingTransport(),
        default_backend="offline_sim",
    )

    backends = provider.list_devices(n_qubits=3)

    assert backends[0].name == "offline_sim"
    assert backends[0].n_qubits == 3


def test_discover_backends_remains_a_list_devices_alias() -> None:
    class Provider(InMemoryRemoteTarget):
        def list_devices(self, *, n_qubits=None):
            self.requested_width = n_qubits
            return super().list_devices(n_qubits=n_qubits)

    provider = Provider()
    backends = provider.discover_backends(n_qubits=3)

    assert backends[0].n_qubits == 3


def test_list_devices_supports_legacy_provider_implementations() -> None:
    devices = InMemoryRemoteTarget().list_devices(n_qubits=3)
    assert len(devices) == 1 and devices[0].n_qubits == 3


def test_quafu_injected_transport_can_read_legacy_status(monkeypatch) -> None:
    class StatusTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/task/status/0"):
                return {"Dongling": 0, "Miaofeng": "Offline"}
            raise AssertionError(url)

    monkeypatch.setenv("QUAFU_API_TOKEN", "env-secret")
    transport = StatusTransport()
    provider = QuafuProvider(base_url="https://quafu.test", transport=transport)

    backends = provider.list_devices(n_qubits=5)

    assert [backend.name for backend in backends] == ["Dongling", "Miaofeng"]
    assert all(backend.n_qubits == 5 for backend in backends)
    assert transport.gets[-1][1] == {"token": "env-secret"}


def test_quafu_injected_transport_can_read_task_api_devices() -> None:
    class DeviceTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/devices"):
                return {
                    "devices": [
                        {
                            "name": "Shenglian",
                            "status": "online",
                            "n_qubits": 84,
                            "basis_gates": ["h", "rx", "ry", "rz", "cz"],
                            "queue": 4,
                        }
                    ],
                    "simulator": {"name": "sim", "status": "online", "max_qubits": 24},
                }
            raise AssertionError(url)

    provider = QuafuProvider(
        task_server_url="https://quafu.test/api/v1", transport=DeviceTransport()
    )
    backends = provider.list_devices()

    assert [(backend.name, backend.n_qubits) for backend in backends] == [
        ("Shenglian", 84),
        ("sim", 24),
    ]
    assert backends[0].metadata["source"] == "quafu-task-api-devices"


def test_quafu_provider_fetches_verbatim_legacy_chip_info() -> None:
    class ChipInfoTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/task/backendtest/Baihua1"):
                return {
                    "calibration_time": "2026-08-07 08:32:27",
                    "qubits_info": {"Q1": {}},
                    "couplers_info": {"C0": {}},
                }
            raise AssertionError(url)

    transport = ChipInfoTransport()
    provider = QuafuProvider(
        base_url="https://quafu.test", token="secret", transport=transport
    )

    info = provider.fetch_chip_info("Baihua")

    assert info["calibration_time"] == "2026-08-07 08:32:27"
    assert transport.gets[0][1] == {"token": "secret"}


def test_quafu_injected_transport_fetches_current_and_historical_calibration() -> None:
    payload = {
        "id": "cal-1",
        "chip": "Baihua",
        "calibrated_at": "2026-09-24T14:36:26+08:00",
        "qubits": [{"index": 0}],
        "couplers": [],
    }

    class CalibrationTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            return payload

    transport = CalibrationTransport()
    provider = QuafuProvider(
        task_server_url="https://quafu.test/api/v1", transport=transport
    )

    assert provider.fetch_calibration("Baihua") is payload
    assert provider.fetch_calibration("Baihua", calibration_id="cal-1") is payload
    assert [call[0] for call in transport.gets] == [
        "https://quafu.test/api/v1/devices/Baihua/calibration",
        "https://quafu.test/api/v1/calibrations/cal-1",
    ]


def test_quafu_injected_transport_rejects_mismatched_calibration() -> None:
    class CalibrationTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            return {
                "id": "other-calibration",
                "chip": "Dongling",
                "calibrated_at": "2026-09-24T14:36:29+08:00",
                "qubits": [],
                "couplers": [],
            }

    provider = QuafuProvider(
        task_server_url="https://quafu.test/api/v1",
        transport=CalibrationTransport(),
    )

    with pytest.raises(RuntimeError, match="belongs to 'Dongling'"):
        provider.fetch_calibration("Baihua", calibration_id="cal-1")
