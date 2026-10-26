"""Tests for generic cloud providers and Quafu's retained read-only bridge."""

from types import SimpleNamespace

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote import HttpQuantumProvider, ProviderCredentials, QuafuProvider
from flagquantum.remote.qpu import _quafu_sdk
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


class QuafuSdkReadClient:
    def devices(self):
        return [{"name": "Baihua", "kind": "real", "n_qubits": 156}]

    def metadata(self, target):
        return {
            "n_qubits": 156,
            "basis_gates": ["h", "rx", "ry", "rz", "cz"],
            "limits": {"max_shots": 8192},
            "calibration_id": "cal-1",
        }

    def device_calibration(self, device):
        return SimpleNamespace(raw=self._calibration(device))

    def calibration(self, calibration_id):
        payload = self._calibration("Baihua")
        payload["id"] = calibration_id
        return SimpleNamespace(raw=payload)

    @staticmethod
    def _calibration(device):
        return {
            "id": "cal-1",
            "chip": device,
            "calibrated_at": "2026-10-25T10:00:00+08:00",
            "qubits": [{"index": 0}, {"index": 1}],
            "couplers": [{"qubits": [0, 1], "cz_fidelity": 0.99}],
        }


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


def test_quafu_injected_transport_cannot_replace_sdk_read_paths() -> None:
    transport = FakeTransport()
    adapter = _quafu_sdk._QuafuSdkAdapter.for_discovery(client=QuafuSdkReadClient())
    provider = QuafuProvider(transport=transport, _sdk_adapter=adapter)

    backends = provider.list_devices(n_qubits=5)
    calibration = provider.fetch_calibration("Baihua")
    topology = provider._qsteed_chip_info("Baihua")

    assert [(backend.name, backend.n_qubits) for backend in backends] == [
        ("Baihua", 156)
    ]
    assert calibration["id"] == "cal-1"
    assert topology["couplers_info"]["C0"]["fidelity"] == 0.99
    assert transport.gets == []


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
