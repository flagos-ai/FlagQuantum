"""Tests for quantum cloud provider adapters."""

from urllib.error import HTTPError

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.deployment import CloudBackendProfile
from flagquantum.remote import (
    HttpQuantumProvider,
    ProviderCredentials,
    QuafuProvider,
)
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.integration


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


def _package(provider, n_wires=2, *, name="chip", shots=None, metadata=None):
    circuit = fq.Circuit(n_wires)
    circuit.h(0).cx(0, 1)
    backend = CloudBackendProfile(provider=provider, name=name, n_qubits=8)
    if provider == "quafu" and metadata is None:
        metadata = {"provider_options": {"compiler": None, "target_qubits": [3, 4]}}
    selected_shots = (1024 if provider == "quafu" else 8) if shots is None else shots
    return fqd.create_deployment_package(
        circuit, backend=backend, shots=selected_shots, metadata=metadata
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
    backend = provider.list_devices(n_qubits=2)[0]
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

    backends = provider.list_devices(n_qubits=3)

    assert backends[0].provider == "offline"
    assert backends[0].name == "offline_sim"
    assert backends[0].n_qubits == 3


def test_discover_backends_remains_a_deprecated_list_devices_alias():
    provider = HttpQuantumProvider(
        provider="fake",
        base_url="https://example.test",
        transport=FakeTransport(),
    )

    with pytest.warns(DeprecationWarning, match="list_devices"):
        legacy = provider.discover_backends(2)

    current = provider.list_devices(n_qubits=2)
    assert legacy == current


def test_list_devices_supports_legacy_provider_implementations():
    devices = InMemoryRemoteTarget().list_devices(n_qubits=3)

    assert len(devices) == 1
    assert devices[0].n_qubits == 3


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


def test_quafu_provider_prefers_task_api_and_keeps_lifecycle_on_it():
    class TaskApiTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            self.gets.append((url, headers, timeout))
            if url.endswith("/healthz"):
                return {"healthy": True}
            if url.endswith("/status"):
                return {"status": "Finished"}
            if url.endswith("/results"):
                return {
                    "bit_order": "c0_rightmost",
                    "counts": [{"00": 512, "11": 512}],
                    "shots_returned": [1024],
                }
            raise AssertionError(url)

        def post_json(self, url, payload, headers, timeout):
            self.posts.append((url, payload, headers, timeout))
            if url.endswith("/cancel"):
                return {"status": "Cancelled"}
            return {"job_id": "u-task-1", "status": "Queued"}

    transport = TaskApiTransport()
    provider = QuafuProvider(
        base_url="https://legacy.test",
        task_server_url="https://quafu.test/api/v1",
        api_key="qf_test",
        token="legacy-secret",
        transport=transport,
    )
    package = _package(
        "quafu",
        shots=1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    handle = provider.submit(package)

    assert handle.task_id == "u-task-1"
    assert handle.payload["quafu_protocol"] == "task_api_v1"
    assert transport.gets[0][0] == "https://quafu.test/api/v1/healthz"
    url, payload, headers, _ = transport.posts[0]
    assert url == "https://quafu.test/api/v1/jobs"
    assert payload["mode"] == "single"
    assert payload["target"] == "chip"
    assert payload["circuits"] == [{"qasm": package.qasm}]
    assert len(payload["client_ref"]) == 36
    assert headers == {"Authorization": "Bearer qf_test"}
    assert provider.query_status(handle) == "Finished"
    result = provider.fetch_result(handle)
    assert result.counts == {"00": 512, "11": 512}
    assert result.metadata["quafu_protocol"] == "task_api_v1"
    provider.cancel(handle)
    assert transport.posts[-1][0] == "https://quafu.test/api/v1/jobs/u-task-1/cancel"


def test_quafu_provider_falls_back_when_task_api_health_check_fails():
    class OfflineTaskApiTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                raise OSError("task API unavailable")
            return super().get_json(url, headers, timeout)

    transport = OfflineTaskApiTransport()
    provider = QuafuProvider(
        base_url="https://legacy.test",
        task_server_url="https://quafu.test/api/v1",
        api_key="qf_test",
        token="legacy-secret",
        transport=transport,
    )
    package = _package(
        "quafu",
        shots=1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    handle = provider.submit(package)

    assert handle.payload["quafu_protocol"] == "sqc_legacy"
    assert transport.posts[0][0].startswith("https://legacy.test/task/run/?")
    assert transport.posts[0][2] == {"token": "legacy-secret"}


@pytest.mark.parametrize("target", ["sim", "Baihua-sim", "all-race", "all-redispatch"])
def test_quafu_provider_does_not_legacy_fallback_task_api_only_targets(target):
    class OfflineTaskApiTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                raise OSError("task API unavailable")
            return super().get_json(url, headers, timeout)

    transport = OfflineTaskApiTransport()
    provider = QuafuProvider(
        base_url="https://legacy.test",
        task_server_url="https://quafu.test/api/v1",
        api_key="qf_test",
        token="legacy-secret",
        transport=transport,
    )
    package = _package(
        "quafu",
        name=target,
        shots=1000 if target == "sim" or target.endswith("-sim") else 1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    with pytest.raises(RuntimeError, match="requires the task API"):
        provider.submit(package)

    assert transport.posts == []


def test_quafu_provider_falls_back_after_definite_task_api_http_error():
    class RejectedTaskApiTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                return {"healthy": True}
            return super().get_json(url, headers, timeout)

        def post_json(self, url, payload, headers, timeout):
            if url.endswith("/jobs"):
                raise HTTPError(url, 401, "invalid_api_key", {}, None)
            return super().post_json(url, payload, headers, timeout)

    transport = RejectedTaskApiTransport()
    provider = QuafuProvider(
        base_url="https://legacy.test",
        api_key="qf_test",
        token="legacy-secret",
        transport=transport,
    )
    package = _package(
        "quafu",
        shots=1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    handle = provider.submit(package)

    assert handle.payload["quafu_protocol"] == "sqc_legacy"
    assert len(transport.posts) == 1
    assert transport.posts[0][0].startswith("https://legacy.test/task/run/?")


def test_quafu_provider_does_not_fallback_rejected_device_simulator():
    class RejectedTaskApiTransport(FakeTransport):
        def post_json(self, url, payload, headers, timeout):
            self.posts.append((url, payload, headers, timeout))
            raise HTTPError(url, 422, "invalid simulator request", {}, None)

        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                return {"healthy": True}
            return super().get_json(url, headers, timeout)

    transport = RejectedTaskApiTransport()
    provider = QuafuProvider(
        base_url="https://legacy.test",
        task_server_url="https://quafu.test/api/v1",
        api_key="qf_test",
        token="legacy-secret",
        transport=transport,
    )
    package = _package(
        "quafu",
        name="Baihua-sim",
        shots=1000,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    with pytest.raises(HTTPError) as rejected:
        provider.submit(package)

    assert rejected.value.code == 422
    assert [call[0] for call in transport.posts] == ["https://quafu.test/api/v1/jobs"]


def test_quafu_provider_rejects_mapping_for_task_api_only_target():
    provider = QuafuProvider(
        api_key="qf_test",
        token="legacy-secret",
        transport=FakeTransport(),
    )
    package = _package(
        "quafu",
        name="Baihua-sim",
        shots=1000,
        metadata={
            "provider_options": {
                "compiler": "quarkcircuit",
                "target_qubits": [3, 4],
            }
        },
    )

    with pytest.raises(ValueError, match="requires logical-circuit submission"):
        provider.submit(package)


def test_quafu_provider_does_not_duplicate_ambiguous_task_api_submission():
    class AmbiguousTaskApiTransport(FakeTransport):
        def get_json(self, url, headers, timeout):
            if url.endswith("/healthz"):
                return {"healthy": True}
            return super().get_json(url, headers, timeout)

        def post_json(self, url, payload, headers, timeout):
            raise TimeoutError("response lost")

    provider = QuafuProvider(
        base_url="https://legacy.test",
        api_key="qf_test",
        token="legacy-secret",
        transport=AmbiguousTaskApiTransport(),
    )
    package = _package(
        "quafu",
        shots=1024,
        metadata={
            "provider_options": {"compiler": "quarkcircuit", "target_qubits": []}
        },
    )

    with pytest.raises(TimeoutError, match="response lost"):
        provider.submit(package)


def test_quafu_provider_rejects_insecure_task_api_url():
    with pytest.raises(ValueError, match="HTTPS"):
        QuafuProvider(task_server_url="http://quafu.test/api/v1")


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

    backends = provider.list_devices(n_qubits=5)

    assert [backend.name for backend in backends] == ["Dongling", "Miaofeng"]
    assert all(backend.n_qubits == 5 for backend in backends)
    assert backends[0].metadata["queue_status"] == 0
    assert transport.gets[0][0] == "https://quafu.com.cn/api/v1/devices"
    assert transport.gets[-1][1] == {"token": "env-secret"}


def test_quafu_provider_prefers_task_api_device_discovery():
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
                            "simulator_available": True,
                            "calibration_id": "calibration-1",
                            "calibrated_at": "2026-09-24T09:56:41+08:00",
                            "calibration_source": "live",
                            "data_fetched_at": "2026-09-24T11:56:46+08:00",
                            "live_status_stale": False,
                        }
                    ],
                    "simulator": {
                        "name": "sim",
                        "status": "online",
                        "max_qubits": 24,
                    },
                }
            raise AssertionError(url)

    transport = DeviceTransport()
    provider = QuafuProvider(
        task_server_url="https://quafu.test/api/v1", transport=transport
    )

    backends = provider.list_devices()

    assert [(backend.name, backend.n_qubits) for backend in backends] == [
        ("Shenglian", 84),
        ("sim", 24),
    ]
    assert backends[0].basis_gates == ("h", "rx", "ry", "rz", "cz")
    assert backends[0].metadata == {
        "source": "quafu-task-api-devices",
        "status": "online",
        "queue": 4,
        "simulator_available": True,
        "calibration_id": "calibration-1",
        "calibrated_at": "2026-09-24T09:56:41+08:00",
        "calibration_source": "live",
        "data_fetched_at": "2026-09-24T11:56:46+08:00",
        "live_status_stale": False,
    }
    assert transport.gets == [
        ("https://quafu.test/api/v1/devices", {}, provider.timeout)
    ]


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


def test_quafu_provider_fetches_current_and_historical_task_calibration():
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
        task_server_url="https://quafu.test/api/v1",
        api_key="must-not-be-sent",
        transport=transport,
    )

    current = provider.fetch_calibration("Baihua")
    historical = provider.fetch_calibration("Baihua", calibration_id="cal-1")

    assert current is payload
    assert historical is payload
    assert transport.gets == [
        (
            "https://quafu.test/api/v1/devices/Baihua/calibration",
            {},
            provider.timeout,
        ),
        (
            "https://quafu.test/api/v1/calibrations/cal-1",
            {},
            provider.timeout,
        ),
    ]


def test_quafu_provider_rejects_mismatched_task_calibration():
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
    with pytest.raises(ValueError, match="between 1024 and 8192"):
        provider.submit_qasm(
            qasm,
            chip="Baihua",
            name="direct",
            shots=9216,
            target_qubits=(3, 4),
        )
