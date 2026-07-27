"""Tests for quantum cloud provider adapters."""

import pytest

import flagquantum as fq


class FakeTransport:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.forms = []

    def post_json(self, url, payload, headers, timeout):
        self.posts.append((url, payload, headers, timeout))
        if url.endswith("/task/submit"):
            return {"tasks": [{"id": "tx-1"}]}
        if url.endswith("/task/detail"):
            return {
                "task": {"state": "completed", "result": {"counts": {"00": 3, "11": 5}}}
            }
        if url.endswith("/task/run"):
            return {"task_id": "fq-1", "status": "submitted"}
        if "temporary/save" in url:
            return {"code": 0, "data": {"query_ids": ["cq-1"]}}
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
        if "quantumComputer/list" in url:
            return {"code": 0, "data": [{"machineName": "cqpu", "qubits": 8}]}
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
    backend = fq.CloudBackendProfile(provider=provider, name="chip", n_wires=8)
    return fq.create_deployment_package(
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
    provider = fq.HttpQuantumProvider(
        provider="fake",
        base_url="https://example.test",
        credentials=fq.ProviderCredentials(token="secret"),
        transport=transport,
    )
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    backend = provider.discover_backends(2)[0]
    package = fq.create_deployment_package(circuit, backend=backend, shots=8)

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

    provider = fq.HttpQuantumProvider(
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
    provider = fq.QuafuProvider(
        base_url="https://quafu.test", token="secret", transport=transport
    )
    package = _package("quafu")

    handle = provider.submit(package)

    assert handle.task_id == "job-1"
    url, payload, headers, _ = transport.posts[0]
    assert url.startswith("https://quafu.test/task/run/?")
    assert "chip=chip" in url
    assert payload["circuit"].startswith("OPENQASM")
    assert payload["compile"] is False
    assert headers == {"token": "secret"}


def test_quafu_provider_flips_returned_counts():
    transport = FakeTransport()
    provider = fq.QuafuProvider(base_url="https://quafu.test", transport=transport)
    package = _package("quafu")
    handle = provider.submit(package)

    result = provider.fetch_result(handle)

    assert result.counts == {"00": 3, "11": 5}
    _assert_identity_chain(package, result)


def test_tencent_provider_uses_real_task_contract():
    transport = FakeTransport()
    provider = fq.TencentQuantumProvider(
        base_url="https://tencent.test/cloud/quk", token="secret", transport=transport
    )
    package = _package("tencent")

    result = provider.run(package)

    submit_url, submit_payload, submit_headers, _ = transport.posts[0]
    assert submit_url == "https://tencent.test/cloud/quk/task/submit"
    assert submit_payload["device"] == "chip?o=2"
    assert submit_payload["lang"] == "OPENQASM"
    assert submit_headers["Authorization"] == "Bearer secret"
    assert submit_headers["user-agent"] == "Mozilla/5.0"
    assert result.counts == {"00": 3, "11": 5}
    _assert_identity_chain(package, result)


def test_fieldquantum_provider_uses_sample_mode_contract():
    transport = FakeTransport()
    provider = fq.FieldQuantumProvider(
        base_url="https://field.test", transport=transport
    )
    package = _package("fieldquantum")

    result = provider.run(package)

    assert transport.posts[0][0] == "https://field.test/task/run"
    assert transport.posts[0][1]["mode"] == "sample"
    assert transport.gets[0][0] == "https://field.test/task/status/fq-1"
    assert transport.gets[1][0] == "https://field.test/task/result/fq-1"
    assert result.counts == {"00": 3, "11": 5}
    _assert_identity_chain(package, result)


def test_cqlib_provider_logs_in_discovers_and_requires_qcis_for_submit():
    transport = FakeTransport()
    provider = fq.TianyanProvider(token="open-id", transport=transport)

    backends = provider.discover_backends(2)

    assert transport.forms[0][0] == "https://qc.zdxlz.com/qccp-auth/oauth2/sdk/opnId"
    assert transport.forms[0][1]["grant_type"] == "openId"
    assert backends[0].supports_qcis is True
    assert backends[0].supports_openqasm is False

    package = _package("tianyan")
    with pytest.raises(ValueError, match="QCIS-native"):
        provider.submit(package)


def test_cqlib_provider_submits_qcis_and_extracts_matrix_counts():
    class CqlibTransport(FakeTransport):
        def post_json(self, url, payload, headers, timeout):
            self.posts.append((url, payload, headers, timeout))
            if "temporary/save" in url:
                return {"code": 0, "data": {"query_ids": ["cq-1"]}}
            if "result/find" in url:
                return {
                    "code": 0,
                    "data": {
                        "experimentResultModelList": [
                            {"resultStatus": [["Q0", "Q1"], [0, 0], [1, 1], [1, 1]]}
                        ]
                    },
                }
            return super().post_json(url, payload, headers, timeout)

    transport = CqlibTransport()
    provider = fq.GuodunProvider(token="open-id", transport=transport)
    backend = provider.discover_backends(2)[0]
    package = fq.create_deployment_package(
        fq.Circuit(2).h(0).cx(0, 1),
        backend=backend,
        shots=3,
    )

    result = provider.run(package)

    submit_payload = transport.posts[0][1]
    assert submit_payload["languageCode"] == "qcis"
    assert submit_payload["inputCode"][0].startswith("Y2M Q0")
    assert "CZ Q0 Q1" in submit_payload["inputCode"][0]
    assert transport.posts[0][2]["basicToken"] == "access-1"
    assert result.counts == {"00": 1, "11": 2}


def test_originq_provider_is_sdk_based_not_fake_http():
    provider = fq.OriginQProvider()
    package = _package("originq")

    with pytest.raises(NotImplementedError, match="sdk adapter"):
        provider.submit(package)


def test_named_quantum_cloud_provider_apis_exist():
    providers = [
        fq.QuafuProvider(base_url="https://example.test", transport=FakeTransport()),
        fq.OriginQProvider(),
        fq.TencentQuantumProvider(
            base_url="https://example.test", transport=FakeTransport()
        ),
        fq.TianyanProvider(token="open-id", transport=FakeTransport()),
        fq.GuodunProvider(token="open-id", transport=FakeTransport()),
        fq.FieldQuantumProvider(
            base_url="https://example.test", transport=FakeTransport()
        ),
    ]

    assert [provider.provider for provider in providers] == [
        "quafu",
        "originq",
        "tencent",
        "tianyan",
        "guodun",
        "fieldquantum",
    ]
    assert fq.deployment.QuafuProvider is fq.QuafuProvider
