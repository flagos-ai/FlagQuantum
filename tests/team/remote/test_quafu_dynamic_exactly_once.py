"""Mock-only conformance for Quafu dynamic exactly-once submission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.compiler import CouplingMap
from flagquantum.deployment import CloudBackendProfile, DeploymentPackage
from flagquantum.deployment.dynamic_sealing import (
    _validate_dynamic_remote_package_seal,
)
from flagquantum.deployment.routing_evidence import stable_payload_sha256
from flagquantum.dynamic import DynamicCircuit
from flagquantum.remote.qpu.quafu_dynamic_submission import (
    _normalize_quafu_dynamic_mock_result,
    _submit_quafu_dynamic_once,
)
from flagquantum.runtime.dynamic import create_dynamic_deployment_package
from flagquantum.runtime.dynamic.routing import route_dynamic_circuit

pytestmark = pytest.mark.unit


def _sealed_package() -> DeploymentPackage:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.5})
    routed = route_dynamic_circuit(circuit, CouplingMap.line(2))
    backend = CloudBackendProfile(
        provider="quafu",
        name="Baihua",
        n_qubits=2,
        basis_gates=("rx",),
        coupling_map=CouplingMap.line(2),
        supports_dynamic_circuits=True,
        max_classical_bits=1,
        dynamic_dialect="openqasm3",
    )
    package = create_dynamic_deployment_package(routed, backend=backend, shots=1024)
    seal = {
        "schema": "flagquantum_dynamic_remote_package_seal_v1",
        "source_requirement_set_id": "1" * 64,
        "target_snapshot_id": "2" * 64,
        "calibration_id": "fixture-calibration-1",
        "routed_artifact_sha256": package.ir.content_hash,
        "routing_evidence_sha256": package.metadata["routing_evidence_sha256"],
        "provider_legality_identity": "3" * 64,
        "package_artifact_sha256": package.metadata["deployment_artifact_sha256"],
    }
    package = replace(
        package,
        metadata={
            **package.metadata,
            "dynamic_remote_package_seal": seal,
            "dynamic_remote_package_seal_sha256": stable_payload_sha256(seal),
        },
    )
    return _validate_dynamic_remote_package_seal(package)


class _Transport:
    def __init__(self, response: Mapping[str, Any] | Exception) -> None:
        self.response = response
        self.posts: list[tuple[str, Mapping[str, Any], Mapping[str, str], float]] = []

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        self.posts.append((path, payload, headers, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_one_attempt_creates_one_post_bound_to_the_package() -> None:
    package = _sealed_package()
    transport = _Transport({"job_id": "mock-job-1"})

    handle = _submit_quafu_dynamic_once(package, transport=transport)

    assert len(transport.posts) == 1
    path, payload, headers, timeout = transport.posts[0]
    assert path == "/jobs"
    assert headers == {}
    assert timeout == 0.0
    assert payload["client_ref"] == (
        "fq-dynamic-" + package.metadata["dynamic_remote_package_seal_sha256"]
    )
    assert handle.task_id == "mock-job-1"
    assert handle.payload["target_snapshot_id"] == "2" * 64
    assert handle.payload["calibration_id"] == "fixture-calibration-1"


@pytest.mark.parametrize(
    "response, message",
    (
        (TimeoutError("response lost"), "quafu_dynamic_submission_ambiguous"),
        ({"accepted": True}, "missing_job_id"),
    ),
)
def test_ambiguous_submission_is_never_retried_or_sent_to_legacy(
    response: Mapping[str, Any] | Exception,
    message: str,
) -> None:
    transport = _Transport(response)

    with pytest.raises(RuntimeError, match=message):
        _submit_quafu_dynamic_once(_sealed_package(), transport=transport)

    assert len(transport.posts) == 1
    assert transport.posts[0][0] == "/jobs"


def test_mock_result_preserves_identity_shots_and_bit_order() -> None:
    handle = _submit_quafu_dynamic_once(
        _sealed_package(), transport=_Transport({"job_id": "mock-job-1"})
    )

    result = _normalize_quafu_dynamic_mock_result(
        handle,
        {"counts": [{"00": 768, "01": 256}], "bit_order": "msb_left"},
    )

    assert result.counts == {"00": 768, "01": 256}
    assert result.shots == 1024
    assert result.metadata["counts_bit_order"] == "msb_left"
    assert result.metadata["counts_width"] == 2
    seal_id = handle.payload["dynamic_remote_package_seal_sha256"]
    assert result.metadata["dynamic_remote_package_seal_sha256"] == seal_id


@pytest.mark.parametrize(
    "response, blocker",
    (
        ({"counts": [{"0": 1024}]}, "bit_order"),
        ({"counts": [{"0": 1024}], "bit_order": "msb_left"}, "bit_width"),
        (
            {
                "counts": [{"0": 1024}],
                "bit_order": "msb_left",
                "mid_circuit_measurements": [],
            },
            "mid_circuit_data",
        ),
    ),
)
def test_unconfirmed_result_fields_fail_closed(
    response: Mapping[str, Any], blocker: str
) -> None:
    handle = _submit_quafu_dynamic_once(
        _sealed_package(), transport=_Transport({"job_id": "mock-job-1"})
    )

    with pytest.raises(RuntimeError, match=blocker):
        _normalize_quafu_dynamic_mock_result(handle, response)


def test_dynamic_submission_contract_is_not_public() -> None:
    assert "submit_quafu_dynamic_once" not in fq.__all__
