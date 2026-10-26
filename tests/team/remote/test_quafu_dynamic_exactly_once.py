"""Mock-only conformance for Quafu dynamic exactly-once submission."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
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
from flagquantum.remote.qpu._quafu_sdk import _QuafuSdkAdapter
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


class _Client:
    def __init__(self, response: Any | Exception) -> None:
        self.response = response
        self.submissions: list[tuple[str, dict[str, Any]]] = []

    def submit(self, qasm: str, **kwargs: Any) -> Any:
        self.submissions.append((qasm, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _adapter(response: Any | Exception) -> tuple[_QuafuSdkAdapter, _Client]:
    client = _Client(response)
    return _QuafuSdkAdapter.for_submission(client=client), client


def test_one_attempt_creates_one_sdk_submission_bound_to_the_package() -> None:
    package = _sealed_package()
    adapter, client = _adapter(SimpleNamespace(id="mock-job-1", n_tasks=1))

    handle = _submit_quafu_dynamic_once(package, adapter=adapter)

    assert len(client.submissions) == 1
    qasm, options = client.submissions[0]
    assert qasm == package.qasm
    assert options == {
        "target": "Baihua",
        "shots": 1024,
        "compiler": "none",
        "name": (
            "fq-dynamic-" + package.metadata["dynamic_remote_package_seal_sha256"]
        ),
    }
    assert handle.task_id == "mock-job-1"
    assert handle.payload["quafu_protocol"] == "quafu_sdk_v1"
    assert handle.payload["target_snapshot_id"] == "2" * 64
    assert handle.payload["calibration_id"] == "fixture-calibration-1"


@pytest.mark.parametrize(
    "response",
    (
        TimeoutError("response lost"),
        SimpleNamespace(id=None, n_tasks=1),
    ),
)
def test_ambiguous_submission_is_never_retried_or_sent_to_legacy(
    response: Any | Exception,
) -> None:
    adapter, client = _adapter(response)

    with pytest.raises(RuntimeError, match="quafu_dynamic_submission_ambiguous"):
        _submit_quafu_dynamic_once(_sealed_package(), adapter=adapter)

    assert len(client.submissions) == 1


def test_mock_result_preserves_identity_shots_and_bit_order() -> None:
    adapter, _ = _adapter(SimpleNamespace(id="mock-job-1", n_tasks=1))
    handle = _submit_quafu_dynamic_once(_sealed_package(), adapter=adapter)

    result = _normalize_quafu_dynamic_mock_result(
        handle,
        {
            "counts": {"00": 768, "01": 256},
            "bit_order": "msb_left",
            "shots_returned": 1024,
            "n_clbits": 2,
            "measurements": None,
        },
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
        ({"counts": {"00": 1024}}, "bit_order"),
        (
            {
                "counts": {"0": 1024},
                "bit_order": "msb_left",
                "shots_returned": 1024,
                "n_clbits": 1,
            },
            "bit_width",
        ),
        (
            {
                "counts": {"00": 1024},
                "bit_order": "msb_left",
                "shots_returned": 1024,
                "n_clbits": 2,
                "measurements": [],
            },
            "mid_circuit_data",
        ),
    ),
)
def test_unconfirmed_result_fields_fail_closed(
    response: dict[str, Any], blocker: str
) -> None:
    adapter, _ = _adapter(SimpleNamespace(id="mock-job-1", n_tasks=1))
    handle = _submit_quafu_dynamic_once(_sealed_package(), adapter=adapter)

    with pytest.raises(RuntimeError, match=blocker):
        _normalize_quafu_dynamic_mock_result(handle, response)


def test_dynamic_submission_contract_is_not_public() -> None:
    assert "submit_quafu_dynamic_once" not in fq.__all__
