"""Verify the public source-to-QASM and source-to-package paths offline."""

import math
import socket

import pytest
import torch

import flagquantum as fq
from examples.synthesis_offline_deployment import build_artifacts
from flagquantum.compiler import CouplingMap
from flagquantum.compute.remote.packages import validate_deployment_package

pytestmark = pytest.mark.integration


def test_qfunc_exports_openqasm_and_builds_an_offline_deployment_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Offline deployment must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    monkeypatch.setattr(socket.socket, "connect", reject_connection)

    qasm, package = build_artifacts()

    assert package.name == "bell"
    assert package.shots == 256
    assert package.backend.provider == "local"
    assert package.backend.name == "offline-line"
    assert package.backend.coupling_map == CouplingMap.line(2)
    assert package.n_qubits == 2
    assert package.qasm_version == 2.0
    assert package.metadata["routing_evidence"]["routing_plan"]["mapping_restored"]
    validate_deployment_package(package)

    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    amplitude = 1 / math.sqrt(2)
    expected = torch.tensor([amplitude, 0, 0, amplitude], dtype=torch.complex128)
    for source, version in ((qasm, 3.0), (package.qasm, 2.0)):
        imported = fq.from_openqasm(source)
        assert imported.version == version
        assert imported.measurement_qubits == (0, 1)
        result = fq.run(imported.to_circuit(), options=options)
        torch.testing.assert_close(
            result.statevector().reshape(-1), expected, atol=1e-12, rtol=0
        )
