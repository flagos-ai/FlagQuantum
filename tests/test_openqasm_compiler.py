from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.openqasm import emit_openqasm

pytestmark = pytest.mark.unit


def test_emit_openqasm_uses_canonical_ir_for_both_versions() -> None:
    circuit = fq.Circuit(3).h(0).cx(0, 1).swap(1, 2)

    qasm2 = emit_openqasm(circuit, version=2.0)
    qasm3 = emit_openqasm(circuit.to_ir(), version=3.0)

    assert qasm2.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";')
    assert qasm3.startswith('OPENQASM 3.0;\ninclude "stdgates.inc";')
    for text in (qasm2, qasm3):
        assert "h q[0];" in text
        assert "cx q[0], q[1];" in text
        assert "swap q[1], q[2];" in text
    assert qasm2.count("measure") == 3
    assert qasm3.count("measure") == 1


def test_version_specific_phase_and_sqrt_x_lowering_is_valid() -> None:
    circuit = fq.Circuit(2).phase(0, theta=0.25).cphase(0, 1, theta=0.5)
    circuit.sx(0).sxdg(1)

    qasm2 = emit_openqasm(circuit, version=2.0)
    qasm3 = emit_openqasm(circuit, version=3.0)

    assert "u1(0.25) q[0];" in qasm2
    assert "cu1(0.5) q[0], q[1];" in qasm2
    assert "sx " not in qasm2 and "sxdg " not in qasm2
    assert "p(0.25) q[0];" in qasm3
    assert "cp(0.5) q[0], q[1];" in qasm3
    assert "sx q[0];" in qasm3
    assert "pow(-1) @ sx q[1];" in qasm3


def test_openqasm3_uses_standard_u_gate_spelling() -> None:
    circuit = fq.Circuit(3)
    circuit.u1(0, theta=0.1)
    circuit.u2(1, phi=0.2, lbd=0.3)
    circuit.u3(2, theta=0.4, phi=0.5, lbd=0.6)

    qasm = emit_openqasm(circuit, version=3.0)

    assert "p(0.1) q[0];" in qasm
    assert f"U({repr(math.pi / 2)}, 0.2, 0.3) q[1];" in qasm
    assert "U(0.4, 0.5, 0.6) q[2];" in qasm


def test_emit_openqasm_canonicalizes_gate_aliases() -> None:
    circuit = fq.Circuit(2).gate("cnot", (0, 1))

    qasm = emit_openqasm(circuit)

    assert "cx q[0], q[1];" in qasm
    assert "cnot" not in qasm


def _interaction_reference(name: str, theta: float) -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128)
    if name == "rxx":
        circuit.h(0).h(1)
    elif name == "ryy":
        circuit.rx(0, theta=math.pi / 2).rx(1, theta=math.pi / 2)
    circuit.cx(0, 1).rz(1, theta=theta).cx(0, 1)
    if name == "rxx":
        circuit.h(0).h(1)
    elif name == "ryy":
        circuit.rx(0, theta=-math.pi / 2).rx(1, theta=-math.pi / 2)
    return circuit


@pytest.mark.parametrize("name", ("rxx", "ryy", "rzz"))
def test_interaction_decompositions_are_exact(name: str) -> None:
    theta = 0.37
    source = fq.Circuit(2, dtype=torch.complex128)
    getattr(source, name)(0, 1, theta=theta)

    torch.testing.assert_close(
        source.state(),
        _interaction_reference(name, theta).state(),
        atol=1e-12,
        rtol=1e-12,
    )
    qasm = emit_openqasm(source, version=3.0)
    assert name not in qasm
    assert f"rz({theta}) q[1];" in qasm
    assert qasm.count("cx q[0], q[1];") == 2


def test_emit_openqasm_requires_bound_finite_parameters() -> None:
    unbound = fq.Circuit(1).rx(0, theta=fq.Parameter("theta"))
    non_finite = fq.Circuit(1).rx(0, theta=float("nan"))

    with pytest.raises(ValueError, match="bind_parameters"):
        emit_openqasm(unbound)
    with pytest.raises(ValueError, match="finite"):
        emit_openqasm(non_finite)


def test_emit_openqasm_rejects_old_device_inputs_and_invalid_versions() -> None:
    class RecordedDevice:
        n_wires = 1
        op_history: list[object] = []

    with pytest.raises(TypeError, match="CircuitIR"):
        emit_openqasm(RecordedDevice())
    with pytest.raises(ValueError, match="2.0 or 3.0"):
        emit_openqasm(fq.Circuit(1), version=1.0)


def test_emit_openqasm_rejects_arbitrary_matrix_gates() -> None:
    circuit = fq.Circuit(1).gate("x", (0,), matrix=torch.eye(2, dtype=torch.complex128))

    with pytest.raises(ValueError, match="arbitrary matrix"):
        emit_openqasm(circuit)


def test_emit_openqasm_is_deterministic() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    assert emit_openqasm(circuit) == emit_openqasm(circuit)
