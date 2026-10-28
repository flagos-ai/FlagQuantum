"""End-to-end contract for API Change 098 remaining fixed gates."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.compiler.qir import emit_qir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

_GATES = ("ch", "csx", "csxdg", "dcx", "ecr")
_FIXED_BASIS = ("h", "s", "sdg", "t", "tdg", "x", "cx", "rz")


def test_static_and_dynamic_builders_expose_the_remaining_batch() -> None:
    static = fq.Circuit(2)
    dynamic = DynamicCircuit(2)

    for opcode in _GATES:
        assert getattr(static, opcode)(0, 1) is static
        assert getattr(dynamic, opcode.upper())(0, 1) is dynamic

    dynamic.measure(1, classical_bit=0)
    assert [item.name for item in static.to_ir().instructions] == list(_GATES)
    assert [item.name for item in dynamic.to_ir().instructions] == [
        *_GATES,
        "measure",
    ]


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("opcode", _GATES)
def test_direct_and_decomposed_operators_agree(opcode: str, dtype: torch.dtype) -> None:
    source = getattr(fq.Circuit(2, dtype=dtype), opcode)(0, 1)
    lowered = convert_basis(source, gates=_FIXED_BASIS).program

    assert {item.name for item in lowered.instructions} <= set(_FIXED_BASIS)
    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        get_unitary(lowered),
        get_unitary(source),
        rtol=0.0,
        atol=tolerance,
    )


def test_adjoint_rules_close_over_named_gates_and_ordered_dcx() -> None:
    assert fq.Circuit(2).ch(0, 1).adjoint().to_ir().instructions[0].name == "ch"
    assert fq.Circuit(2).ecr(0, 1).adjoint().to_ir().instructions[0].name == "ecr"
    assert fq.Circuit(2).csx(0, 1).adjoint().to_ir().instructions[0].name == "csxdg"
    assert fq.Circuit(2).csxdg(0, 1).adjoint().to_ir().instructions[0].name == "csx"
    dcx = fq.Circuit(2).dcx(0, 1).adjoint().to_ir().instructions[0]
    assert (dcx.name, dcx.wires) == ("dcx", (1, 0))


@pytest.mark.parametrize("opcode", _GATES)
def test_ir_json_round_trip_preserves_each_named_gate(opcode: str) -> None:
    source = getattr(fq.Circuit(2), opcode)(0, 1).to_ir()
    assert fq.CircuitIR.from_json(source.to_json()) == source


def test_static_exporters_lower_every_remaining_gate() -> None:
    circuit = fq.Circuit(2)
    for opcode in _GATES:
        getattr(circuit, opcode)(0, 1)

    qasm = emit_openqasm(circuit, version=3.0)
    qir = emit_qir(circuit)
    qcis = emit_qcis(circuit)

    assert "cx q[0], q[1];" in qasm
    assert "__quantum__qis__cnot__body" in qir
    assert "CZ Q0 Q1" in qcis
