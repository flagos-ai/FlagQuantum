"""End-to-end contract for API Change 098 Batch A1."""

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


def _expected(phase: complex) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, phase, 0], [0, phase, 0, 0], [0, 0, 0, 1]],
        dtype=torch.complex128,
    )


def test_user_builders_expose_iswap_and_its_adjoint() -> None:
    static = fq.Circuit(2, dtype=torch.complex128).iswap(0, 1)
    dynamic = DynamicCircuit(2, dtype=torch.complex128).iswapdg(0, 1).measure(1)

    assert [item.name for item in static.to_ir().instructions] == ["iswap"]
    assert [item.name for item in dynamic.to_ir().instructions] == [
        "iswapdg",
        "measure",
    ]
    assert static.ISWAPDG(0, 1) is static
    assert isinstance(dynamic.ISWAP(0, 1), DynamicCircuit)


def test_iswap_ir_round_trip_preserves_the_named_pair() -> None:
    source = fq.Circuit(2).iswap(0, 1).iswapdg(1, 0).to_ir()
    restored = fq.CircuitIR.from_json(source.to_json())

    assert restored == source
    assert [item.name for item in restored.instructions] == ["iswap", "iswapdg"]
    assert [item.wires for item in restored.instructions] == [(0, 1), (1, 0)]


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_iswap_matrices_and_adjoint_are_exact(dtype: torch.dtype) -> None:
    iswap = fq.Circuit(2, dtype=dtype).iswap(0, 1)
    iswapdg = fq.Circuit(2, dtype=dtype).iswapdg(0, 1)
    expected = _expected(1j).to(dtype=dtype)
    expected_adjoint = _expected(-1j).to(dtype=dtype)

    torch.testing.assert_close(get_unitary(iswap), expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        get_unitary(iswapdg), expected_adjoint, rtol=0.0, atol=0.0
    )
    assert iswap.adjoint().to_ir().instructions[0].name == "iswapdg"
    assert iswapdg.adjoint().to_ir().instructions[0].name == "iswap"
    torch.testing.assert_close(
        get_unitary(iswap.compose(iswap.adjoint())),
        torch.eye(4, dtype=dtype),
        rtol=0.0,
        atol=0.0,
    )


def test_iswap_exactly_lowers_to_the_existing_fixed_basis() -> None:
    basis = ("h", "s", "sdg", "cx")
    for opcode, phase in (("iswap", 1j), ("iswapdg", -1j)):
        source = getattr(fq.Circuit(2, dtype=torch.complex128), opcode)(0, 1)
        lowered = convert_basis(source, gates=basis).program
        assert {item.name for item in lowered.instructions} <= set(basis)
        torch.testing.assert_close(
            get_unitary(lowered), _expected(phase), rtol=0.0, atol=1e-12
        )


def test_iswap_has_qasm_qir_and_qcis_exact_lowerings() -> None:
    circuit = fq.Circuit(2).iswap(0, 1).iswapdg(0, 1)

    qasm = emit_openqasm(circuit, version=3.0)
    assert qasm.count("cz q[0], q[1];") == 2
    assert qasm.count("swap q[0], q[1];") == 2
    assert "s q[0];" in qasm and "sdg q[0];" in qasm

    qir = emit_qir(circuit)
    assert qir.count("__quantum__qis__cz__body") >= 2
    assert qir.count("__quantum__qis__swap__body") >= 2
    assert "__quantum__qis__s__body" in qir
    assert "__quantum__qis__s__adj" in qir

    qcis = emit_qcis(circuit)
    assert "CZ Q0 Q1" in qcis
    assert "RZ Q0 1.5707965" in qcis
    assert "RZ Q0 -1.5707965" in qcis
