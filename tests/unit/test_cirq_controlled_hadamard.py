"""Approved static CH conversion preserves atomic gates and absolute phase."""

from typing import Any
from unittest.mock import patch

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq import (
    CirqConversionError,
    export_cirq,
    from_cirq,
    import_cirq,
    to_cirq,
)
from flagquantum.simulation.unitary import get_unitary

cirq = pytest.importorskip("cirq")
sympy = pytest.importorskip("sympy")
pytestmark = [pytest.mark.unit, pytest.mark.cirq]


@pytest.mark.parametrize("wires", ((0, 1), (1, 0), (2, 0), (0, 2)))
def test_static_atomic_roundtrip_full_complex_operator_and_inverse(
    wires: tuple[int, int],
) -> None:
    width = max(wires) + 1
    spectator = tuple(q for q in range(width) if q not in wires)
    gates = (
        *(Instruction("rz", (q,), params={"theta": 0.31}) for q in spectator),
        Instruction("ch", wires),
    )
    native = CircuitIR(width, gates)
    exported = export_cirq(native)
    imported = import_cirq(exported.circuit)
    assert exported.report.lossless and imported.report.lossless
    assert imported.ir.instructions == native.instructions
    operations = tuple(exported.circuit.all_operations())
    assert operations[-1].gate == cirq.H.controlled()
    assert operations[-1].qubits == tuple(cirq.LineQubit(q) for q in wires)
    expected = torch.as_tensor(
        exported.circuit.unitary(qubit_order=cirq.LineQubit.range(width)),
        dtype=torch.complex128,
    )
    circuit = Circuit(width, dtype=torch.complex128)
    with patch.dict("sys.modules", {"cirq": None, "qualtran": None}):
        circuit.append(FieldOperation(width, imported.ir.instructions))
    torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(width, (Instruction("ch", wires),)))
    remainder = Circuit(width, dtype=torch.complex128).append(
        FieldOperation(width, gates[:-1])
    )
    torch.testing.assert_close(
        get_unitary(circuit), get_unitary(remainder), rtol=0, atol=2e-12
    )
    assert len(gates) == len(imported.ir.instructions)
    assert sum(g.name == "ch" for g in gates) == 1
    assert not any(g.name in {"measure", "reset"} for g in gates)


@pytest.mark.parametrize(
    "gate",
    (
        cirq.H.controlled(control_values=(0,)),
        cirq.H.controlled(num_controls=2),
        cirq.H.controlled(control_qid_shape=(3,)),
        cirq.HPowGate(exponent=0.5).controlled(),
        cirq.HPowGate(exponent=-1).controlled(),
        cirq.HPowGate(global_shift=1).controlled(),
        cirq.HPowGate(exponent=sympy.Symbol("a")).controlled(),
        cirq.Y.controlled(),
    ),
)
def test_only_canonical_positive_binary_control_is_admitted(gate: Any) -> None:
    qubits = tuple(
        (
            cirq.LineQubit(index)
            if dimension == 2
            else cirq.LineQid(index, dimension=dimension)
        )
        for index, dimension in enumerate(cirq.qid_shape(gate))
    )
    with pytest.raises(CirqConversionError):
        from_cirq(cirq.Circuit(gate.on(*qubits)))


def test_tags_and_sparse_wire_extent_keep_existing_rejections() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    for operation in (
        cirq.H.controlled().on(q0, q1).with_tags("tag"),
        cirq.H.controlled().on(q0, q2),
    ):
        with pytest.raises(CirqConversionError):
            from_cirq(cirq.Circuit(operation))
    with pytest.raises(CirqConversionError, match="idle_wire_extent"):
        to_cirq(CircuitIR(3, (Instruction("ch", (0, 2)),)))
