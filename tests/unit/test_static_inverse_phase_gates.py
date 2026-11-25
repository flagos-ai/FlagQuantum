"""Approved inverse phase adapters preserve exact atoms and absolute phases."""

from typing import Literal
from unittest.mock import patch

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ("cirq", "pennylane"))
@pytest.mark.parametrize("opcode", ("sdg", "tdg"))
@pytest.mark.parametrize("wire", (0, 1))
def test_static_phase_atoms_complex_operator_and_inverse(
    backend: Literal["cirq", "pennylane"], opcode: str, wire: int
) -> None:
    other = 1 - wire
    gates = (
        Instruction("h", (other,)),
        Instruction("rx", (wire,), params={"theta": 0.37}),
        Instruction(opcode, (wire,)),
    )
    ir = CircuitIR(2, gates)
    if backend == "cirq":
        cirq = pytest.importorskip("cirq")
        from flagquantum.ecosystem.cirq import from_cirq, to_cirq

        exported = to_cirq(ir)
        imported = from_cirq(exported)
        expected = torch.as_tensor(
            exported.unitary(qubit_order=cirq.LineQubit.range(2))
        )
        assert tuple(exported.all_operations())[-1].gate == (
            cirq.S**-1 if opcode == "sdg" else cirq.T**-1
        )
    else:
        qml = pytest.importorskip("pennylane")
        from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane

        exported = to_pennylane(ir)
        imported = from_pennylane(exported)
        expected = torch.as_tensor(qml.matrix(exported, wire_order=(0, 1)))
        assert exported.operations[-1] == qml.adjoint(
            (qml.S if opcode == "sdg" else qml.T)(wires=wire)
        )
    assert imported.instructions == ir.instructions
    native = Circuit(2, dtype=torch.complex128)
    with patch.dict("sys.modules", {"cirq": None, "pennylane": None, "qualtran": None}):
        native.append(FieldOperation(2, imported.instructions))
    torch.testing.assert_close(get_unitary(native), expected, rtol=0, atol=2e-12)
    native.append(FieldOperation(2, (Instruction(opcode[0], (wire,)),)))
    prefix = Circuit(2, dtype=torch.complex128).append(FieldOperation(2, gates[:-1]))
    torch.testing.assert_close(
        get_unitary(native), get_unitary(prefix), rtol=0, atol=2e-12
    )
    assert len(imported.instructions) == 3
    assert not any(g.name in {"measure", "reset"} for g in imported.instructions)


def test_pennylane_rejects_unapproved_adjoint_and_lookalike_forms() -> None:
    qml = pytest.importorskip("pennylane")
    from flagquantum.ecosystem.pennylane import PennyLaneConversionError, from_pennylane

    lookalike_s = type("LookalikeS", (qml.S,), {"name": property(lambda self: "S")})
    operations = (
        qml.adjoint(lookalike_s(wires=0)),
        qml.adjoint(qml.RX(0.2, wires=0)),
        qml.adjoint(qml.adjoint(qml.S(wires=0))),
        qml.pow(qml.S(wires=0), -1),
        qml.ctrl(qml.adjoint(qml.T(wires=1)), control=0),
    )
    for operation in operations:
        with pytest.raises(PennyLaneConversionError, match="unsupported_operation"):
            from_pennylane(qml.tape.QuantumScript([operation]))
