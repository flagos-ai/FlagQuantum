"""Canonical Cirq CY import preserves atoms and measured feedback branches."""

import pytest
import torch

from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from tests.unit.test_cirq_feedback_import import compare_every_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("inverse", (False, True))
def test_canonical_cy_retains_atomic_instruction_and_complex_phase(
    inverse: bool,
) -> None:
    import cirq

    from flagquantum.core.ir import Instruction
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from flagquantum.simulation.unitary import get_unitary

    wires = tuple(cirq.LineQubit.range(2))
    operation = cirq.Y.controlled().on(*wires)
    if inverse:
        operation = cirq.inverse(operation)
    result = import_cirq_feedback(cirq.Circuit(operation), qubit_order=wires)
    assert result.ir.instructions == (Instruction("cy", (0, 1)),)
    torch.testing.assert_close(
        get_unitary(result.ir, dtype=torch.complex128),
        torch.as_tensor(cirq.unitary(operation)),
        rtol=0,
        atol=2e-12,
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_canonical_cy_feedback_keeps_every_complex_branch(inverse: bool) -> None:
    import cirq

    control, target, record = cirq.LineQubit.range(3)
    operation = cirq.Y.controlled().on(control, target)
    if inverse:
        operation = cirq.inverse(operation)
    source = cirq.Circuit(
        cirq.H(record),
        cirq.measure(record, key="record"),
        operation.with_classical_controls("record"),
    )
    compare_every_branch(source, (control, target, record))
    result = import_cirq_feedback(source, qubit_order=(control, target, record))
    assert result.ir.instructions[-1].name == "cy"
    assert result.ir.instructions[-1].metadata["conditions"] == ((0, 1),)
