"""Keep explicit nonunitary source declarations out of recursive matrix fallback."""

from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("values", tuple(product((0, 1), repeat=3)))
def test_computed_controlled_and_retains_original_full_unitary(
    values: tuple[int, int, int],
) -> None:
    import cirq
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt import And

    cv, cv1, cv2 = values
    source = And(cv1, cv2).controlled(CtrlSpec(cvs=cv))
    operation = source.on(*cirq.LineQubit.range(4))
    expected = torch.tensor(cirq.unitary(source), dtype=torch.complex128)
    program = import_cirq_register_optree(operation)
    bound = bind_register_program(program, ((2,), (0,), (3,), (1,)))
    assert not bound.events
    circuit = Circuit(4)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    axes = (2, 0, 3, 1)
    for index in range(16):
        source_index = sum(
            ((index >> (3 - wire)) & 1) << (3 - offset)
            for offset, wire in enumerate(axes)
        )
        state = torch.eye(16, dtype=torch.complex128)[index]
        actual = forced_branch(state, circuit.to_ir().instructions, ())
        result = torch.zeros_like(state)
        for output in range(16):
            native_index = sum(
                ((output >> (3 - offset)) & 1) << (3 - wire)
                for offset, wire in enumerate(axes)
            )
            result[native_index] = expected[output, source_index]
        torch.testing.assert_close(actual, result, rtol=0, atol=2e-12)


@pytest.mark.parametrize("controls", (1, 2))
@pytest.mark.parametrize("value", (0, 1))
def test_erasing_controlled_and_respects_explicit_nonunitary_source(
    controls: int, value: int
) -> None:
    import cirq
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt import And

    child = And(uncompute=True)
    assert child._has_unitary_() is False
    source = child.controlled(CtrlSpec(cvs=(value,) * controls))
    with pytest.raises(ValueError, match="Source leaf explicitly declares no unitary"):
        import_cirq_register_optree(source.on(*cirq.LineQubit.range(3 + controls)))
