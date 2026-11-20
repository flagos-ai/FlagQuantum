"""Stored Kaliski paired exchanges retain source roots and exact primitives."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.modular.inversion.inverse_exchange_component import (
    inverse_exchange_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_inverse_exchange import words
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_paired_source_calls_operands_and_models(width: int, adjoint: bool) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep4
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    program = inverse_exchange_program(width, adjoint=adjoint)
    assert verify_program_record(program) == ()
    root = program.functions[0]
    calls = root.blocks[0].operations
    assert len(calls) == 2 and all(op.opcode == "call" for op in calls)
    assert calls[0].callee == calls[1].callee
    assert root.registers is not None
    assert tuple(r.name for r in root.registers) == ("u", "v", "r", "s", "a")
    u, v, r, s = words(width)
    bound = bind_register_program(program, (u, v, r, s, (0,)))
    source = _KaliskiIterationStep4(width)
    if adjoint:
        source = source.adjoint()
    expected = reference_arithmetic_gates(
        source,
        dict(u=u, v=v, r=r, s=s, a=(0,)),
        (),
        {},
        preserve_components=True,
    )
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == (u, v, r, s, (0,))
    assert bound.events == () and bound.classical_outputs == ()
    emitted = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert (
        emitted["t"] + emitted["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == 14 * width
    )
    assert model.total_t_count() == 8 * width
    assert emitted["cx"] == 16 * width and emitted["h"] == 4 * width
    assert emitted["measure"] == model.measurement == 0
    assert get_cost_value(source, QubitCount()) == 4 * width + 1


@pytest.mark.parametrize("width", [1, 2, 3])
def test_append_complex_state_spectator_and_exact_restore(width: int) -> None:
    n_qubits = 4 * width + 2
    data = torch.arange(1 << n_qubits)
    mask = (1 << width) - 1
    control = data >> (4 * width + 1)
    a, b, c, d = ((data >> (1 + i * width)) & mask for i in (3, 2, 1, 0))
    exchanged = (b << (3 * width)) | (a << (2 * width)) | (d << width) | c
    original = (data >> 1) & ((1 << (4 * width)) - 1)
    indices = (
        (control << (4 * width + 1))
        | (torch.where(control.bool(), exchanged, original) << 1)
        | (data & 1)
    )
    initial = torch.complex(data.double() + 1, 3 * data.double() - 5)
    initial /= initial.norm()
    expected = torch.empty_like(initial)
    expected[indices] = initial
    inputs = (*words(width), (0,))
    forward = bind_register_program(inverse_exchange_program(width), inputs)
    inverse = bind_register_program(
        inverse_exchange_program(width, adjoint=True), forward.outputs
    )
    circuit = Circuit(n_qubits, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(n_qubits, forward.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    circuit.append(FieldOperation(n_qubits, inverse.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=1e-12)


@pytest.mark.parametrize("width", [0, -1, True, 2.0])
def test_invalid_width_refused_before_graph(width: int) -> None:
    with pytest.raises(ValueError):
        inverse_exchange_program(width)
