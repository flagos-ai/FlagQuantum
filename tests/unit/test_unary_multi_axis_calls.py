"""Compose source prefix-folded multi-axis traversal through native typed calls."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.unary_component_calls import (
    append_bounded_unary_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_subtraction_foreign_file import native_signature
from tests.unit.test_unary_component_calls import source_slot_names, typed_inverse
from tests.unit.test_unary_iteration_file import source as source
from tests.unit.test_unary_iteration_source_binding import (
    folded_source as folded_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


def multi_axis_program(signature: Signature) -> ProgramRecord:
    """Leaf and nested predicates share the normal Program lifetime owner."""
    builder = ComponentProgramBuilder(signature)
    rows = builder.inputs["row"]
    columns = builder.inputs["column"]
    target_word = builder.inputs["target"]
    assert isinstance(rows, RegisterBinding)
    assert isinstance(columns, RegisterBinding)
    assert isinstance(target_word, RegisterBinding)
    column_bits = tuple(builder.split(columns))
    target = builder.split(target_word)[0]
    control = builder.inputs.get("control")
    assert control is None or isinstance(control, RegisterBinding)
    table = (("cz", "cz", "cx", "cx"), ("cx", "cz", "cz", "cx"))
    visited = []

    def emit_row(row: int, predicate: RegisterBinding) -> RegisterBinding:
        nonlocal column_bits

        def emit_column(column: int, inner: RegisterBinding) -> RegisterBinding:
            nonlocal target
            visited.append((row, column))
            name = table[row][column]
            leaf = bit_gate_function(
                name,
                (Register("bits", quantum_bit_type(), (2,)),),
                CircuitIR(2, (Instruction(name, (0, 1)),)),
            )
            outputs = np.asarray(
                builder.call_program(
                    ProgramRecord((leaf,)),
                    {"bits": np.array([inner, target], dtype=object)},
                    entry=name,
                )["bits"]
            )
            target = outputs[1]
            successor = outputs[0]
            assert isinstance(successor, RegisterBinding)
            return successor

        column_bits, successor = append_bounded_unary_calls(
            builder,
            column_bits,
            4,
            emit_column,
            control=predicate,
            break_early=lambda left, right: row == 0 and right - left == 2,
        )
        assert isinstance(successor, RegisterBinding)
        return successor

    row_bits, control = append_bounded_unary_calls(
        builder, tuple(builder.split(rows)), 2, emit_row, control=control
    )
    assert visited == [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    outputs: dict[str, object] = {
        "row": builder.join(row_bits, rows.register.dtype),
        "column": builder.join(column_bits, columns.register.dtype),
        "target": builder.join((target,), target_word.register.dtype),
    }
    if control is not None:
        outputs["control"] = control
    return builder.finalize(outputs)


@pytest.mark.parametrize("controlled", (False, True))
def test_nested_graph_stream_branches_and_independent_resources(
    folded_source: Any, controlled: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = folded_source(controlled)
    signature = Signature(native_signature(original.signature))
    forward = multi_axis_program(signature)
    logical = 4 + int(controlled)
    work = tuple(range(logical, logical + 2 + int(controlled)))
    inputs = []
    ports = {}
    offset = 0
    for register in signature:
        width = int(register.bitsize)
        wires = tuple(range(offset, offset + width))
        inputs.append(wires)
        ports[register.name] = wires
        offset += width
    for adjoint, record in enumerate((forward, typed_inverse(forward))):
        bound = bind_register_program(record, tuple(inputs), clean=work)
        assert bound.outputs == tuple(inputs)
        direction = original.adjoint() if adjoint else original
        reference = reference_arithmetic_gates(
            direction,
            ports,
            work,
            dict(zip(work, range(len(work)), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, work), reference
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == model.measurement == 4 + int(controlled)
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert get_cost_value(direction, QubitCount()) == logical + len(work)
        events = Counter(event.kind for event in bound.events)
        assert events["allocate"] == events["free"]
        circuit = Circuit(logical + len(work))
        circuit.append(FieldOperation(logical + len(work), bound.circuit.instructions))
        appended = circuit.to_ir()
        assert CircuitIR.from_json(appended.to_json()) == appended
        # Every column proves the complete linear operator in each history.
        # A dense complex superposition additionally checks coherent phase sums.
        states = []
        for basis in range(1 << logical):
            initial = torch.zeros(1 << (logical + len(work)), dtype=torch.complex128)
            initial[basis << len(work)] = torch.tensor(0.6 + 0.8j, dtype=initial.dtype)
            states.append(initial)
        coherent = sum(
            (
                complex(index + 1, 2 - index) * state
                for index, state in enumerate(states)
            ),
            torch.zeros_like(states[0]),
        )
        states.append(coherent / coherent.norm())
        for state in states:
            for history in product((0, 1), repeat=counts["measure"]):
                actual = forced_branch(state, appended.instructions, history)
                expected = forced_branch(state, reference, history)
                torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
                assert (
                    torch.count_nonzero(actual.reshape(-1, 1 << len(work))[:, 1:]) == 0
                )
