"""Native conditional arithmetic retains the complete source component chain."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.arithmetic.conditional_addition_component import (
    conditional_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide, RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_typed_addition_component import integer_types

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", CHECKS)
@pytest.mark.parametrize("signed", (False, True))
def test_complete_eight_bit_concrete_factories_use_existing_five_modes(
    kind: CheckKind, signed: bool
) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        _ctrl_add_or_sub_signed,
        _ctrl_add_or_sub_unsigned,
    )

    source = (_ctrl_add_or_sub_signed if signed else _ctrl_add_or_sub_unsigned).make()
    dtype = RegisterType("signed" if signed else "uint", 8)
    ports = {"ctrl": (0,), "a": tuple(range(1, 9)), "b": tuple(range(9, 17))}
    work = tuple(range(17, 24))
    original = reference_arithmetic_gates(
        source, ports, work, dict(zip(work, range(7), strict=True)), cirq_style=True
    )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(record, tuple(ports.values()), clean=work).circuit

    example = RegisterExample(
        lambda: conditional_addition_program(dtype),
        lower,
        lambda: Counter(gate.name for gate in original),
    )
    result = check_example(example, kind)
    assert result.status is CheckStatus.PASS, result.message


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2), (2, 3), (3, 5), (8, 8)))
@pytest.mark.parametrize("add_on", (False, True))
def test_complete_source_hierarchy_and_original_physical_feedback(
    kind: str, left: int, right: int, add_on: bool
) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    graph = conditional_addition_program(
        a_dtype, b_dtype, add_when_control_is_on=add_on
    )
    assert verify_program_record(graph) == ()
    source = ControlledAddOrSubtract(source_a, source_b, add_when_ctrl_is_on=add_on)
    # Compare every controlled-NOT/OnEach/rebinding level. Add's internals have
    # independent source parity in the typed Add and existing factorized suites.
    terminals = {
        function.name
        for function in graph.functions
        if function.name.endswith("bitwise_controlled_x")
        or function.name in {"conditional_add", "conditional_x"}
        or any(
            register.side != RegisterSide.THRU for register in function.registers or ()
        )
    }
    pairs = compare_layers(graph, source, terminals)
    native = program_component_call_graph(graph, keep=terminals.__contains__)
    kept = {pairs[name] for name in terminals if name in pairs}
    original, original_totals = source.call_graph(keep=kept.__contains__)
    # The source's declared controlled-OnEach recipe omits zero-cost views.
    # Their actual stored occurrences were compared by compare_layers above.
    observed_edges: Counter[tuple[object, object]] = Counter()
    observed_leaves: Counter[object] = Counter()
    for (parent, child), count in native.direct_calls.items():
        if not isinstance(pairs[child], (Split, Join)):
            observed_edges[pairs[parent], pairs[child]] += int(count)
    for name, count in native.leaf_counts.items():
        if not isinstance(pairs[name], (Split, Join)):
            observed_leaves[pairs[name]] += int(count)
    assert observed_edges == Counter(
        {(parent, child): data["n"] for parent, child, data in original.edges.data()}
    )
    assert observed_leaves == Counter(original_totals)
    logical = 1 + left + right
    reversed_data = tuple(reversed(range(logical)))
    ports = {
        "ctrl": reversed_data[:1],
        "a": reversed_data[1 : left + 1],
        "b": reversed_data[left + 1 :],
    }
    work = tuple(range(logical, logical + right - 1))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    expected = reference_arithmetic_gates(source, ports, work, slots, cirq_style=True)
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == tuple(ports.values())
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (right - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == right - 1
    assert get_cost_value(source, QubitCount()) == logical + right - 1
    assert counts["x"] == 2 * int(add_on)
    assert Counter(event.kind for event in bound.events) == (
        {"allocate": right - 1, "free": right - 1} if right > 1 else {}
    )
    if right == 1:
        assert model.clifford == counts["cx"] + counts["x"] - 4


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2), (2, 3)))
@pytest.mark.parametrize("add_on", (False, True))
def test_append_raw_complex_amplitudes_all_histories_and_opposite_polarity_inverse(
    kind: str, left: int, right: int, add_on: bool
) -> None:
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    logical, work_width = 1 + left + right, right - 1
    total = logical + work_width + 1
    order = tuple(reversed(range(logical)))
    inputs = (order[:1], order[1 : left + 1], order[left + 1 :])
    work = tuple(range(logical + 1, total))
    graph = conditional_addition_program(
        a_dtype, b_dtype, add_when_control_is_on=add_on
    )
    inverse = conditional_addition_program(
        a_dtype, b_dtype, add_when_control_is_on=not add_on
    )
    bound = bind_register_program(graph, inputs, clean=work)
    backward = bind_register_program(inverse, inputs, clean=work)
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << work_width] = amplitudes
    circuit = Circuit(total)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    restored_ir = CircuitIR.from_json(circuit.to_ir().to_json())
    assert restored_ir.instructions == bound.circuit.instructions
    source = ControlledAddOrSubtract(source_a, source_b, add_when_ctrl_is_on=add_on)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    axes = (*order, logical, *work)
    reverse_axes = tuple(axes.index(index) for index in range(total))
    expected = matrix.reshape(1 << logical, 1 << logical) @ initial.reshape(
        (2,) * total
    ).permute(axes).reshape(1 << logical, -1)
    expected = expected.reshape((2,) * total).permute(reverse_axes).flatten()
    histories = tuple(product((0, 1), repeat=work_width))
    for first in histories:
        actual = forced_branch(initial, restored_ir.instructions, first)
        torch.testing.assert_close(
            actual, expected / (2 ** (work_width / 2)), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << work_width)[:, 1:]) == 0
        for second in histories:
            recovered = forced_branch(actual, backward.circuit.instructions, second)
            torch.testing.assert_close(
                recovered, initial / (2**work_width), rtol=0, atol=3e-12
            )


def test_default_target_descriptor_and_explicit_boolean_polarity() -> None:
    dtype = RegisterType("signed", 2)
    graph = conditional_addition_program(dtype)
    entry = next(function for function in graph.functions if function.name == "main")
    assert entry.registers is not None
    assert entry.registers[1].dtype == entry.registers[2].dtype == dtype
    with pytest.raises(ValueError, match="boolean"):
        conditional_addition_program(dtype, add_when_control_is_on=1)  # type: ignore[arg-type]


@pytest.mark.integration
def test_native_typed_program_runs_through_dynamic_append_and_inverse() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )

    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    dtype = RegisterType("uint", 2)
    ports = ((0,), (1, 2), (3, 4))
    forward = bind_register_program(
        conditional_addition_program(dtype), ports, clean=(5,)
    )
    backward = bind_register_program(
        conditional_addition_program(dtype, add_when_control_is_on=False),
        ports,
        clean=(5,),
    )
    raw = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = amplitudes
    matrix = torch.as_tensor(
        ControlledAddOrSubtract(QUInt(2)).tensor_contract(), dtype=torch.complex128
    )
    expected = torch.zeros_like(initial)
    expected[::2] = matrix.reshape(32, 32) @ amplitudes
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, forward.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), rtol=0, atol=2e-12
    )
    circuit.append(FieldOperation(6, backward.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), rtol=0, atol=2e-12
    )
