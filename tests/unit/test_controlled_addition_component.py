"""Stored CAdd calls preserve source views, exact adjoints and measured work."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.arithmetic.controlled_add_resource_model import (
    source_controlled_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterType
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


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2), (2, 3), (3, 5), (8, 8)))
@pytest.mark.parametrize("cv", (0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_source_calls_operands_feedback_and_independent_resources(
    kind: str, left: int, right: int, cv: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.bookkeeping import Cast, Join, Split
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    program = controlled_addition_program(
        a_dtype, b_dtype, control_value=cv, adjoint=adjoint
    )
    assert verify_program_record(program) == ()
    source = CAdd(source_a, source_b, cv)
    if adjoint:
        source = source.adjoint()
    # Root calls include every zero-cost view and retain exact SSA predecessors.
    # Add and And internals have independent source-level acceptance evidence.
    terminals = {
        function.name for function in program.functions if function.name != "main"
    }
    pairs = compare_layers(program, source, terminals)
    observed = program_component_call_graph(program, keep=terminals.__contains__)
    nonviews: Counter[object] = Counter()
    for name, count in observed.leaf_counts.items():
        if not isinstance(pairs[name], (Split, Join, Cast)):
            nonviews[pairs[name]] += int(count)
    _, declared = source.call_graph(keep=lambda bloq: bloq in nonviews)
    assert nonviews == Counter(declared)
    logical = 1 + left + right
    order = tuple(reversed(range(logical)))
    ports = {"ctrl": order[:1], "a": order[1 : left + 1], "b": order[left + 1 :]}
    work = tuple(range(logical, logical + left + right - 1))
    bound = bind_register_program(program, tuple(ports.values()), clean=work)
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    expected = reference_arithmetic_gates(source, ports, work, slots)
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == tuple(ports.values())
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    native_model = source_controlled_add_resource_formula(left, right, control_value=cv)
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(native_model, category) == getattr(model, category)
    assert native_model.logical_qubits == logical + len(work)
    assert (
        counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (left + right - 1)
    )
    assert counts["measure"] == counts["reset"] == model.measurement == left + right - 1
    assert counts["x"] == 4 * left * (1 - cv)
    assert get_cost_value(source, QubitCount()) == logical + len(work)
    assert Counter(event.kind for event in bound.events) == {
        "allocate": len(work),
        "free": len(work),
    }
    assert len(set(slots.values())) == len(work)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("left,right", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("cv", (0, 1))
def test_complete_complex_tensors_all_raw_histories_inverse_and_zero_carries(
    kind: str, left: int, right: int, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, left, right)
    logical, work_width = 1 + left + right, left + right - 1
    total = logical + 1 + work_width
    order = tuple(reversed(range(logical)))
    inputs = (order[:1], order[1 : left + 1], order[left + 1 :])
    work = tuple(range(logical + 1, total))
    forward = bind_register_program(
        controlled_addition_program(a_dtype, b_dtype, control_value=cv),
        inputs,
        clean=work,
    )
    backward = bind_register_program(
        controlled_addition_program(a_dtype, b_dtype, control_value=cv, adjoint=True),
        inputs,
        clean=work,
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 5 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << work_width] = amplitudes
    circuit = Circuit(total)
    circuit.append(FieldOperation(total, forward.circuit.instructions))
    restored = CircuitIR.from_json(circuit.to_ir().to_json())
    assert restored.instructions == forward.circuit.instructions
    source = CAdd(source_a, source_b, cv)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    axes = (*order, logical, *work)
    reverse_axes = tuple(axes.index(index) for index in range(total))
    arranged = initial.reshape((2,) * total).permute(axes).reshape(1 << logical, -1)
    expected = (
        (matrix.reshape(1 << logical, 1 << logical) @ arranged)
        .reshape((2,) * total)
        .permute(reverse_axes)
        .flatten()
    )
    expected_inverse = (
        (matrix.reshape(1 << logical, 1 << logical).mH @ arranged)
        .reshape((2,) * total)
        .permute(reverse_axes)
        .flatten()
    )
    histories = tuple(product((0, 1), repeat=work_width))
    scale = 2 ** (work_width / 2)
    for first in histories:
        actual = forced_branch(initial, restored.instructions, first)
        inverse_only = forced_branch(initial, backward.circuit.instructions, first)
        torch.testing.assert_close(actual, expected / scale, rtol=0, atol=3e-12)
        torch.testing.assert_close(
            inverse_only, expected_inverse / scale, rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << work_width)[:, 1:]) == 0
        assert (
            torch.count_nonzero(inverse_only.reshape(-1, 1 << work_width)[:, 1:]) == 0
        )
        for second in histories:
            recovered = forced_branch(actual, backward.circuit.instructions, second)
            torch.testing.assert_close(
                recovered, initial / (2**work_width), rtol=0, atol=3e-12
            )


@pytest.mark.parametrize("kind", CHECKS)
def test_actual_small_source_factory_uses_all_existing_native_check_modes(
    kind: CheckKind,
) -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import _cadd_small

    source = _cadd_small.make()
    ports = {"ctrl": (0,), "a": (1, 2, 3), "b": (4, 5, 6)}
    work = tuple(range(7, 12))
    original = reference_arithmetic_gates(
        source, ports, work, dict(zip(work, range(5), strict=True))
    )

    def lower(program: ProgramRecord) -> CircuitIR:
        return bind_register_program(program, tuple(ports.values()), clean=work).circuit

    example = RegisterExample(
        lambda: controlled_addition_program(RegisterType("uint", 3)),
        lower,
        lambda: Counter(gate.name for gate in original),
    )
    result = check_example(example, kind)
    assert result.status is CheckStatus.PASS, result.message


def test_large_source_factory_native_calls_and_primitive_accounting() -> None:
    from qualtran.bloqs.arithmetic.controlled_addition import _cadd_large
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = _cadd_large.make()
    width = int(source.a_dtype.bitsize)
    assert width == 1000
    program = controlled_addition_program(RegisterType("uint", width))
    logical = 1 + 2 * width
    work = tuple(range(logical, 4 * width))
    bound = bind_register_program(
        program,
        ((0,), tuple(range(1, width + 1)), tuple(range(width + 1, logical))),
        clean=work,
    )
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 7996
    assert counts["measure"] == counts["reset"] == model.measurement == 1999
    assert len(work) == 1999 and bound.circuit.n_wires == 4000
    calls = Counter(op.callee for op in program.functions[0].blocks[0].operations)
    assert calls["cadd_and"] == calls["cadd_erase"] == 1000 and calls["cadd_add"] == 1


@pytest.mark.parametrize("cv", (True, 0.0, -1, 2))
def test_invalid_control_values_remain_explicit(cv: object) -> None:
    with pytest.raises(ValueError, match="zero or one"):
        controlled_addition_program(RegisterType("uint", 2), control_value=cv)  # type: ignore[arg-type]


def test_default_target_and_strict_direction() -> None:
    dtype = RegisterType("signed", 2)
    program = controlled_addition_program(dtype)
    entry = program.functions[0]
    assert entry.registers is not None
    assert entry.registers[1].dtype == entry.registers[2].dtype == dtype
    with pytest.raises(ValueError, match="boolean"):
        controlled_addition_program(dtype, adjoint=1)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.parametrize("cv", (0, 1))
def test_existing_append_example_runs_native_calls_with_coherent_control(
    cv: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    from examples.algorithms.register_controlled_addition import ControlledAddition
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    raw = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    initial[::32] = amplitudes
    matrix = torch.as_tensor(
        CAdd(QUInt(2), QUInt(3), cv).tensor_contract(), dtype=torch.complex128
    )
    expected = torch.zeros_like(initial)
    expected[::32] = matrix @ amplitudes
    circuit = DynamicCircuit(11, inputs=initial, dtype=torch.complex128)
    circuit.append(ControlledAddition(cv))
    result = run_dynamic(circuit, shots=4, seed=43, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), rtol=0, atol=3e-12
    )


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery", "mixed"))
@pytest.mark.parametrize("old_value,new_value", tuple(product((0, 1), repeat=2)))
def test_actual_source_outer_control_callback_reuses_shared_native_control_owner(
    kind: str, old_value: int, new_value: int
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.control.specialized_component_control import (
        specialized_component_control_system,
    )
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_component_graph import _native_trace, _source, _source_trace
    from tests.unit.test_component_structured_control import normalize_trace

    a_dtype, b_dtype, source_a, source_b = integer_types(kind, 1, 2)
    positive = controlled_addition_program(a_dtype, b_dtype)
    module, adder = specialized_component_control_system(
        positive, "ctrl", old_value, new_value
    )
    source, callback = CAdd(source_a, source_b, old_value).get_ctrl_system(
        CtrlSpec(cvs=new_value)
    )
    signature = Signature(
        (
            Register("added", quantum_bit_type()),
            *(positive.functions[0].registers or ()),
        )
    )
    native = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {name: native.inputs[name] for name in ("ctrl", "a", "b")}
    original_values = {name: observer.inputs[name] for name in values}
    added, outputs = adder(native, native.inputs["added"], values)
    original_added, original_outputs = callback(
        observer.builder, (observer.inputs["added"],), original_values
    )
    caller = native.finalize(
        dict(zip((r.name for r in signature), (added, *outputs), strict=True))
    )
    composite = observer.builder.finalize(
        **dict(
            zip(
                (r.name for r in signature),
                (*original_added, *original_outputs),
                strict=True,
            )
        )
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    ports = {"added": (0,), "ctrl": (1,), "a": (2,), "b": (3, 4)}
    work = (6, 7, 8)
    bound = bind_register_program(
        caller, tuple(ports.values()), clean=work, entry="caller"
    )
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(composite, ports, work, slots),
    )
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12
    assert counts["measure"] == model.measurement == 3
    assert get_cost_value(source, QubitCount()) == 8
    raw = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[::8] = amplitudes
    matrix = torch.as_tensor(composite.tensor_contract(), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    expected[::8] = (matrix.reshape(32, 32) @ amplitudes.reshape(32, 2)).flatten()
    for history in product((0, 1), repeat=3):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / (8**0.5), rtol=0, atol=3e-12)
        assert torch.count_nonzero(actual.reshape(-1, 8)[:, 1:]) == 0
