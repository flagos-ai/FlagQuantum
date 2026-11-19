"""Native cycle hierarchy retains immutable source calls and physical effects."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
)
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_cycle_program,
    permutation_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_permutation import CASES
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,size,cycle", CASES)
@pytest.mark.parametrize("composed", (False, True))
def test_cycle_root_source_dependencies_feedback_and_model(
    width: int, size: int, cycle: tuple[int, ...], composed: bool
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    cycles = (cycle, cycle[::-1]) if composed else (cycle,)
    graph = (
        permutation_program(size, cycles, width)
        if composed
        else permutation_cycle_program(size, cycle, width)
    )
    source = (
        Permutation(size, cycles, width)
        if composed
        else PermutationCycle(size, cycle, width)
    )
    compare_layers(graph, source, atomic_leaves(graph))
    bound = bind_register_program(
        graph, (tuple(range(width)),), clean=tuple(range(width, 2 * width))
    )
    marker, carries = width, tuple(range(width + 1, 2 * width))
    allocation_plan = tuple(
        wire
        for c in cycles
        for i in range(len(c) + 1)
        for wire in (*carries, *((marker,) if i == 0 else ()))
    )
    slots = MeasurementSlots(bound.circuit.instructions)
    original = reference_arithmetic_gates(
        source,
        {"x": tuple(range(width))},
        allocation_plan,
        slots,
        reuse_allocations=False,
    )
    assert_source(bound.circuit.instructions, original)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement
    assert get_cost_value(source, QubitCount()) == 2 * width


@pytest.mark.parametrize(
    "width,size,cycles",
    [(1, 2, ((0, 1),)), (2, 3, ((0, 2),)), (2, 4, ((0, 1), (1, 2))), (3, 8, ((2,),))],
)
def test_full_source_tensors_spectator_and_all_raw_histories(
    width: int, size: int, cycles: tuple[tuple[int, ...], ...]
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation

    logical = width + 1
    raw = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + width), dtype=torch.complex128)
    initial[:: 1 << width] = amplitudes
    inputs, work = (tuple(range(width)),), tuple(range(width + 1, 2 * width + 1))
    forward = bind_register_program(
        permutation_program(size, cycles, width), inputs, clean=work
    )
    inverse = tuple(c[::-1] for c in reversed(cycles))
    backward = bind_register_program(
        permutation_program(size, inverse, width), inputs, clean=work
    )
    matrix = torch.as_tensor(
        Permutation(size, cycles, width).tensor_contract(), dtype=torch.complex128
    ).reshape(1 << width, 1 << width)
    expected = torch.zeros_like(initial)
    expected[:: 1 << width] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    events = sum(g.name == "measure" for g in forward.circuit.instructions)
    for history in product((0, 1), repeat=events):
        state = forced_branch(initial, forward.circuit.instructions, history)
        torch.testing.assert_close(
            state, expected / 2 ** (events / 2), rtol=0, atol=3e-12
        )
        residual = state.reshape(-1, 1 << width)[:, 1:]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=3e-12
        )
        restored = forced_branch(expected, backward.circuit.instructions, history)
        torch.testing.assert_close(
            restored, initial / 2 ** (events / 2), rtol=0, atol=3e-12
        )


def test_empty_permutation_is_identity_and_invalid_inputs_fail() -> None:
    graph = permutation_program(3, (), 2)
    bound = bind_register_program(graph, ((0, 1),))
    assert bound.circuit.instructions == ()
    for cycle in ((), (0, 0), (True,), (-1,), (3,)):
        with pytest.raises(ValueError, match="cycle"):
            permutation_cycle_program(3, cycle, 2)
    with pytest.raises(ValueError, match="size"):
        permutation_program(5, (), 2)
    with pytest.raises(ValueError, match="binary"):
        multi_control_x_program((True,))
    with pytest.raises(ValueError, match="constant"):
        equality_constant_program(2, 4)


@pytest.mark.parametrize(
    "factory", ("_permutation_cycle", "_permutation", "_sparse_permutation")
)
@pytest.mark.parametrize("kind", CHECKS)
def test_actual_concrete_factories_use_existing_checks(
    factory: str, kind: CheckKind
) -> None:
    from qualtran.bloqs.arithmetic import permutation as original

    source = getattr(original, factory).make()
    width, size = int(source.bitsize), int(source.N)
    single = hasattr(source, "cycle")
    cycles = (source.cycle,) if single else source.cycles
    graph = (
        permutation_cycle_program(size, cycles[0], width)
        if single
        else permutation_program(size, cycles, width)
    )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(
            record, (tuple(range(width)),), clean=tuple(range(width, 2 * width))
        ).circuit

    bound = lower(graph)
    marker, carries = width, tuple(range(width + 1, 2 * width))
    plan = tuple(
        wire
        for cycle in cycles
        for i in range(len(cycle) + 1)
        for wire in (*carries, *((marker,) if i == 0 else ()))
    )
    expected = reference_arithmetic_gates(
        source,
        {"x": tuple(range(width))},
        plan,
        MeasurementSlots(bound.instructions),
        reuse_allocations=False,
    )
    result = check_example(
        RegisterExample(
            lambda: graph, lower, lambda: Counter(g.name for g in expected)
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


@pytest.mark.integration
def test_normal_append_consumer_preserves_full_coherent_permutation_and_inverse() -> (
    None
):
    from examples.algorithms.register_permutation import main

    main()
