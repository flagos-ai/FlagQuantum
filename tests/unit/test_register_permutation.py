"""Literal cycle source streams and complete coherent permutation semantics."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.permutation import (
    permutation_cycle_ir,
    permutation_ir,
)
from flagquantum.compiler.registers.permutation_resource_model import (
    source_permutation_cost,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (1, 1, (0,)),
    (1, 2, (0, 1)),
    (2, 4, (1,)),
    (2, 3, (0, 2)),
    (2, 4, (0, 1, 3)),
    (3, 6, (0, 2, 5)),
    (3, 8, (1, 6)),
    (5, 32, (1, 17, 31)),
    (8, 256, (3, 70, 131, 255)),
]


def layout(n: int) -> tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...]]:
    return tuple(range(n)), n, tuple(range(n + 1, 2 * n)), tuple(range(40, 40 + n - 1))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,size,cycle", CASES)
@pytest.mark.parametrize("composed", [False, True])
def test_cycles_source_gates_feedback_and_model(
    n: int, size: int, cycle: tuple[int, ...], composed: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, marker, carries, bits = layout(n)
    cycles = (cycle, cycle[::-1]) if composed else (cycle,)
    source = (
        Permutation(size, cycles, bitsize=n)
        if composed
        else PermutationCycle(size, cycle, bitsize=n)
    )
    gates = (
        permutation_ir(target, cycles, marker, carries, bits, size=size)
        if composed
        else permutation_cycle_ir(target, cycle, marker, carries, bits, size=size)
    )
    # The source DAG delays marker allocation until the first comparison copy.
    # Give each equality its explicit scratch lifetime, reusing caller wire
    # names only after the prior equality erased them. No source gates change.
    allocation_plan = tuple(
        wire
        for c in cycles
        for i in range(len(c) + 1)
        for wire in (*carries, *((marker,) if i == 0 else ()))
    )
    expected = reference_arithmetic_gates(
        source,
        {"x": target},
        allocation_plan,
        dict(zip(carries, bits, strict=True)),
        reuse_allocations=False,
    )
    assert_source(gates, expected)
    equalities = sum(len(c) + 1 for c in cycles)
    controlled_flips = sum(
        (c[i] ^ c[(i + 1) % len(c)]).bit_count() for c in cycles for i in range(len(c))
    )
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    estimated = estimate_component_resources(source_permutation_cost(size, n, cycles))
    assert {
        key: value for key, value in asdict(estimated.gates).items() if value
    } == model.asdict()
    assert estimated.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert (
        counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1) * equalities
    )
    assert counts["measure"] == model.measurement == (n - 1) * equalities
    assert counts["cx"] == (6 * (n - 1) + 1) * equalities + controlled_flips


@pytest.mark.parametrize(
    "n,size,cycles",
    [
        (1, 1, ((0,),)),
        (1, 2, ((0, 1),)),
        (2, 4, ((2,),)),
        (2, 3, ((0, 2),)),
        (2, 4, ((0, 1, 3),)),
        (2, 4, ((0, 1), (1, 2))),
        (3, 8, ((1, 6),)),
    ],
)
def test_all_coherent_inputs_spectator_work_and_every_measurement_event(
    n: int, size: int, cycles: tuple[tuple[int, ...], ...]
) -> None:
    target, marker, carries, bits = layout(n)
    width = 2 * n + 1  # Last physical wire is an entangled spectator.
    data = torch.arange(1 << (n + 1), dtype=torch.int64)
    x, spectator = data >> 1, data & 1
    mapped = x.clone()
    for cycle in cycles:
        prior = mapped.clone()
        for i, value in enumerate(cycle):
            mapped[prior == value] = cycle[(i + 1) % len(cycle)]
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[(x << (n + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(mapped << (n + 1)) | spectator] = amps
    gates = permutation_ir(target, cycles, marker, carries, bits, size=size)
    events = sum(g.name == "measure" for g in gates)
    # Classical slots are overwritten. Enumerate independent events, not just
    # the final n-1 stored bits; this includes every earlier feedback branch.
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
    inverse = tuple(c[::-1] for c in reversed(cycles))
    restored = permutation_ir(target, inverse, marker, carries, bits, size=size)
    # The original compute is not a static Circuit: its measured Ands must be
    # checked with full classical event history, including inverse feedback.
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(expected, restored, outcomes)
        torch.testing.assert_close(
            actual, state / 2 ** (events / 2), rtol=0, atol=1e-12
        )


def test_permutation_validation() -> None:
    x, marker, carry, bits = layout(2)
    for size in (0, 5, True):
        with pytest.raises(ValueError, match="size"):
            permutation_cycle_ir(x, (0, 1), marker, carry, bits, size=size)
    for cycle in ((), (0, 0), (-1,), (4,), (True,)):
        with pytest.raises(ValueError, match="cycle"):
            permutation_cycle_ir(x, cycle, marker, carry, bits, size=4)
    with pytest.raises(ValueError, match="distinct"):
        permutation_cycle_ir(x, (0, 1), 0, carry, bits, size=4)
    with pytest.raises(ValueError, match="classical"):
        permutation_cycle_ir(x, (0, 1), marker, carry, (), size=4)
    with pytest.raises(ValueError, match="nonempty"):
        permutation_ir(x, (), marker, carry, bits, size=4)


def test_user_append_permutation_example() -> None:
    from examples.algorithms.register_permutation import main

    main()
