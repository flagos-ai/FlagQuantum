"""Commuting trial proposals preserve operators, source values and route limits."""

from __future__ import annotations

import itertools
import random
from unittest.mock import patch

import pytest
import torch

from benchmarks.compiler_commuting_frontier import (
    _MAX_COMMUTING_RUN,
    _ORIGINAL,
    _commuting_dependencies,
    _select,
)
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.optimization.commutation import commute
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from tests.unit.test_native_block_optimization import _operator

pytestmark = pytest.mark.unit


def _reaches(successors: list[list[int]], start: int, goal: int) -> bool:
    remaining = list(successors[start])
    visited: set[int] = set()
    while remaining:
        position = remaining.pop()
        if position == goal:
            return True
        if position not in visited:
            visited.add(position)
            remaining.extend(successors[position])
    return False


def _star() -> CircuitIR:
    return CircuitIR(
        4,
        (
            Instruction("h", (0,)),
            *(
                Instruction("cphase", (0, wire), {"theta": 0.17 * wire})
                for wire in (3, 1, 2)
            ),
            Instruction("h", (0,)),
        ),
    )


def test_every_permitted_star_order_preserves_the_complete_operator() -> None:
    source = _star()
    dependencies = _commuting_dependencies(source.instructions)
    assert dependencies is not None
    successors, pending = dependencies
    assert pending == (0, 1, 1, 1, 3)
    expected = _operator(source)
    accepted_orders = 0
    for order in itertools.permutations(range(5)):
        positions = {position: index for index, position in enumerate(order)}
        if any(
            positions[before] >= positions[after]
            for before, afters in enumerate(successors)
            for after in afters
        ):
            continue
        actual = CircuitIR(4, tuple(source.instructions[index] for index in order))
        torch.testing.assert_close(_operator(actual), expected, atol=1e-12, rtol=0)
        accepted_orders += 1
    assert accepted_orders == 6


@pytest.mark.parametrize("seed", range(6))
def test_relaxed_graph_orders_every_unproven_overlapping_pair(seed: int) -> None:
    rng = random.Random(seed)
    instructions = []
    for index in range(40):
        wires = tuple(rng.sample(range(4), 2))
        instructions.append(Instruction("cphase", wires, {"theta": index / 13}))
        if index % 4 == 0:
            instructions.append(Instruction("rx", (wires[0],), {"theta": 0.31}))
    dependencies = _commuting_dependencies(tuple(instructions))
    assert dependencies is not None
    successors, pending = dependencies
    assert sum(pending) == sum(map(len, successors))
    for left, right in itertools.combinations(range(len(instructions)), 2):
        if not commute(instructions[left], instructions[right]):
            assert _reaches(successors, left, right)


def test_commuting_graph_has_a_linear_bounded_edge_count() -> None:
    instructions = tuple(
        Instruction("cz", (0, 1)) for _ in range(8 * _MAX_COMMUTING_RUN)
    )
    dependencies = _commuting_dependencies(instructions)
    assert dependencies is not None
    successors, pending = dependencies
    assert sum(pending) == sum(map(len, successors))
    assert sum(pending) <= 2 * _MAX_COMMUTING_RUN * len(instructions)
    assert pending[_MAX_COMMUTING_RUN] == 2 * _MAX_COMMUTING_RUN


@pytest.mark.parametrize(
    "boundary",
    [
        Instruction("measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}),
        Instruction("reset", (0,), metadata={"is_dynamic": True}),
        Instruction("barrier", (0,), metadata={"is_dynamic": True}),
        Instruction("h", (0,), metadata={"label": "retained annotation"}),
        Instruction("h", (0,), matrix=torch.eye(2)),
        Instruction("unknown", (0,), matrix=torch.eye(2)),
    ],
)
def test_unmodelled_programs_retain_the_complete_original_path(
    boundary: Instruction,
) -> None:
    assert _commuting_dependencies((*_star().instructions, boundary)) is None


@pytest.mark.parametrize("seed", range(6))
def test_routed_numeric_workflow_preserves_full_operator_and_restoration(
    seed: int,
) -> None:
    rng = random.Random(seed)
    instructions = []
    for index in range(12):
        wires = tuple(rng.sample(range(4), 2))
        instructions.append(Instruction("rzz", wires, {"theta": rng.uniform(-2, 2)}))
        if index % 3 == 0:
            instructions.append(Instruction("h", (wires[0],)))
    source = CircuitIR(4, tuple(instructions))
    coupling = CouplingMap.line(4)
    baseline = sabre.plan_sabre_swaps(source, coupling)
    random_state = random.getstate()
    with patch.object(sabre, "_select_sabre_trials", _select):
        selected = sabre.plan_sabre_swaps(source, coupling)
        assert selected == sabre.plan_sabre_swaps(source, coupling)
        output = route_to_topology(source, coupling, strategy="sabre")
    assert random.getstate() == random_state
    assert sabre._restored_swap_count(selected, coupling) <= sabre._restored_swap_count(
        baseline, coupling
    )
    assert output.metadata["routing"]["mapping_restored"] is True
    assert all(
        len(item.wires) == 1 or coupling.has_edge(*item.wires) for item in output
    )
    torch.testing.assert_close(_operator(output), _operator(source), atol=1e-11, rtol=0)


def test_symbolic_and_trainable_angles_keep_their_original_identity() -> None:
    theta = Parameter("theta")
    trainable = torch.tensor(0.4, requires_grad=True)
    source = CircuitIR(
        4,
        (
            Instruction("rzz", (0, 3), {"theta": theta}),
            Instruction("rzz", (0, 2), {"theta": trainable}),
            Instruction("rzz", (0, 1), {"theta": theta}),
        ),
    )
    assert _commuting_dependencies(source.instructions) is not None
    with patch.object(sabre, "_select_sabre_trials", _select):
        output = route_to_topology(source, CouplingMap.line(4), strategy="sabre")
    angles = [item.params["theta"] for item in output if item.name == "rzz"]
    assert sum(angle is theta for angle in angles) == 2
    assert sum(angle is trainable for angle in angles) == 1
    assert trainable.requires_grad and trainable.grad is None


def test_already_local_program_does_not_add_a_trial() -> None:
    source = CircuitIR(3, (Instruction("cz", (0, 1)), Instruction("cz", (1, 2))))
    with (
        patch.object(sabre, "_select_sabre_trials", _select),
        patch.object(
            sabre, "_plan_sabre_trial", wraps=sabre._plan_sabre_trial
        ) as trial,
    ):
        sabre.plan_sabre_swaps(source, CouplingMap.line(3))
    assert trial.call_count == 1


def test_existing_custom_cost_remains_the_selection_objective() -> None:
    source = _star()
    coupling = CouplingMap.line(4)
    baseline = sabre._plan_sabre_trial(source, coupling)
    assert baseline is not None

    def cost(plan: sabre.SabrePlan) -> int:
        return sabre._restored_swap_count(plan, coupling) + len(plan.swaps)

    incumbent = _ORIGINAL(source, coupling, baseline, cost=cost)
    assert cost(_select(source, coupling, baseline, cost=cost)) <= cost(incumbent)
