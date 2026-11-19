"""Legal circuits must make progress even when the routing heuristic cycles."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler import compile as compile_circuit
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit

# Reduced from Random(500), 88 CX operations on line(11). The previous planner
# failed after 45 consecutive heuristic swaps. This is a development regression
# case, not a held-out benchmark or a comparison with another SDK.
_CYCLING_PAIRS = (
    (7, 3),
    (3, 4),
    (4, 7),
    (7, 2),
    (9, 7),
    (8, 0),
    (10, 1),
    (7, 1),
    (5, 10),
    (6, 2),
    (4, 5),
    (0, 9),
    (0, 8),
    (0, 8),
    (9, 7),
    (6, 9),
    (7, 8),
    (9, 10),
    (2, 3),
    (5, 0),
    (4, 1),
    (5, 7),
    (0, 5),
    (3, 10),
    (5, 10),
    (0, 3),
    (4, 2),
    (1, 8),
    (7, 8),
)


def _basis_permutation(program: CircuitIR) -> tuple[int, ...]:
    """Compute the complete CX/SWAP operator as a permutation of basis indices."""
    indices = list(range(1 << program.n_wires))
    for instruction in program:
        left, right = instruction.wires
        left_mask = 1 << (program.n_wires - 1 - left)
        right_mask = 1 << (program.n_wires - 1 - right)
        if instruction.name == "cx":
            indices = [
                index ^ (right_mask if index & left_mask else 0) for index in indices
            ]
        elif instruction.name == "swap":
            indices = [
                (
                    index ^ (left_mask | right_mask)
                    if bool(index & left_mask) != bool(index & right_mask)
                    else index
                )
                for index in indices
            ]
        else:
            raise AssertionError(f"Unsupported reference opcode: {instruction.name}")
    return tuple(indices)


def test_repeated_layout_recovers_and_preserves_the_complete_operator() -> None:
    program = CircuitIR(11, tuple(Instruction("cx", pair) for pair in _CYCLING_PAIRS))
    topology = CouplingMap.line(11)
    first = route_to_topology(program, topology, strategy="sabre")
    second = route_to_topology(program, topology, strategy="sabre")
    assert first.instructions == second.instructions
    assert sabre.plan_sabre_swaps(program, topology) == sabre.plan_sabre_swaps(
        program, topology
    )
    assert all(topology.has_edge(*instruction.wires) for instruction in first)
    assert _basis_permutation(first) == _basis_permutation(program)
    assert first.metadata["routing"]["mapping_restored"] is True
    assert sorted(
        item.metadata["source_instruction_index"]
        for item in first
        if item.metadata.get("routing_phase") is None
    ) == list(range(len(program)))
    # Automatic compilation prices SABRE even when another candidate wins. A
    # heuristic cycle must not prevent the ordinary compiler workflow either.
    compiled = compile_circuit(program, coupling_map=topology, optimize=False)
    assert compiled.metadata["routing"]["mapping_restored"] is True
    assert _basis_permutation(compiled) == _basis_permutation(program)


@pytest.mark.parametrize("n_wires", (4, 5, 6, 7))
def test_budget_release_valve_stops_one_edge_short(
    monkeypatch: pytest.MonkeyPatch, n_wires: int
) -> None:
    # Exercise the budget path independently of whether a heuristic happens to
    # cycle, including both odd and even shortest-path lengths.
    monkeypatch.setattr(sabre, "STALLED_SWAP_LIMIT_PER_WIRE", 0)
    program = CircuitIR(n_wires, (Instruction("cx", (0, n_wires - 1)),))
    plan = sabre.plan_sabre_swaps(program, CouplingMap.line(n_wires))
    assert len(plan.swaps) == n_wires - 2
    assert plan.sequence == (0,)
    assert abs(plan.placements[0][0] - plan.placements[0][1]) == 1
    routed = route_to_topology(program, CouplingMap.line(n_wires), strategy="sabre")
    assert _basis_permutation(routed) == _basis_permutation(program)


def test_repeated_search_discards_speculative_swaps() -> None:
    program = CircuitIR(4, (Instruction("cx", (0, 3)),))
    # A tied heuristic repeatedly selects edge (0, 1). The second choice undoes
    # the first. Rollback must remove both speculative SWAPs from the output.
    with patch.object(sabre, "_shifted_distance", return_value=0.0):
        plan = sabre.plan_sabre_swaps(program, CouplingMap.line(4))
    assert len(plan.swaps) == 2
    assert plan.swap_placements == (0, 0)
    assert plan.swap_anchors == (0, 0)
    assert abs(plan.placements[0][0] - plan.placements[0][1]) == 1


@pytest.mark.parametrize("seed", range(8))
def test_rollback_keeps_layout_and_attribution_consistent(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    monkeypatch.setattr(sabre, "STALLED_SWAP_LIMIT_PER_WIRE", 1)
    rng = random.Random(seed)
    program = CircuitIR(
        6, tuple(Instruction("cx", tuple(rng.sample(range(6), 2))) for _ in range(30))
    )
    topology = CouplingMap.grid(2, 3)
    routed = route_to_topology(program, topology, strategy="sabre")
    assert _basis_permutation(routed) == _basis_permutation(program)
    assert all(topology.has_edge(*instruction.wires) for instruction in routed)
    assert routed.metadata["routing"]["planned_inserted_swap_count"] == sum(
        item.metadata.get("routing_phase") is not None for item in routed
    )


def test_progress_preserves_symbolic_parameters_and_classical_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sabre, "STALLED_SWAP_LIMIT_PER_WIRE", 0)
    theta = Parameter("theta")
    program = CircuitIR(
        6,
        (
            Instruction(
                "measure",
                (0,),
                metadata={"classical_bit": 0, "is_dynamic": True},
            ),
            Instruction("cx", (1, 5)),
            Instruction(
                "rx",
                (4,),
                {"theta": theta},
                metadata={"conditions": ((0, 1),), "is_dynamic": True},
            ),
            Instruction("cx", (0, 4)),
        ),
    )
    plan = sabre.plan_sabre_swaps(program, CouplingMap.line(6))
    assert plan.sequence.index(0) < plan.sequence.index(2)
    routed = route_to_topology(program, CouplingMap.line(6), strategy="sabre")
    conditional = next(item for item in routed if item.name == "rx")
    assert conditional.params["theta"] is theta
    assert conditional.metadata["conditions"] == ((0, 1),)


def test_disconnected_coupling_is_still_rejected() -> None:
    program = CircuitIR(4, (Instruction("cx", (0, 3)),))
    with pytest.raises(ValueError, match="no coupling path"):
        sabre.plan_sabre_swaps(program, CouplingMap(4, ((0, 1), (2, 3))))
