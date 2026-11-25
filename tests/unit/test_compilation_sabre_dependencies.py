"""Repeated routing trials share dependencies without consuming one another."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _source() -> CircuitIR:
    rng = random.Random(3)
    return CircuitIR(
        6,
        tuple(Instruction("cx", tuple(rng.sample(range(6), 2))) for _ in range(24)),
    )


def test_local_route_does_not_prepare_unused_trials() -> None:
    source = CircuitIR(6, (Instruction("cx", (0, 1)),))
    with patch.object(
        sabre, "_build_sabre_dependencies", wraps=sabre._build_sabre_dependencies
    ) as build:
        assert not sabre.plan_sabre_swaps(source, CouplingMap.line(6)).swaps
    assert build.call_count == 1


def test_five_trials_build_only_two_dependency_graphs() -> None:
    with (
        patch.object(
            sabre, "_build_sabre_dependencies", wraps=sabre._build_sabre_dependencies
        ) as build,
        patch.object(
            sabre, "_plan_sabre_trial", wraps=sabre._plan_sabre_trial
        ) as trial,
    ):
        sabre.plan_sabre_swaps(_source(), CouplingMap.line(6))
    assert trial.call_count == 5
    assert build.call_count == 2
    shared = trial.call_args_list[1].kwargs["_dependencies"]
    assert all(
        call.kwargs["_dependencies"] is shared for call in trial.call_args_list[1:]
    )


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
@pytest.mark.parametrize("layers", [False, True])
def test_reused_dependencies_preserve_each_bounded_and_unbounded_plan(
    seed: int | None, layers: bool
) -> None:
    source = _source()
    device = CouplingMap.line(6)
    dependencies = sabre._build_sabre_dependencies(source.instructions)
    snapshot = tuple(tuple(row) for row in dependencies[0]), dependencies[1]
    for limit in (None, 0, 2, 100):
        shared = sabre._plan_sabre_trial(
            source,
            device,
            tie_rng=None if seed is None else random.Random(seed),
            layer_lookahead=layers,
            committed_swap_limit=limit,
            _dependencies=dependencies,
        )
        rebuilt = sabre._plan_sabre_trial(
            source,
            device,
            tie_rng=None if seed is None else random.Random(seed),
            layer_lookahead=layers,
            committed_swap_limit=limit,
        )
        assert shared == rebuilt
        assert (
            tuple(tuple(row) for row in dependencies[0]),
            dependencies[1],
        ) == snapshot


def test_separate_planning_calls_read_current_classical_metadata() -> None:
    # Metadata mappings are mutable. No prepared graph may outlive one selection.
    source = CircuitIR(
        6,
        (
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "cx", (1, 5), metadata={"conditions": ((0, 1),), "is_dynamic": True}
            ),
            Instruction(
                "measure", (2,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
        ),
    )
    before = sabre.plan_sabre_swaps(source, CouplingMap.line(6))
    assert before.sequence.index(1) < before.sequence.index(2)
    stored_metadata = source.instructions[2].metadata
    assert isinstance(stored_metadata, dict)
    stored_metadata["classical_bit"] = 1
    after = sabre.plan_sabre_swaps(source, CouplingMap.line(6))
    assert after.sequence.index(2) < after.sequence.index(1)


def test_duplicate_quantum_edges_and_all_classical_readers_are_preserved() -> None:
    source = CircuitIR(
        6,
        (
            Instruction("cx", (0, 1)),
            Instruction("cx", (0, 1)),
            Instruction(
                "measure", (2,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "x", (3,), metadata={"conditions": ((0, 1),), "is_dynamic": True}
            ),
            Instruction(
                "x", (4,), metadata={"conditions": ((0, 1),), "is_dynamic": True}
            ),
            Instruction(
                "measure", (5,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
        ),
    )
    successors, pending = sabre._build_sabre_dependencies(source.instructions)
    assert successors[0] == [1, 1]
    assert pending[1] == 2
    assert successors[2] == [3, 4, 5]
    assert successors[3] == successors[4] == [5]
    assert pending[5] == 3
