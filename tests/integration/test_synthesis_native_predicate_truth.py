"""Bounded predicate lowering preserves every basis state and complex phase."""

import cmath
from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum import synth
from flagquantum.compiler._construction.boolean_lowering import (
    BooleanCandidateError,
    lower_boolean_value,
)
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction, capture
from flagquantum.compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_register_program,
)
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


def union(x: int, y: int) -> int:
    return (
        (x == 0)
        | (x == 15)
        | (y == 0)
        | (y == 15)
        | (((x & 14) != 4) & ((x & 14) != 10) & ((y & 14) == 6))
    )


def nonlinear(x: int, y: int) -> int:
    return (x * y) % 16 == 3


def dense_nonlinear(x: int, y: int) -> int:
    return (x * x + y * y < 111) ^ ((x * y) % 16 == 3)


@pytest.mark.parametrize("function", [union, nonlinear])
def test_all_eight_bit_assignments_and_arbitrary_xor(
    function: Callable[[int, int], int],
) -> None:
    graph = capture(function, {"x": 4, "y": 4})
    args = graph.body.blocks[0].arguments
    bindings = {args[0].id: tuple(range(4)), args[1].id: tuple(range(4, 8))}
    candidate = lower_boolean_value(
        (graph,),
        bindings,
        (8,),
        entry=graph.name,
        n_qubits=15,
        workspace=tuple(range(9, 15)),
        workspace_state="fresh_zero",
        predicate_only=True,
    )
    assert len(candidate.circuit.instructions) <= 256
    assert len(candidate.workspace) <= 6
    for x in range(16):
        for y in range(16):
            for marker in (0, 1):
                initial = encode(
                    (x, y, marker), (tuple(range(4)), tuple(range(4, 8)), (8,))
                )
                expected = encode(
                    (x, y, marker ^ function(x, y)),
                    (tuple(range(4)), tuple(range(4, 8)), (8,)),
                )
                # Equality includes every input and every clean work bit.
                assert classical_apply(candidate.circuit, initial) == expected
    with pytest.raises(BooleanCandidateError) as caught:
        lower_boolean_value(
            (graph,),
            bindings,
            (8,),
            entry=graph.name,
            n_qubits=15,
            workspace=tuple(range(9, 15)),
            workspace_state="fresh_zero",
            predicate_only=True,
            max_operations=len(candidate.circuit.instructions) - 1,
        )
    assert caught.value.exhausted


def test_dense_predicate_keeps_the_existing_mapped_gate_ceiling() -> None:
    graph = capture(dense_nonlinear, {"x": 4, "y": 4})
    args = graph.body.blocks[0].arguments
    with pytest.raises(
        BooleanCandidateError, match="Boolean mapped operations"
    ) as caught:
        lower_boolean_value(
            (graph,),
            {args[0].id: tuple(range(4)), args[1].id: tuple(range(4, 8))},
            (8,),
            entry=graph.name,
            n_qubits=15,
            workspace=tuple(range(9, 15)),
            workspace_state="fresh_zero",
            predicate_only=True,
        )
    assert not caught.value.exhausted


def _definition() -> _QuantumFunction:
    return program_definition("""
from flagquantum.synth import R, RZ, X, control, phase
@qfunc
def main(x: Output[QNum[4]], y: Output[QNum[4]], target: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    allocate(target)
    hadamard_transform(x)
    hadamard_transform(y)
    R(0.73, 0, target)
    RZ(0.41, target)
    phase(0.19*x + 0.31*y, 1)
    phase(1, 0.17)
    control(
        (x == 0) | (x == 15) | (y == 0) | (y == 15)
        | ((x[1:4] != 2) & (x[1:4] != 5) & (y[1:4] == 3)),
        lambda: X(target),
    )
""")


def test_complete_portfolio_uses_checked_predicates_within_original_bounds() -> None:
    root, functions = capture_program(_definition())
    result = select_register_program(
        root,
        functions,
        constraints=Constraints(CircuitLimits(max_qubits=16), max_workspace=7),
        budget=ConstructionBudget(qubits=16),
        include_boolean=True,
    )
    assert result.complete and result.selected is not None
    assert all(row.feasible for row in result.rows)
    assert len(result.rows) == 3


def test_complex_state_matches_independent_circuit_and_cleans_workspace() -> None:
    program = synth.synthesize(_definition(), max_qubits=16, max_workspace=7)
    reference = fq.Circuit(9, dtype=torch.complex128)
    for wire in range(8):
        reference.h(wire)
        coefficient = 0.19 if wire < 4 else 0.31
        reference.gate("phase", wire, theta=coefficient * (1 << (3 - wire % 4)))
    reference.rx(8, 0.73).rz(8, 0.41)
    reference.any(0, matrix=cmath.exp(0.17j) * torch.eye(2, dtype=torch.complex128))
    permutation = torch.zeros((512, 512), dtype=torch.complex128)
    for value in range(512):
        x, y = value >> 5, (value >> 1) & 15
        destination = value ^ int(
            x in (0, 15) or y in (0, 15) or (x >> 1 not in (2, 5) and y >> 1 == 3)
        )
        permutation[destination, value] = 1
    reference.any(*range(9), matrix=permutation)
    actual = fq.run(program).statevector().reshape(512, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    with pytest.raises(ValueError, match="operations|budget"):
        synth.synthesize(_definition(), max_operations=resources.n_operations - 1)
