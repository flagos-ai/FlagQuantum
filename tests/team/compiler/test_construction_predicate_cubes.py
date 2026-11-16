"""Bounded predicate partitions preserve every XOR target and complex amplitude."""

import cmath
from collections.abc import Callable
from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._construction.boolean_lowering import (
    BooleanCandidateError,
    _disjoint_cubes,
    lower_boolean_value,
)
from flagquantum.compiler._construction.frontend_python import capture

pytestmark = pytest.mark.unit


def range_predicate(change: int, fixed: int, u: int) -> int:
    return (u == 3) & (fixed >= 2) & (fixed <= 5) & ((change == 1) | (change == 5))


def split_ranges(change: int, fixed: int, u: int) -> int:
    return (u <= 1) & ((change < 2) | (change >= 6)) & (fixed >= 3)


def signed_cube(change: int, fixed: int, u: int) -> int:
    return (u == 1) & (change == 6) & (fixed == 2)


@pytest.mark.parametrize("truth", range(256))
def test_every_three_bit_truth_table_has_an_exact_disjoint_partition(
    truth: int,
) -> None:
    assignments = tuple(word for word in range(8) if truth >> word & 1)
    cubes = _disjoint_cubes(assignments, 3)
    assert cubes == _disjoint_cubes(tuple(reversed(assignments)), 3)
    for word in range(8):
        matches = sum(word & mask == value for mask, value in cubes)
        assert matches == int(word in assignments)


@pytest.mark.parametrize(
    "function,oracle",
    [
        (range_predicate, lambda a, b, u: int(u == 3 and 2 <= b <= 5 and a in {1, 5})),
        (
            split_ranges,
            lambda a, b, u: int(
                u in {0, 1} and a in {0, 1, 6, 7} and b in {3, 4, 5, 6, 7}
            ),
        ),
        (signed_cube, lambda a, b, u: int((a, b, u) == (6, 2, 1))),
    ],
)
def test_full_eight_input_domain_nonzero_targets_and_coherence(
    function: Callable[[int, int, int], int],
    oracle: Callable[[int, int, int], int],
) -> None:
    graph = capture(function, {"change": 3, "fixed": 3, "u": 2})
    arguments = graph.body.blocks[0].arguments
    bindings = {
        arguments[0].id: (0, 1, 2),
        arguments[1].id: (3, 4, 5),
        arguments[2].id: (6, 7),
    }
    candidate = lower_boolean_value(
        (graph,),
        bindings,
        (8,),
        entry=graph.name,
        n_qubits=15,
        workspace=tuple(range(9, 15)),
        workspace_state="fresh_zero",
    )
    registers = (*bindings.values(), (8,))
    initial_values = [0j] * (1 << 15)
    expected_values = [0j] * (1 << 15)
    for a, b, u, target in product(range(8), range(8), range(4), range(2)):
        initial = encode((a, b, u, target), registers)
        expected = encode((a, b, u, target ^ oracle(a, b, u)), registers)
        assert classical_apply(candidate.circuit, initial) == expected
        amplitude = cmath.exp(1j * (0.13 * a + 0.17 * b + 0.19 * u + 0.23 * target)) / (
            512**0.5
        )
        initial_index = (a << 12) | (b << 9) | (u << 7) | (target << 6)
        expected_index = (
            (a << 12) | (b << 9) | (u << 7) | ((target ^ oracle(a, b, u)) << 6)
        )
        initial_values[initial_index] = amplitude
        expected_values[expected_index] = amplitude
    circuit = fq.Circuit.from_ir(
        candidate.circuit,
        inputs=torch.tensor(initial_values, dtype=torch.complex128),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.tensor(expected_values, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


def test_cube_count_width_and_assignment_limits_refuse() -> None:
    parity = tuple(word for word in range(256) if word.bit_count() % 2)
    with pytest.raises(BooleanCandidateError, match="cube count"):
        _disjoint_cubes(parity, 8)
    with pytest.raises(BooleanCandidateError, match="input width"):
        _disjoint_cubes((0,), 9)
    with pytest.raises(BooleanCandidateError, match="assignment"):
        _disjoint_cubes((8,), 3)


def nine_inputs(a: int, b: int, c: int) -> int:
    return (a == 3) & (b == 4) & (c == 5)


@pytest.mark.parametrize("constraint", ["width", "workspace", "operations"])
def test_predicate_candidate_budgets_refuse(constraint: str) -> None:
    graph = capture(
        nine_inputs if constraint == "width" else range_predicate,
        (
            {"a": 3, "b": 3, "c": 3}
            if constraint == "width"
            else {"change": 3, "fixed": 3, "u": 2}
        ),
    )
    arguments = graph.body.blocks[0].arguments
    last_width = 3 if constraint == "width" else 2
    output = 6 + last_width
    bindings = {
        arguments[0].id: (0, 1, 2),
        arguments[1].id: (3, 4, 5),
        arguments[2].id: tuple(range(6, output)),
    }
    with pytest.raises(
        BooleanCandidateError,
        match={
            "width": "active inputs",
            "workspace": "clean workspace",
            "operations": "operations",
        }[constraint],
    ):
        lower_boolean_value(
            (graph,),
            bindings,
            (output,),
            entry=graph.name,
            n_qubits=output + 7,
            workspace=(
                ()
                if constraint == "workspace"
                else tuple(range(output + 1, output + 7))
            ),
            workspace_state="fresh_zero",
            max_operations=1 if constraint == "operations" else 256,
        )
