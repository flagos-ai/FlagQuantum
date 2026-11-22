"""Exact independent permutation and user-workflow checks for linear synthesis."""

from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile
from flagquantum.compiler.optimization import _linear_cnot as linear
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CompilationError

pytestmark = pytest.mark.unit


def _apply(operations: tuple[tuple[int, int], ...], bits: int) -> int:
    for control, target in operations:
        if bits & (1 << control):
            bits ^= 1 << target
    return bits


def test_every_three_wire_binary_matrix_against_its_complete_truth_table() -> None:
    accepted = 0
    for matrix in itertools.product(range(8), repeat=3):
        expected = [
            sum(
                ((row & bits).bit_count() % 2) << index
                for index, row in enumerate(matrix)
            )
            for bits in range(8)
        ]
        if len(set(expected)) < 8:
            with pytest.raises(ValueError, match="invertible"):
                linear._pmh_synthesize(matrix)
        else:
            candidate = linear._pmh_synthesize(matrix)
            assert candidate is not None
            assert [_apply(candidate, bits) for bits in range(8)] == expected
            accepted += 1
    assert accepted == 168


@pytest.mark.parametrize("width", [4, 8, 16, 32, 64])
def test_seeded_wider_networks_against_independent_bit_execution(width: int) -> None:
    rng = random.Random(1729 + width)
    for _ in range(20):
        source = tuple(
            (control, target)
            for control, target in (
                rng.sample(range(width), 2) for _ in range(8 * width)
            )
        )
        candidate = linear._pmh_synthesize(linear._linear_matrix(source, width))
        assert candidate is not None
        for bits in [0, (1 << width) - 1, *(rng.getrandbits(width) for _ in range(30))]:
            assert _apply(source, bits) == _apply(candidate, bits)


@pytest.mark.parametrize("matrix", [(2,), (-1,), (1, 4), (True,)])
def test_invalid_binary_rows_are_refused(matrix: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="binary masks"):
        linear._pmh_synthesize(matrix)


def _relay(wires: tuple[int, int, int] = (0, 1, 2)) -> tuple[Instruction, ...]:
    first, middle, last = wires
    return tuple(
        Instruction("cx", pair)
        for pair in ((first, middle), (middle, last), (first, middle), (middle, last))
    )


def test_multiqubit_relay_reduces_without_relabeling_sparse_or_reversed_wires() -> None:
    for wires in ((0, 1, 2), (2, 1, 0), (7, 1, 4)):
        program = CircuitIR(
            max(wires) + 1, _relay(wires), metadata={"purpose": "relay"}
        )
        result = linear.resynthesize_cnot_blocks(program)
        assert result.instructions == (Instruction("cx", (wires[0], wires[2])),)
        assert result.metadata == program.metadata
        assert result.n_wires == program.n_wires
        assert linear.resynthesize_cnot_blocks(result) is result


@pytest.mark.parametrize(
    "boundary",
    [
        Instruction("rz", (0,), {"theta": 0.23}),
        Instruction("cx", (0, 1), metadata={"source_instruction_index": 3}),
        Instruction("h", (0,), metadata={"barrier": True}),
        Instruction("measure", (2,), metadata={"classical_bit": 0, "is_dynamic": True}),
    ],
)
def test_annotations_and_non_cnot_operations_split_runs(boundary: Instruction) -> None:
    program = CircuitIR(
        3, (Instruction("cx", (0, 1)), boundary, Instruction("cx", (0, 1)))
    )
    assert linear.resynthesize_cnot_blocks(program) is program


def test_mutating_metadata_cannot_reuse_an_earlier_admission() -> None:
    program = CircuitIR(3, _relay())
    assert len(linear.resynthesize_cnot_blocks(program).instructions) == 1
    metadata = program.instructions[1].metadata
    assert isinstance(metadata, dict)
    metadata["conditions"] = ((0, 1),)
    assert linear.resynthesize_cnot_blocks(program) is program


def test_corrupt_shorter_proposal_is_refused() -> None:
    with (
        patch.object(linear, "_pmh_synthesize", return_value=((2, 0),)),
        pytest.raises(CompilationError, match="exact binary"),
    ):
        linear.resynthesize_cnot_blocks(CircuitIR(3, _relay()))


def test_complete_program_depth_guard_retains_incumbent() -> None:
    rng = random.Random(0)
    operations: list[tuple[int, int]] = []
    for _ in range(10):
        order = list(range(8))
        rng.shuffle(order)
        operations.extend((order[i], order[i + 1]) for i in range(0, 8, 2))
    program = CircuitIR(8, tuple(Instruction("cx", pair) for pair in operations))
    candidate = linear._rewrite_block(list(program.instructions))
    assert len(candidate) == 23 < len(program.instructions)
    assert linear._two_body_depth(program.instructions, 8) == 10
    assert linear._two_body_depth(candidate, 8) == 14
    assert linear.resynthesize_cnot_blocks(program) is program


def test_long_runs_are_split_without_losing_operations() -> None:
    program = CircuitIR(3, _relay() * 129)
    result = linear.resynthesize_cnot_blocks(program)
    for bits in range(8):
        expected = _apply(
            tuple((g.wires[0], g.wires[1]) for g in program.instructions), bits
        )
        actual = _apply(
            tuple((g.wires[0], g.wires[1]) for g in result.instructions), bits
        )
        assert actual == expected
    assert len(result.instructions) < len(program.instructions)


def test_existing_pass_hook_preserves_full_operator_and_shared_angle_gradient() -> None:
    for bits in range(8):
        preparation = fq.Circuit(3, dtype=torch.complex128)
        for wire in range(3):
            if bits & (1 << wire):
                preparation.x(wire)
        source = replace(
            preparation.to_ir(),
            instructions=preparation.to_ir().instructions + _relay(),
        )
        result = compile(
            source,
            optimization_level=0,
            passes={"example.linear": linear.resynthesize_cnot_blocks},
        )
        torch.testing.assert_close(
            fq.Circuit.from_ir(result).state(),
            fq.Circuit.from_ir(source).state(),
            rtol=0,
            atol=0,
        )
    angle = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    other = angle.detach().clone().requires_grad_(True)

    def program(theta: torch.Tensor) -> CircuitIR:
        circuit = fq.Circuit(3, dtype=torch.complex128).h(0).ry(1, theta=theta)
        for gate in _relay():
            circuit.cx(*gate.wires)
        return circuit.rz(2, theta=theta).to_ir()

    baseline = fq.Circuit.from_ir(program(angle)).state()
    optimized = fq.Circuit.from_ir(
        compile(
            program(other),
            optimization_level=0,
            passes={"example.linear": linear.resynthesize_cnot_blocks},
        )
    ).state()
    torch.testing.assert_close(optimized, baseline, rtol=0, atol=1e-13)
    weights = torch.arange(8, dtype=torch.float64)
    expected = torch.autograd.grad(
        (baseline.abs().square().flatten() * weights).sum(), angle
    )[0]
    actual = torch.autograd.grad(
        (optimized.abs().square().flatten() * weights).sum(), other
    )[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("pair", [(0, 0), (-1, 1), (0, 2)])
def test_invalid_cnot_operands_are_refused(pair: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="distinct matrix indices"):
        linear._linear_matrix((pair,), 2)


def test_empty_matrix_and_program_remain_empty() -> None:
    assert linear._pmh_synthesize(()) == ()
    source = CircuitIR(3, ())
    assert linear.resynthesize_cnot_blocks(source) is source


def test_shorter_nonlocal_proposal_keeps_the_legal_line_incumbent() -> None:
    source = CircuitIR(3, _relay())
    # Four adjacent relay CXs implement one remote CX. Replacing them by that
    # CX would discard a valid bridge and make later routing do extra work.
    assert len(linear.resynthesize_cnot_blocks(source).instructions) == 1
    assert (
        linear.resynthesize_cnot_blocks(source, coupling_map=CouplingMap.line(3))
        is source
    )
    complete = CouplingMap(3, ((0, 1), (1, 2), (0, 2)))
    accepted = linear.resynthesize_cnot_blocks(source, coupling_map=complete)
    assert accepted.instructions == (Instruction("cx", (0, 2)),)


def test_undersized_topology_is_refused() -> None:
    with pytest.raises(ValueError, match="smaller than the program"):
        linear.resynthesize_cnot_blocks(
            CircuitIR(3, _relay()), coupling_map=CouplingMap.line(2)
        )


def test_default_compiler_does_not_install_the_private_policy() -> None:
    with patch.object(
        linear, "_pmh_synthesize", side_effect=AssertionError("unexpected admission")
    ):
        assert compile(CircuitIR(3, _relay())).n_wires == 3


def test_existing_routed_pass_workflow_preserves_line_evidence() -> None:
    source = CircuitIR(3, _relay())
    topology = CouplingMap.line(3)
    result = compile(
        source,
        coupling_map=topology,
        optimization_level=0,
        passes={
            "example.linear": partial(
                linear.resynthesize_cnot_blocks, coupling_map=topology
            )
        },
    )
    assert result.metadata["routing"]["mapping_restored"] is True
    assert sum(len(item.wires) == 2 for item in result.instructions) == 4
    assert all(topology.has_edge(*item.wires) for item in result.instructions)


@pytest.mark.parametrize("width", [4, 6])
def test_dense_networks_preserve_non_clifford_input_states(width: int) -> None:
    for seed in range(3):
        rng = random.Random(seed)
        circuit = fq.Circuit(width, dtype=torch.complex128)
        for wire in range(width):
            circuit.ry(wire, theta=0.17 * (wire + 1)).rz(wire, theta=-0.23 * wire)
        for _ in range(width * 8):
            control, target = rng.sample(range(width), 2)
            circuit.cx(control, target)
        source = circuit.to_ir()
        result = linear.resynthesize_cnot_blocks(source)
        torch.testing.assert_close(
            fq.Circuit.from_ir(result).state(), circuit.state(), atol=1e-13, rtol=0
        )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_operation_limits_are_refused(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        linear._pmh_synthesize((1, 2), operation_limit=limit)


@pytest.mark.parametrize("width", [3, 8, 16])
def test_emission_budget_preserves_every_strictly_shorter_proposal(width: int) -> None:
    rng = random.Random(width)
    for _ in range(20):
        source = tuple(
            (a, b) for a, b in (rng.sample(range(width), 2) for _ in range(4 * width))
        )
        matrix = linear._linear_matrix(source, width)
        expected = linear._pmh_synthesize(matrix)
        assert expected is not None
        accepted = linear._pmh_synthesize(matrix, operation_limit=len(expected) + 1)
        assert accepted == expected
        if expected:
            assert linear._pmh_synthesize(matrix, operation_limit=len(expected)) is None


def test_exact_search_replaces_pmh_without_changing_the_pass_consumer() -> None:
    identity = tuple(range(8))
    shortest: dict[tuple[int, ...], tuple[tuple[int, int], ...]] = {identity: ()}
    pending = deque([identity])
    while pending:
        permutation = pending.popleft()
        for pair in itertools.permutations(range(3), 2):
            following = tuple(_apply((pair,), bits) for bits in permutation)
            if following not in shortest:
                shortest[following] = shortest[permutation] + (pair,)
                pending.append(following)
    assert len(shortest) == 168

    def exact_search(
        matrix: tuple[int, ...], *, operation_limit: int | None = None
    ) -> tuple[tuple[int, int], ...] | None:
        assert len(matrix) == 3
        permutation = tuple(
            sum(
                ((row & bits).bit_count() % 2) << index
                for index, row in enumerate(matrix)
            )
            for bits in range(8)
        )
        result = shortest[permutation]
        return (
            None
            if operation_limit is not None and len(result) >= operation_limit
            else result
        )

    source = CircuitIR(3, _relay())
    with patch.object(linear, "_pmh_synthesize", exact_search):
        result = compile(
            source,
            optimization_level=0,
            passes={"example.linear": linear.resynthesize_cnot_blocks},
        )
    assert result.instructions == (Instruction("cx", (0, 2)),)
