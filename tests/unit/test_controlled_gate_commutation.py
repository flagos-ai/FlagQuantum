"""Controlled commutation preserves full operators and useful cancellation."""

from __future__ import annotations

import itertools

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile, optimize
from flagquantum.compiler.optimization.commutation import commute
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from tests.unit.test_compilation_commutation import _matrix
from tests.unit.test_native_block_optimization import _operator

pytestmark = pytest.mark.unit

_CONTROLLED = ("cx", "cy", "crx", "cry")


def _gate(name: str, wires: tuple[int, int], angle: float = 0.37) -> Instruction:
    return Instruction(name, wires, {"theta": angle} if name.startswith("cr") else {})


@pytest.mark.parametrize("names", tuple(itertools.product(_CONTROLLED, repeat=2)))
def test_common_control_with_distinct_targets_commutes(
    names: tuple[str, str],
) -> None:
    left, right = _gate(names[0], (0, 1)), _gate(names[1], (0, 2), -0.83)
    assert commute(left, right) and commute(right, left)
    torch.testing.assert_close(
        _matrix(left, 3) @ _matrix(right, 3),
        _matrix(right, 3) @ _matrix(left, 3),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "names",
    [
        (a, b)
        for family in (("cx", "crx"), ("cy", "cry"))
        for a, b in itertools.product(family, repeat=2)
    ],
)
def test_common_target_with_one_pauli_axis_commutes(
    names: tuple[str, str],
) -> None:
    left, right = _gate(names[0], (0, 1)), _gate(names[1], (2, 1), -0.83)
    assert commute(left, right) and commute(right, left)
    torch.testing.assert_close(
        _matrix(left, 3) @ _matrix(right, 3),
        _matrix(right, 3) @ _matrix(left, 3),
        atol=1e-12,
        rtol=0,
    )


def test_all_controlled_placements_and_angles_match_independent_operators() -> None:
    placements = tuple(itertools.permutations(range(3), 2))
    claimed = declined_noncommuting = 0
    for first, second, wires_a, wires_b, angle in itertools.product(
        _CONTROLLED, _CONTROLLED, placements, placements, (-1.13, 0.29, 2.07)
    ):
        left, right = _gate(first, wires_a, angle), _gate(second, wires_b, -0.71)
        a, b = _matrix(left, 3), _matrix(right, 3)
        equal = bool(torch.allclose(a @ b, b @ a, atol=1e-12, rtol=0))
        decision = commute(left, right)
        assert decision == commute(right, left)
        assert decision == equal, (first, wires_a, second, wires_b, angle)
        claimed += decision
        declined_noncommuting += not equal
    assert claimed > 0 and declined_noncommuting > 0


@pytest.mark.parametrize("wires", ((1, 2), (1, 0)))
def test_control_target_crossings_block_reordering(wires: tuple[int, int]) -> None:
    left, right = _gate("cx", (0, 1)), _gate("crx", wires)
    assert not commute(left, right) and not commute(right, left)


def test_shared_target_with_distinct_axes_blocks_reordering() -> None:
    assert not commute(_gate("cx", (0, 1)), _gate("cry", (2, 1)))


def test_matrix_override_does_not_inherit_the_opcode_proof() -> None:
    override = Instruction("cx", (0, 1), matrix=torch.eye(4))
    assert not commute(override, _gate("cx", (0, 2)))


@pytest.mark.parametrize("gap_wires", ((0, 2), (2, 1)))
def test_default_compilation_cancels_an_uncomputed_controlled_gate(
    gap_wires: tuple[int, int],
) -> None:
    source = fq.Circuit(3).cx(0, 1).cx(*gap_wires).cx(0, 1)
    result = compile(source)
    assert [(gate.name, gate.wires) for gate in result] == [("cx", gap_wires)]
    torch.testing.assert_close(
        _operator(result), _operator(source.to_ir()), atol=1e-12, rtol=0
    )


def test_default_compilation_keeps_noncommuting_uncomputation() -> None:
    source = fq.Circuit(3).cx(0, 1).cx(1, 2).cx(0, 1)
    result = compile(source)
    assert len(result.instructions) == 3
    torch.testing.assert_close(
        _operator(result), _operator(source.to_ir()), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "theta", [Parameter("theta"), torch.tensor([0.2, 0.5], requires_grad=True)]
)
def test_commuting_cancellation_keeps_symbolic_or_batched_parameter_identity(
    theta: object,
) -> None:
    source = CircuitIR(
        3,
        (
            _gate("cx", (0, 1)),
            Instruction("crx", (0, 2), {"theta": theta}),
            _gate("cx", (0, 1)),
        ),
    )
    result = optimize(source)
    assert len(result.instructions) == 1
    assert result.instructions[0].params["theta"] is theta
    if isinstance(theta, torch.Tensor):
        assert theta.requires_grad and theta.grad is None


def test_historical_attribution_substitutes_only_the_new_reach() -> None:
    from benchmarks.compiler_commutation_cancellation import (
        _before_controlled_commutation_extension,
    )
    from flagquantum.compiler.optimization import commutation

    current = commutation.commute
    placements = tuple(itertools.permutations(range(3), 2))
    with _before_controlled_commutation_extension():
        for first, second, a, b in itertools.product(
            _CONTROLLED, _CONTROLLED, placements, placements
        ):
            left, right = _gate(first, a), _gate(second, b)
            expected = a == b and (
                (first in {"cx", "crx"} and second in {"cx", "crx"})
                or (first in {"cy", "cry"} and second in {"cy", "cry"})
            )
            assert commutation.commute(left, right) == expected
        source = fq.Circuit(3).cx(0, 1).cx(0, 2).cx(0, 1)
        assert len(compile(source).instructions) == 3
        assert commutation.commute(_gate("cx", (0, 1)), _gate("crx", (0, 1)))
        assert not commutation.commute(_gate("cx", (0, 1)), _gate("cx", (1, 2)))
    assert commutation.commute is current
    assert len(compile(source).instructions) == 1
