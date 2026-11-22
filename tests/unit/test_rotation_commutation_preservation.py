"""Rotation merges retain exact scoped partitions without skipping fixed points."""

from __future__ import annotations

import random
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile
from flagquantum.compiler.optimization import commutation
from flagquantum.compiler.optimization.canonicalization import _ROTATION_PARAM
from flagquantum.compiler.optimization.commutation_cancellation import (
    merge_commuting_rotations,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter
from tests.unit.test_native_block_optimization import _operator

pytestmark = pytest.mark.unit


def _program(angle: object = 0.3) -> CircuitIR:
    return CircuitIR(
        3,
        (
            Instruction("rz", (0,), {"theta": angle}),
            Instruction("cx", (0, 1)),
            Instruction("rz", (0,), {"theta": 0.4}),
            Instruction("h", (0,)),
            Instruction("cz", (1, 2)),
        ),
    )


def test_nonzero_merge_preserves_exact_partition_counters_and_operator() -> None:
    source = _program()
    with patch.object(
        commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
    ) as sweep:
        with commutation._reuse_commutation_analysis():
            actual = merge_commuting_rotations(source)
            updated = commutation.analyze_commutation(actual)
            assert sweep.call_count == 1
        expected = commutation.analyze_commutation(actual)
        assert sweep.call_count == 2
    assert updated == expected
    torch.testing.assert_close(_operator(actual), _operator(source), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "angle", [Parameter("theta"), torch.tensor(0.3, requires_grad=True)]
)
def test_symbolic_and_trainable_values_remain_owned_by_the_program(
    angle: object,
) -> None:
    with commutation._reuse_commutation_analysis():
        result = merge_commuting_rotations(_program(angle))
        updated = commutation.analyze_commutation(result)
    assert updated == commutation.analyze_commutation(result)
    total = result.instructions[0].params["theta"]
    if isinstance(angle, torch.Tensor):
        assert isinstance(total, torch.Tensor)
        (gradient,) = torch.autograd.grad(total, angle)
        torch.testing.assert_close(gradient, torch.tensor(1.0))
    else:
        assert "theta" in str(total)


def test_zero_sum_deletion_recomputes_instead_of_inheriting_old_boundaries() -> None:
    source = CircuitIR(
        1,
        (
            Instruction("x", (0,)),
            Instruction("rz", (0,), {"theta": 0.3}),
            Instruction("rz", (0,), {"theta": -0.3}),
            Instruction("x", (0,)),
        ),
    )
    with (
        patch.object(
            commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
        ) as sweep,
        commutation._reuse_commutation_analysis(),
    ):
        result = merge_commuting_rotations(source)
        updated = commutation.analyze_commutation(result)
        assert sweep.call_count == 2
    assert updated == commutation.analyze_commutation(result)
    assert updated.blocks_by_qubit[0] == ((0, 1),)


def test_retained_gate_mutation_invalidates_the_remapped_entry() -> None:
    with commutation._reuse_commutation_analysis():
        result = merge_commuting_rotations(_program())
        previous = commutation.analyze_commutation(result)
        changed = replace(
            result,
            instructions=(replace(result.instructions[0], matrix=torch.eye(2)),)
            + result.instructions[1:],
        )
        actual = commutation.analyze_commutation(changed)
        assert actual is not previous
    assert actual == commutation.analyze_commutation(changed)


def test_random_rotations_barriers_and_shared_controls_match_fresh_analysis() -> None:
    rng = random.Random(92713)
    choices = (
        Instruction("h", (0,)),
        Instruction("x", (1,)),
        Instruction("cx", (0, 1)),
        Instruction("cx", (0, 2)),
        Instruction("cy", (2, 1)),
        Instruction("cz", (1, 2)),
        Instruction("fence", (0, 1, 2), matrix=torch.eye(8)),
        Instruction("amplitude_damping", (1,), {"gamma": 0.2}),
        Instruction("rz", (0,), {"theta": 0.3}),
        Instruction("rz", (0,), {"theta": -0.3}),
        Instruction("rx", (1,), {"theta": 0.4}),
        Instruction("ry", (2,), {"theta": 0.2}),
    )
    changed = 0
    for _ in range(300):
        source = CircuitIR(3, tuple(rng.choice(choices) for _ in range(48)))
        with commutation._reuse_commutation_analysis():
            result = merge_commuting_rotations(source)
            actual = commutation.analyze_commutation(result)
        expected = commutation.analyze_commutation(result)
        assert actual == expected
        changed += result.instructions != source.instructions
    assert changed > 100


def test_standalone_merge_does_not_publish_an_analysis_to_the_next_call() -> None:
    with patch.object(
        commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
    ) as sweep:
        result = merge_commuting_rotations(_program())
        commutation.analyze_commutation(result)
        assert sweep.call_count == 2


@pytest.mark.parametrize("opcode", tuple(_ROTATION_PARAM))
@pytest.mark.parametrize("reversed_wires", (False, True))
def test_every_rotation_family_and_operand_order_matches_recomputation(
    opcode: str, reversed_wires: bool
) -> None:
    arity = OPERATOR_SCHEMAS[opcode].arity
    assert isinstance(arity, int)
    wires = tuple(reversed(range(arity))) if reversed_wires else tuple(range(arity))
    source = CircuitIR(
        arity + 1,
        (
            Instruction(opcode, wires, {"theta": 0.2}),
            Instruction("h", (arity,)),
            Instruction(opcode, wires, {"theta": 0.3}),
        ),
    )
    with commutation._reuse_commutation_analysis():
        result = merge_commuting_rotations(source)
        actual = commutation.analyze_commutation(result)
    assert actual == commutation.analyze_commutation(result)
    torch.testing.assert_close(_operator(result), _operator(source), rtol=0, atol=1e-12)


def test_default_workflow_matches_replacement_and_independent_operator() -> None:
    source = fq.Circuit(2).rz(0, theta=0.2).cx(0, 1).rz(0, theta=0.3).h(0)
    actual = compile(source)
    with patch(
        "flagquantum.compiler.optimization.commutation_cancellation."
        "_preserve_commutation_after_rotation_merge"
    ):
        recomputed = compile(source)
    assert actual.to_dict() == recomputed.to_dict()
    torch.testing.assert_close(
        _operator(actual), _operator(source.to_ir()), rtol=0, atol=1e-12
    )
