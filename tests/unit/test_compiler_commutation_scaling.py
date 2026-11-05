"""Long commuting blocks retain the exact ungrouped partition and counters."""

import random
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.commutation import (
    analyze_commutation,
    commute,
)
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def ungrouped(
    ir: CircuitIR,
) -> tuple[
    dict[int, tuple[tuple[int, ...], ...]], dict[tuple[int, int], int], int, int
]:
    blocks_by_qubit = {}
    indices = {}
    queried = commuting = 0
    for qubit in range(ir.n_wires):
        blocks = []
        opened = []
        for position, gate in enumerate(ir.instructions):
            if qubit not in gate.wires:
                continue
            accepted = True
            for member in opened:
                queried += 1
                if commute(ir.instructions[member], gate):
                    commuting += 1
                else:
                    accepted = False
                    break
            if not accepted:
                blocks.append(tuple(opened))
                opened = []
            opened.append(position)
            indices[position, qubit] = len(blocks)
        if opened:
            blocks.append(tuple(opened))
            blocks_by_qubit[qubit] = tuple(blocks)
    return blocks_by_qubit, indices, queried, commuting


@pytest.mark.parametrize("seed", range(8))
def test_parameter_blind_grouping_preserves_exact_sweep(seed: int) -> None:
    rng = random.Random(seed)
    choices: tuple[Callable[[], Instruction], ...] = (
        lambda: Instruction("rz", (rng.randrange(3),), {"theta": rng.random()}),
        lambda: Instruction("rx", (rng.randrange(3),), {"theta": rng.random()}),
        lambda: Instruction("cx", (0, 1)),
        lambda: Instruction("cz", (1, 2)),
        lambda: Instruction(
            "u3",
            (1,),
            {"theta": rng.random(), "phi": rng.random(), "lbd": rng.random()},
        ),
        lambda: Instruction("h", (1,), matrix=torch.eye(2, dtype=torch.complex128)),
        lambda: Instruction("x", (2,)),
    )
    ir = CircuitIR(3, tuple(rng.choice(choices)() for _ in range(200)))
    expected = ungrouped(ir)
    actual = analyze_commutation(ir)
    assert (
        dict(actual.blocks_by_qubit),
        dict(actual.block_index),
        actual.queried_pairs,
        actual.commuting_pairs,
    ) == expected


def test_large_diagonal_run_avoids_quadratic_predicate_evaluation() -> None:
    count = 10000
    ir = CircuitIR(
        2,
        tuple(
            (
                Instruction("rz", (0,), {"theta": i / 1000})
                if i % 2
                else Instruction("cz", (0, 1))
            )
            for i in range(count)
        ),
    )
    with patch("flagquantum.compiler.commutation.commute", wraps=commute) as predicate:
        result = analyze_commutation(ir)
    assert predicate.call_count < 3 * count
    assert result.blocks_by_qubit[0] == (tuple(range(count)),)
    assert (
        result.queried_pairs
        == count * (count - 1) // 2 + (count // 2) * (count // 2 - 1) // 2
    )
    assert result.commuting_pairs == result.queried_pairs
