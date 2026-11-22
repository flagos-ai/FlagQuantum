"""Long commuting blocks retain the exact ungrouped partition and counters."""

import random
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization.commutation import (
    analyze_commutation,
    commute,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


def ungrouped(
    ir: CircuitIR,
) -> tuple[
    dict[int, tuple[tuple[int, ...], ...]], dict[tuple[int, int], int], int, int
]:
    blocks_by_qubit: dict[int, tuple[tuple[int, ...], ...]] = {}
    indices: dict[tuple[int, int], int] = {}
    queried = commuting = 0
    for qubit in range(ir.n_wires):
        blocks: list[tuple[int, ...]] = []
        opened: list[int] = []
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
    with patch(
        "flagquantum.compiler.optimization.commutation.commute", wraps=commute
    ) as predicate:
        result = analyze_commutation(ir)
    assert predicate.call_count < 3 * count
    assert result.blocks_by_qubit[0] == (tuple(range(count)),)
    assert (
        result.queried_pairs
        == count * (count - 1) // 2 + (count // 2) * (count // 2 - 1) // 2
    )
    assert result.commuting_pairs == result.queried_pairs


def test_repeated_barriers_keep_exact_counters() -> None:
    ir = CircuitIR(
        1,
        tuple(Instruction(name, (0,)) for _ in range(100) for name in ("h", "x")),
    )
    expected = ungrouped(ir)
    result = analyze_commutation(ir)
    assert (
        dict(result.blocks_by_qubit),
        dict(result.block_index),
        result.queried_pairs,
        result.commuting_pairs,
    ) == expected


def test_eviction_preserves_first_failure_and_all_logical_pairs() -> None:
    ir = CircuitIR(
        25,
        tuple(
            instruction
            for _ in range(3)
            for instruction in (
                *(Instruction("cz", (0, wire)) for wire in range(1, 25)),
                Instruction("rx", (0,), {"theta": 0.37}),
                Instruction("cz", (1, 0)),
                Instruction("h", (0,), matrix=torch.eye(2)),
            )
        ),
    )
    result = analyze_commutation(ir)
    assert (
        dict(result.blocks_by_qubit),
        dict(result.block_index),
        result.queried_pairs,
        result.commuting_pairs,
    ) == ungrouped(ir)


def test_analysis_rechecks_mutable_parameters_on_each_call() -> None:
    gate = Instruction("h", (0,))
    ir = CircuitIR(1, (gate, Instruction("h", (0,))))
    assert analyze_commutation(ir).commuting_pairs == 1
    assert isinstance(gate.params, dict)
    gate.params["extra"] = 0.2
    assert analyze_commutation(ir).commuting_pairs == 0
    gate.params.clear()
    assert analyze_commutation(ir).commuting_pairs == 1


def test_symbolic_batched_and_matrix_override_classes_remain_distinct() -> None:
    angle = torch.nn.Parameter(torch.tensor([0.2, 0.7]))
    ir = CircuitIR(
        1,
        tuple(
            instruction
            for _ in range(8)
            for instruction in (
                Instruction("rz", (0,), {"theta": Parameter("theta")}),
                Instruction("rz", (0,), {"theta": angle}),
                Instruction("h", (0,), matrix=torch.eye(2)),
                Instruction("h", (0,)),
                Instruction("h", (0,), {"extra": 1}),
                Instruction("rx", (0,), {"theta": 0.3}),
            )
        ),
    )
    result = analyze_commutation(ir)
    assert (
        dict(result.blocks_by_qubit),
        dict(result.block_index),
        result.queried_pairs,
        result.commuting_pairs,
    ) == ungrouped(ir)
    assert angle.grad is None
    assert ir.instructions[1].params["theta"] is angle


def test_failure_does_not_retain_answers_for_the_next_analysis() -> None:
    ir = CircuitIR(1, (Instruction("h", (0,)), Instruction("x", (0,))))
    with (
        patch(
            "flagquantum.compiler.optimization.commutation.commute",
            side_effect=RuntimeError("predicate failure"),
        ),
        pytest.raises(RuntimeError, match="predicate failure"),
    ):
        analyze_commutation(ir)
    with patch(
        "flagquantum.compiler.optimization.commutation.commute", wraps=commute
    ) as predicate:
        assert analyze_commutation(ir).queried_pairs == 1
    assert predicate.call_count == 1
