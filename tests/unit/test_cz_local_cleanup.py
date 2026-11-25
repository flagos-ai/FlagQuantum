"""Independent complete-operator checks of Cirq-derived native local cleanup."""

from __future__ import annotations

import math
import random
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import _cz_blocks
from flagquantum.compiler.optimization._cz_local_cleanup import compact_native_cz_locals
from flagquantum.compiler.optimization.two_qubit_synthesis import synthesize_two_qubit
from flagquantum.core.ir import Instruction
from flagquantum.core.parameters import Parameter
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit


def _operator(
    instructions: tuple[Instruction, ...], pair: tuple[int, int] = (0, 1)
) -> torch.Tensor:
    result = torch.eye(4, dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    for instruction in instructions:
        size = 1 << len(instruction.wires)
        matrix = gate_matrix(
            instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(size, size)
        if len(instruction.wires) == 1:
            matrix = (
                torch.kron(matrix, identity)
                if instruction.wires[0] == pair[0]
                else torch.kron(identity, matrix)
            )
        result = matrix @ result
    return result


def _equivalent(
    source: tuple[Instruction, ...],
    output: tuple[Instruction, ...],
    pair: tuple[int, int],
) -> None:
    expected, actual = _operator(source, pair), _operator(output, pair)
    phase = torch.vdot(actual.flatten(), expected.flatten()) / 4
    assert float(abs(abs(phase) - 1)) < 1e-11
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("pair", [(0, 1), (4, 1)])
def test_generated_kak_locals_preserve_operator_and_reduce_cost(
    seed: int, pair: tuple[int, int]
) -> None:
    rng = random.Random(seed)
    words = tuple(
        Instruction(opcode, (wire,), {"theta": rng.uniform(-math.pi, math.pi)})
        for _ in range(4)
        for wire in pair
        for opcode in ("rz", "rx")
    )
    source = words[:8] + (Instruction("cz", pair[::-1]),) + words[8:]
    matrix = _operator(source, pair).tolist()
    generated = synthesize_two_qubit(
        matrix, qubits=pair, entangler="cz", z_rotation="rz", pulse_opcode="rx"
    )
    assert generated is not None
    output = compact_native_cz_locals(generated, pair)
    assert len(output) < len(generated)
    assert [item for item in output if item.name == "cz"] == [
        item for item in generated if item.name == "cz"
    ]
    assert all(item.name in {"rx", "rz", "cz"} for item in output)
    _equivalent(source, output, pair)


@pytest.mark.parametrize("middle", [0.0, math.pi, 0.37, 1e-12, math.pi - 1e-12])
def test_diagonal_antidiagonal_and_near_singular_runs(middle: float) -> None:
    source = (
        Instruction("rz", (0,), {"theta": 0.13}),
        Instruction("rx", (0,), {"theta": math.pi / 2}),
        Instruction("rz", (0,), {"theta": middle}),
        Instruction("rx", (0,), {"theta": -math.pi / 2}),
        Instruction("rz", (0,), {"theta": -0.29}),
    )
    result = compact_native_cz_locals(source, (0, 1))
    assert len(result) <= 3
    _equivalent(source, result, (0, 1))


def test_cz_order_and_single_gate_fast_path_are_retained() -> None:
    first = Instruction("cz", (1, 0))
    second = Instruction("cz", (0, 1))
    source = (
        Instruction("rx", (0,), {"theta": 0.2}),
        first,
        Instruction("rz", (1,), {"theta": 0.3}),
        second,
    )
    assert compact_native_cz_locals(source, (0, 1)) is source
    assert compact_native_cz_locals((), (0, 1)) == ()


@pytest.mark.parametrize(
    "held",
    [
        Instruction("rx", (0,), {"theta": Parameter("theta")}),
        Instruction("rz", (0,), {"theta": torch.tensor(0.2, requires_grad=True)}),
        Instruction("rx", (0,), {"theta": torch.tensor([0.1, 0.2])}),
        Instruction("rz", (0,), {"theta": 0.2}, metadata={"conditions": ((0, 1),)}),
        Instruction("rx", (0,), {"theta": 0.2}, matrix=((1, 0), (0, 1))),
        Instruction("rx", (2,), {"theta": 0.2}),
        Instruction("sx", (0,)),
    ],
)
def test_unsupported_or_annotated_inputs_are_unchanged(held: Instruction) -> None:
    source = (
        Instruction("rz", (0,), {"theta": 0.1}),
        held,
        Instruction("rz", (0,), {"theta": 0.3}),
    )
    assert compact_native_cz_locals(source, (0, 1)) is source


def test_independent_block_verifier_rejects_corrupt_cleanup() -> None:
    block = (Instruction("swap", (0, 1)), Instruction("cx", (1, 0)))
    rejected: list[str] = []
    with patch.object(
        _cz_blocks,
        "compact_native_cz_locals",
        return_value=(Instruction("cz", (0, 1)),),
    ):
        assert (
            _cz_blocks._candidate(block, (0, 1), on_rejection=rejected.append) is None
        )
    assert rejected == ["residual_rejected"]


def test_raw_proposal_growth_is_checked_before_cleanup() -> None:
    block = (Instruction("swap", (0, 1)), Instruction("cx", (1, 0)))
    padded = (Instruction("rz", (0,), {"theta": 0.0}),) * 100
    with (
        patch.object(_cz_blocks, "synthesize_two_qubit", return_value=padded),
        patch.object(
            _cz_blocks, "compact_native_cz_locals", side_effect=AssertionError
        ),
    ):
        assert _cz_blocks._candidate(block, (0, 1)) is None
