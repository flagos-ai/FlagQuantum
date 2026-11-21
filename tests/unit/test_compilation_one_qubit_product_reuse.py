"""Verify streaming local products against full operators and mutable inputs."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import one_qubit_optimization as local
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.test_compilation_target_cz_resynthesis import _operator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("seed", range(6))
def test_interleaved_runs_preserve_every_complex_amplitude(
    width: int, seed: int
) -> None:
    rng = random.Random(seed)
    instructions: list[Instruction] = []
    for position in range(36):
        wire = rng.randrange(width)
        if position % 7 == 6:
            target = (wire + 1 + rng.randrange(width - 1)) % width
            instructions.append(Instruction("cx", (wire, target)))
        else:
            opcode = rng.choice(("h", "x", "sx", "t", "ry", "rz"))
            params = {"theta": rng.uniform(-2, 2)} if opcode in {"ry", "rz"} else {}
            instructions.append(
                Instruction(opcode, (wire,), params, metadata={"source": position})
            )
    source = CircuitIR(width, tuple(instructions), metadata={"case": seed})
    result = local.collapse_one_qubit_runs(source)
    torch.testing.assert_close(_operator(result), _operator(source), atol=2e-14, rtol=0)
    assert result.metadata == source.metadata
    assert local.collapse_one_qubit_runs(result) == result
    assert len(result.instructions) <= len(source.instructions)


def test_each_named_source_matrix_is_read_once_before_run_reconstruction() -> None:
    source = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("h", (1,)),
            Instruction("h", (0,)),
            Instruction("h", (1,)),
            Instruction("cx", (1, 0)),
        ),
    )
    with patch.object(
        local, "_instruction_matrix", wraps=local._instruction_matrix
    ) as read:
        result = local.collapse_one_qubit_runs(source)
    assert result.instructions == (source.instructions[-1],)
    for instruction in source.instructions[:-1]:
        assert sum(call.args[0] is instruction for call in read.call_args_list) == 1


def test_singletons_do_not_accumulate_or_reconstruct_matrix_products() -> None:
    source = CircuitIR(
        2,
        tuple(
            gate
            for _ in range(30)
            for gate in (Instruction("h", (0,)), Instruction("cx", (0, 1)))
        ),
    )
    with patch.object(local, "_matmul", wraps=local._matmul) as multiply:
        result = local.collapse_one_qubit_runs(source)
    assert result == source
    multiply.assert_not_called()


@pytest.mark.parametrize("kind", ["trainable", "batch", "small"])
def test_unreadable_or_protected_rotation_closes_only_its_wire(kind: str) -> None:
    if kind == "trainable":
        angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    elif kind == "batch":
        angle = torch.tensor([0.2, 0.4], dtype=torch.float64)
    else:
        angle = torch.tensor(1e-14, dtype=torch.float64)
    protected = Instruction("ry", (0,), {"theta": angle}, metadata={"keep": True})
    source = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("h", (1,)),
            Instruction("h", (0,)),
            protected,
            Instruction("x", (0,)),
            Instruction("h", (1,)),
            Instruction("x", (0,)),
        ),
    )
    result = local.collapse_one_qubit_runs(source)
    assert result.instructions == (protected,)
    assert result.instructions[0].params["theta"] is angle
    if kind != "batch":
        torch.testing.assert_close(
            _operator(result), _operator(source), atol=2e-14, rtol=0
        )
    if kind == "trainable":
        expected = _operator(source)
        actual = _operator(result)
        expected_loss = expected.real.sum() + 0.3 * expected.imag.sum()
        actual_loss = actual.real.sum() + 0.3 * actual.imag.sum()
        expected_gradient = torch.autograd.grad(expected_loss, angle)[0]
        actual_gradient = torch.autograd.grad(actual_loss, angle)[0]
        assert expected_gradient.abs() > 0
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-14, rtol=0
        )


def test_dense_operand_changes_are_read_by_each_new_pass() -> None:
    operand = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    source = CircuitIR(
        1,
        (
            Instruction("h", (0,)),
            Instruction("any", (0,), matrix=operand),
            Instruction("h", (0,)),
        ),
    )
    before = local.collapse_one_qubit_runs(source)
    expected_before = _operator(source)
    torch.testing.assert_close(_operator(before), expected_before, atol=2e-14, rtol=0)
    operand.copy_(torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128))
    after = local.collapse_one_qubit_runs(source)
    torch.testing.assert_close(_operator(after), _operator(source), atol=2e-14, rtol=0)
    assert not torch.allclose(_operator(after), expected_before)
