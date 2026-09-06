import pytest
import torch

from flagquantum.compiler import optimize
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_self_inverse_gates_cancel_across_disjoint_wire_gate() -> None:
    middle = Instruction("h", (1,))
    ir = CircuitIR(
        2,
        (Instruction("x", (0,)), middle, Instruction("x", (0,))),
    )

    assert optimize(ir).instructions == (middle,)


def test_rotation_gates_merge_across_disjoint_wire_gate() -> None:
    middle = Instruction("h", (1,))
    ir = CircuitIR(
        2,
        (
            Instruction("rx", (0,), params={"theta": 0.1}),
            middle,
            Instruction("rx", (0,), params={"theta": 0.2}),
        ),
    )

    compiled = optimize(ir)

    assert tuple(item.name for item in compiled) == ("rx", "h")
    assert compiled.instructions[0].params["theta"] == pytest.approx(0.3)


def test_touching_gate_blocks_self_inverse_and_rotation_rewrites() -> None:
    self_inverse = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("x", (0,)),
        ),
    )
    rotations = CircuitIR(
        2,
        (
            Instruction("rx", (0,), params={"theta": 0.1}),
            Instruction("cx", (0, 1)),
            Instruction("rx", (0,), params={"theta": 0.2}),
        ),
    )

    assert optimize(self_inverse).instructions == self_inverse.instructions
    assert optimize(rotations).instructions == rotations.instructions


def test_wire_local_trainable_rotation_merge_preserves_both_gradients() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    phi = torch.tensor(-0.1, requires_grad=True)
    ir = CircuitIR(
        2,
        (
            Instruction("ry", (0,), params={"theta": theta}),
            Instruction("h", (1,)),
            Instruction("ry", (0,), params={"theta": phi}),
        ),
    )

    merged = optimize(ir).instructions[0].params["theta"]
    merged.backward()

    assert theta.grad == pytest.approx(1.0)
    assert phi.grad == pytest.approx(1.0)
