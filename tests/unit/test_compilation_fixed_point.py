import pytest
import torch

from flagquantum.compilation.compiler import simple_compile
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_rotation_cancellation_exposes_self_inverse_pair_next_round() -> None:
    ir = CircuitIR(
        1,
        (
            Instruction("x", (0,)),
            Instruction("rz", (0,), params={"theta": 0.2}),
            Instruction("rz", (0,), params={"theta": -0.2}),
            Instruction("x", (0,)),
        ),
    )

    assert simple_compile(ir).instructions == ()


def test_nested_rewrites_converge_across_multiple_pass_families() -> None:
    ir = CircuitIR(
        1,
        (
            Instruction("x", (0,)),
            Instruction("rz", (0,), params={"theta": 0.2}),
            Instruction("h", (0,)),
            Instruction("h", (0,)),
            Instruction("rz", (0,), params={"theta": -0.2}),
            Instruction("x", (0,)),
        ),
    )

    assert simple_compile(ir).instructions == ()


def test_fixed_point_does_not_fold_currently_cancelling_trainable_values() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    phi = torch.tensor(-0.2, requires_grad=True)
    ir = CircuitIR(
        1,
        (
            Instruction("x", (0,)),
            Instruction("rz", (0,), params={"theta": theta}),
            Instruction("rz", (0,), params={"theta": phi}),
            Instruction("x", (0,)),
        ),
    )

    compiled = simple_compile(ir)
    merged = compiled.instructions[1].params["theta"]
    merged.backward()

    assert tuple(item.name for item in compiled) == ("x", "rz", "x")
    assert theta.grad == pytest.approx(1.0)
    assert phi.grad == pytest.approx(1.0)
