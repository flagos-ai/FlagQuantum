import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import optimize
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_zero_initialized_trainable_rotation_is_not_removed() -> None:
    theta = torch.tensor(0.0, requires_grad=True)
    ir = CircuitIR(
        1,
        (Instruction("ry", (0,), params={"theta": theta}),),
    )

    compiled = optimize(ir)

    assert compiled.instructions == ir.instructions
    assert compiled.instructions[0].params["theta"] is theta


def test_cancelling_trainable_rotations_keep_parameter_gradients() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    phi = torch.tensor(-0.2, requires_grad=True)
    ir = CircuitIR(
        1,
        (
            Instruction("ry", (0,), params={"theta": theta}),
            Instruction("ry", (0,), params={"theta": phi}),
        ),
    )

    compiled = optimize(ir)
    merged = compiled.instructions[0].params["theta"]
    merged.backward()

    assert len(compiled.instructions) == 1
    assert merged.requires_grad
    assert theta.grad == pytest.approx(1.0)
    assert phi.grad == pytest.approx(1.0)


def test_constant_zero_rotation_is_still_removed() -> None:
    ir = CircuitIR(
        1,
        (Instruction("ry", (0,), params={"theta": torch.tensor(0.0)}),),
    )

    assert optimize(ir).instructions == ()


def test_compiled_zero_initialized_rotation_remains_trainable_end_to_end() -> None:
    theta = torch.tensor(0.0, requires_grad=True)
    compiled = fq.Circuit(1).ry(0, theta=theta).compile()

    compiled.state().reshape(-1)[1].real.backward()

    assert theta.grad == pytest.approx(0.5)
