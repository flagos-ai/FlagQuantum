"""Single-rank Adam-to-L-BFGS parity with an analytic PyTorch objective."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.executors.mps.training_engine import train_distributed_mps

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]


def test_adam_lbfgs_matches_analytic_single_parameter_training(tmp_path: Path) -> None:
    theta = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    reference = theta.detach().clone().requires_grad_(True)
    adam = torch.optim.Adam([reference], lr=0.02)
    lbfgs = torch.optim.LBFGS([reference], lr=0.2, max_iter=1)
    expected_losses = []
    expected_gradients = []
    for step in range(4):
        expected_losses.append(float(reference.detach().cos()))
        expected_gradients.append(float(-reference.detach().sin()))

        def closure() -> torch.Tensor:
            reference.grad = None
            loss = reference.cos()
            loss.backward()
            return loss

        if step == 0:
            closure()
            adam.step()
        else:
            lbfgs.step(closure)

    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, theta)
        result = train_distributed_mps(
            circuit,
            steps=4,
            observable={0: "z"},
            optimizer="adam_lbfgs",
            lr=0.02,
            lbfgs_start_step=1,
            lbfgs_lr=0.2,
            device="cpu",
            compile_site_kernels=False,
            record_parameter_gradients=True,
        )
    finally:
        dist.destroy_process_group()

    assert result.completed_steps == 4
    assert [step.optimizer_stage for step in result.steps] == [
        "adam",
        "lbfgs",
        "lbfgs",
        "lbfgs",
    ]
    assert result.losses == pytest.approx(expected_losses, abs=1e-9)
    for step, expected_gradient in zip(result.steps, expected_gradients):
        assert len(step.parameter_gradients) == 1
        index, gradient = step.parameter_gradients[0]
        assert index == 0
        assert gradient == pytest.approx(expected_gradient, abs=1e-9)
    torch.testing.assert_close(theta, reference, atol=1e-9, rtol=1e-9)
