from __future__ import annotations

import math

import pytest
import torch

from flagquantum import Circuit, Parameter
from flagquantum.algorithms import pauli_term
from flagquantum.simulation.numerics.double_single import DoubleSingleTensor
from flagquantum.runtime.backends.statevector.split_real_imag_autograd_optimizer import (
    double_single_sgd_step,
    initialize_split_real_imag_double_single_sgd,
    split_real_imag_double_single_sgd_step,
)

pytestmark = pytest.mark.unit


def test_double_single_sgd_retains_updates_below_one_float32_ulp() -> None:
    initial = torch.tensor(1.0, dtype=torch.float32)
    state = initialize_split_real_imag_double_single_sgd({"theta": initial})
    gradient = DoubleSingleTensor.from_float32(
        torch.tensor([2.0**-31], dtype=torch.float32)
    )
    learning_rate = torch.tensor(1.0, dtype=torch.float32)
    float32_parameter = initial.clone()
    for _ in range(64):
        state, _ = double_single_sgd_step(
            state,
            gradient,
            parameter_order=("theta",),
            learning_rate=learning_rate,
        )
        float32_parameter -= gradient.to_float32()[0]
    expected = 1.0 - 64.0 * 2.0**-31
    assert float32_parameter.item() == 1.0
    assert state.cpu_float64()[0].item() == pytest.approx(expected, abs=1e-14)
    assert state.parameters.low[0].item() != 0.0


def test_double_single_sgd_accepts_one_time_high_precision_encoding() -> None:
    master = torch.tensor(0.123456789012345, dtype=torch.float64)
    encoded = DoubleSingleTensor.from_float64(master)

    state = initialize_split_real_imag_double_single_sgd({"theta": encoded})

    assert state.device == encoded.high.device
    assert state.cpu_float64()[0].item() == pytest.approx(master.item(), abs=1e-14)


def test_split_real_imag_double_single_sgd_uses_explicit_p4_gradient() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1).ry(0, theta=theta)
    state = initialize_split_real_imag_double_single_sgd(
        {"theta": torch.tensor(0.23, dtype=torch.float32)}
    )
    result = split_real_imag_double_single_sgd_step(
        circuit,
        pauli_term(1.0, "Z", (0,)),
        state,
        learning_rate=torch.tensor(0.1, dtype=torch.float32),
        preflight=False,
    )
    expected = 0.23 + 0.1 * math.sin(0.23)
    assert result.state.step == 1
    assert result.state.cpu_float64()[0].item() == pytest.approx(expected, abs=1e-7)
    assert result.summary()["tensor_grad_used"] is False
    assert result.summary()["gradient_representation"] == "double_single_high_low"
    assert result.summary()["torch_optimizer_compatible"] is False
    assert result.summary()["native_cuda_evidence"] is True
    assert result.summary()["torch_fl_flagos_evidence"] is True
    assert result.summary()["accelerator_float64_tensor_materialized"] is False
    assert result.summary()["training_state_device_resident"] is True
    assert result.summary()["per_step_host_tensor_transfer"] is False


def test_double_single_sgd_rejects_precision_and_order_demotion() -> None:
    with pytest.raises(TypeError, match="torch.float32"):
        initialize_split_real_imag_double_single_sgd(
            {"theta": torch.tensor(0.2, dtype=torch.float64)}
        )
    state = initialize_split_real_imag_double_single_sgd(
        {"theta": torch.tensor(0.2, dtype=torch.float32)}
    )
    gradient = DoubleSingleTensor.from_float32(torch.tensor([0.1], dtype=torch.float32))
    with pytest.raises(ValueError, match="order"):
        double_single_sgd_step(
            state,
            gradient,
            parameter_order=("other",),
            learning_rate=torch.tensor(0.1, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        double_single_sgd_step(
            state,
            gradient,
            parameter_order=("theta",),
            learning_rate=torch.tensor(0.0, dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="device-resident"):
        double_single_sgd_step(
            state,
            gradient,
            parameter_order=("theta",),
            learning_rate=0.1,
        )
